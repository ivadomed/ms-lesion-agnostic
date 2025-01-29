"""
This script trains a model to segment MS lesion from MRI images.
The training is done using a Mean Teacher model, which is a semi-supervised learning method based on consistency training.
Code was inspired from: https://github.com/yulequan/UA-MT and https://github.com/perone/mean-teacher

Args:
    --msd-data: path to the MSD dataset
    --output-dir: path to the output directory

Returns:
    None

Example:
    $ python meanTeacher/train.py --msd-data data/MSD --output-dir output

Author: Pierre-Louis Benveniste
"""


import os
import torch
import argparse
from loguru import logger
from monai.data import load_decathlon_datalist, CacheDataset, DataLoader
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd, NormalizeIntensityd, RandCropByPosNegLabeld, ResizeWithPadOrCropd, RandFlipd, Rand3DElasticd, RandAffined, RandGaussianNoised, RandSimulateLowResolutiond, RandBiasFieldd
from monai.networks.nets import AttentionUnet
from monai.losses import DiceCELoss
from tqdm import tqdm
import torch.nn.functional as F
from monai.inferers import sliding_window_inference
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system') # Because of this: https://github.com/Project-MONAI/MONAI/issues/701#issuecomment-663887104


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--msd-data', type=str, default='data/MSD', help='path to the MSD dataset')
    parser.add_argument('--output-dir', type=str, default='output', help='path to the output directory')
    return parser.parse_args()

def train_transforms(data):
    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"], reader="NibabelReader"),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RPI"),
            Spacingd(keys=["image", "label"],pixdim=[0.7, 0.7, 0.7],mode=(2, 0)),
            ResizeWithPadOrCropd(keys=["image", "label"],spatial_size=[64, 128, 128]),
            NormalizeIntensityd(keys=["image"], nonzero=False, channel_wise=False),   
        ]
    )
    return train_transforms(data)

def val_transforms(data):
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"], reader="NibabelReader"),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RPI"),
            Spacingd(keys=["image", "label"],pixdim=[0.7, 0.7, 0.7],mode=(2, 0)),
            ResizeWithPadOrCropd(keys=["image", "label"],spatial_size=[64, 128, 128]),
            NormalizeIntensityd(keys=["image"], nonzero=False, channel_wise=False),   
        ]
    )
    return val_transforms(data)


def update_teacher_weights(studentNet, TeacherNet, alpha, epoch):
    """
    This function updates the weights of the teacher model using the student model.
    It uses the exponential moving average of the student model weights to update the teacher model weights.
    """
    alpha = min(1 - 1 / (epoch + 1), alpha)
    for teacher_param, student_param in zip(TeacherNet.parameters(), studentNet.parameters()):
        # teacher_param.data.mul_(alpha).add_(1 - alpha, student_param.data)
        teacher_param.data.mul_(alpha).add_(student_param.data, alpha=1 - alpha) # Changed because of this: https://github.com/clovaai/AdamP/issues/5
    

def main():
    # Parse the arguments
    args = arg_parser()
    msd_data = args.msd_data
    output_dir = args.output_dir

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the data
    logger.info(f'Loading data from {msd_data}')
    train_files = load_decathlon_datalist(msd_data, True, "train")
    val_files = load_decathlon_datalist(msd_data, True, "validation")
    test_files = load_decathlon_datalist(msd_data, True, "test")

    train_cache_rate = 0.5
    val_cache_rate = 0.25
    train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=train_cache_rate, num_workers=8)
    val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=val_cache_rate, num_workers=8)

    # Create the dataloaders
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, persistent_workers=False)

    # Create the models
    studentNet = AttentionUnet(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            channels=[32, 64, 128, 256, 512],
            strides=[2, 2, 2, 2, 2],
            dropout=0.1,
    ).to(device)
    TeacherNet = AttentionUnet(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            channels=[32, 64, 128, 256, 512],
            strides=[2, 2, 2, 2, 2],
            dropout=0.1,
    ).to(device)

    # Create the optimizer
    optimizer = torch.optim.AdamW(studentNet.parameters(), lr=0.0001, weight_decay=0.00001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=250)

    # Create the loss function
    supervised_loss = DiceCELoss(sigmoid=False, smooth_dr=1e-4)
    consistency_loss = F.cross_entropy

    # Initialize the storage of training results
    train_sup_losses = []
    train_cons_losses = []
    train_total_losses = []
    val_sup_losses = []
    val_cons_losses = []
    val_total_losses = []

    # Train the model
    logger.info('Training the model')
    # Iterate over the epochs
    for epoch in range(13):
        logger.info(f'Epoch {epoch}')
        studentNet.train()
        TeacherNet.train()
        epoch_sup_loss = 0
        epoch_cons_loss = 0
        epoch_total_loss = 0
        step = 0
        epoch_iterator = tqdm(train_loader, dynamic_ncols=True)

        for step, batch in enumerate(epoch_iterator):
            step += 1
            inputs, labels = (batch["image"].to(device), batch["label"].to(device))

            # ------------ For the student model ------------
            logit_map_student = studentNet(inputs)
            # Get probabilities from logits
            output_student = F.relu(logit_map_student) / F.relu(logit_map_student).max() if bool(F.relu(logit_map_student).max()) else F.relu(logit_map_student)
            # Compute the supervised loss
            train_sup_loss = supervised_loss(output_student, labels)
            # Divide the loss by the number of elements in the batch
            train_sup_loss /= inputs.size(0)

            # ------------ For the teacher model ------------
            logit_map_teacher = TeacherNet(inputs)
            # Get probabilities from logits
            output_teacher = F.relu(logit_map_teacher) / F.relu(logit_map_teacher).max() if bool(F.relu(logit_map_teacher).max()) else F.relu(logit_map_teacher)
            # Compute the consistency loss
            train_cons_loss = consistency_loss(output_student, output_teacher)
            # Divide the loss by the number of elements in the batch
            train_cons_loss /= inputs.size(0)

            # Compute the total loss
            loss = 0.5*(train_sup_loss + train_cons_loss)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            # Add the losses for each step in the epoch
            epoch_sup_loss += train_sup_loss 
            epoch_cons_loss += train_cons_loss
            epoch_total_loss += loss

        # Aggregate the losses for the epoch
        epoch_sup_loss /= step
        epoch_cons_loss /= step
        epoch_total_loss /= step

        # Add the losses for the epoch in the storage
        train_sup_losses.append(epoch_sup_loss)
        train_cons_losses.append(epoch_cons_loss)
        train_total_losses.append(epoch_total_loss)
    
        # Report the epoch losses in the logger
        logger.info(f'Supervised loss: {epoch_sup_loss}, Consistency loss: {epoch_cons_loss}, Total loss: {epoch_total_loss}')

        # Update the Teacher model weights
        update_teacher_weights(studentNet, TeacherNet, 0.99, epoch) # based on https://github.com/perone/mean-teacher/blob/79bf9fc61f540491b79d3b3e60a213f798c3ea27/pytorch/main.py#L182

        # Evaluate the model
        if epoch % 10 == 0:
            studentNet.eval()
            TeacherNet.eval()
            with torch.no_grad():
                inputs, labels = batch["image"].to(device), batch["label"].to(device)

                # With the student model
                # NOTE: this calculates the loss on the entire image after sliding window
                logits_student = sliding_window_inference(inputs, [64, 128, 128], mode="gaussian",
                                                sw_batch_size=4, predictor=studentNet, overlap=0.5,) 
                # get probabilities from logits
                outputs_student = F.relu(logits_student) / F.relu(logits_student).max() if bool(F.relu(logits_student).max()) else F.relu(logits_student)
                # calculate supervised validation loss
                val_sup_loss = supervised_loss(outputs_student, labels)
                val_sup_losses.append(val_sup_loss.item())

                # With the teacher model
                logits_teacher = sliding_window_inference(inputs, [64, 128, 128], mode="gaussian",
                                                sw_batch_size=4, predictor=TeacherNet, overlap=0.5,)
                # get probabilities from logits
                outputs_teacher = F.relu(logits_teacher) / F.relu(logits_teacher).max() if bool(F.relu(logits_teacher).max()) else F.relu(logits_teacher)
                # calculate consistency validation loss
                val_cons_loss = consistency_loss(outputs_student, outputs_teacher)
                val_cons_losses.append(val_cons_loss.item())

                # calculate total validation loss
                val_total_loss = 0.5*(val_sup_loss + val_cons_loss)
                val_total_losses.append(val_total_loss.item())

                # Report the validation losses in the logger
                logger.info(f'Validation losses: Supervised loss: {val_sup_loss.item()}, Consistency loss: {val_cons_loss.item()}, Total loss: {val_total_loss.item()}')

                # If the total loss is the best, report the performances and save the teacher model
                if val_total_loss == min(val_total_losses):
                    logger.info(f'Best model found at epoch {epoch}')
                    # torch.save(TeacherNet.state_dict(), os.path.join(output_dir, 'best_model.pth'))


if __name__ == '__main__':
    main()