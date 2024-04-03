import os
import argparse
from datetime import datetime
from loguru import logger
import yaml
import nibabel as nib
from datetime import datetime

import numpy as np
import wandb
import torch
import pytorch_lightning as pl
import torch.nn.functional as F
import matplotlib.pyplot as plt
from monai.metrics import DiceMetric
from monai.losses import DiceLoss, DiceCELoss

# Added this to solve problem with too many files open 
## Link here : https://github.com/pytorch/pytorch/issues/11201#issuecomment-421146936
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from losses import AdapWingLoss, SoftDiceLoss

from utils import dice_score, check_empty_patch, multiply_by_negative_one, plot_slices, create_nnunet_from_plans, print_data_types
from monai.networks.nets import UNet, BasicUNet, AttentionUnet

from monai.networks.layers import Norm


from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandShiftIntensityd,
    Spacingd,
    RandRotate90d,
    NormalizeIntensityd,
    RandCropByPosNegLabeld,
    BatchInverseTransform,
    RandAdjustContrastd,
    AsDiscreted,
    RandHistogramShiftd,
    ResizeWithPadOrCropd,
    EnsureTyped,
    RandLambdad,
    CropForegroundd,
    RandGaussianNoised, 
    ConcatItemsd
    )

from monai.utils import set_determinism
from monai.inferers import sliding_window_inference
import time
from monai.data import (DataLoader, CacheDataset, load_decathlon_datalist, decollate_batch)
from monai.transforms import (Compose, EnsureType, EnsureTyped, Invertd, SaveImage)

# Added this because of following warning received:
## You are using a CUDA device ('NVIDIA RTX A6000') that has Tensor Cores. To properly utilize them, you should set `torch.set_float32_matmul_precision('medium' | 'high')`
## which will trade-off precision for performance. For more details, 
## read https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
# torch.set_float32_matmul_precision('medium' | 'high')


def get_parser():
    """
    This function returns the parser for the command line arguments.
    """
    parser = argparse.ArgumentParser(description="Train a nnUNet model using monai")
    parser.add_argument("-c", "--config", help="Path to the config file (.yml file)", required=True)
    return parser


# create a "model"-agnostic class with PL to use different models
class Model(pl.LightningModule):
    def __init__(self, config, data_root, net, loss_function, optimizer_class, exp_id=None, results_path=None):
        super().__init__()
        self.cfg = config
        self.save_hyperparameters(ignore=['net', 'loss_function'])

        self.root = data_root
        self.net = net
        self.lr = config["lr"]
        self.loss_function = loss_function
        self.optimizer_class = optimizer_class
        self.save_exp_id = exp_id
        self.results_path = results_path

        self.best_val_dice, self.best_val_epoch = 0, 0
        self.best_val_loss = float("inf")

        # define cropping and padding dimensions
        # NOTE about patch sizes: nnUNet defines patches using the median size of the dataset as the reference
        # BUT, for SC images, this means a lot of context outside the spinal cord is included in the patches
        # which could be sub-optimal. 
        # On the other hand, ivadomed used a patch-size that's heavily padded along the R-L direction so that 
        # only the SC is in context. 
        self.spacing = config["spatial_size"]
        self.voxel_cropping_size = self.inference_roi_size = config["spatial_size"]

        # define post-processing transforms for validation, nothing fancy just making sure that it's a tensor (default)
        self.val_post_pred = self.val_post_label = Compose([EnsureType()])

        # define evaluation metric
        self.soft_dice_metric = dice_score

        # temp lists for storing outputs from training, validation, and testing
        self.train_step_outputs = []
        self.val_step_outputs = []
        self.test_step_outputs = []


    # --------------------------------
    # FORWARD PASS
    # --------------------------------
    def forward(self, x):
        
        out = self.net(x)  
        # # NOTE: MONAI's models only output the logits, not the output after the final activation function
        # # https://docs.monai.io/en/0.9.0/_modules/monai/networks/nets/unetr.html#UNETR.forward refers to the 
        # # UnetOutBlock (https://docs.monai.io/en/0.9.0/_modules/monai/networks/blocks/dynunet_block.html#UnetOutBlock) 
        # # as the final block applied to the input, which is just a convolutional layer with no activation function
        # # Hence, we are used Normalized ReLU to normalize the logits to the final output
        # normalized_out = F.relu(out) / F.relu(out).max() if bool(F.relu(out).max()) else F.relu(out)

        return out  # returns logits


    # --------------------------------
    # DATA PREPARATION
    # --------------------------------   
    def prepare_data(self):
        # set deterministic training for reproducibility
        set_determinism(seed=self.cfg["seed"])
        
        # define training and validation transforms
        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "sc", "label"], reader="NibabelReader"),
                EnsureChannelFirstd(keys=["image", "sc", "label"]),
                Orientationd(keys=["image", "sc", "label"], axcodes="RPI"),
                Spacingd(
                    keys=["image", "sc", "label"],
                    pixdim=self.cfg["pixdim"],
                    mode=(2, 1, 1),
                ),
                # CropForegroundd(keys=["image", "label"], source_key="label", margin=100),
                ResizeWithPadOrCropd(keys=["image", "sc", "label"], spatial_size=self.cfg["spatial_size"],),
                RandCropByPosNegLabeld(
                    keys=["image", "sc", "label"],
                    label_key="label",
                    spatial_size=self.cfg["spatial_size"],
                    pos=1,
                    neg=1,
                    num_samples=4,
                    image_key="image",
                    image_threshold=0,
                ),
                # Flips the image : left becomes right
                RandFlipd(
                    keys=["image", "sc", "label"],
                    spatial_axis=[0],
                    prob=0.2,
                ),
                # Flips the image : supperior becomes inferior
                RandFlipd(
                    keys=["image", "sc", "label"],
                    spatial_axis=[1],
                    prob=0.2,
                ),
                # Flips the image : anterior becomes posterior
                RandFlipd(
                    keys=["image","sc", "label"],
                    spatial_axis=[2],
                    prob=0.2,
                ),
                # RandAdjustContrastd(
                #     keys=["image"],
                #     prob=0.2,
                #     gamma=(0.5, 4.5),
                #     invert_image=True,
                # ),
                # we add the multiplication of the image by -1
                # RandLambdad(
                #     keys='image',
                #     func=multiply_by_negative_one,
                #     prob=0.2
                #     ),
                
                # Normalize the intensity of the image
                NormalizeIntensityd(
                    keys=["image"], 
                    nonzero=False, 
                    channel_wise=False
                ),
                # RandGaussianNoised(
                #     keys=["image"],
                #     prob=0.2,
                # ), 
                # RandShiftIntensityd(
                #     keys=["image"],
                #     offsets=0.1,
                #     prob=0.2,
                # ),
                # Concatenates the image and the sc
                ConcatItemsd(keys=["sc", "label"], name="outputs"),
                EnsureTyped(keys=["image", "outputs"]),
                # AsDiscreted(
                #     keys=["label"],
                #     num_classes=2,
                #     threshold_values=True,
                #     logit_thresh=0.2,
                # )
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "sc", "label"], reader="NibabelReader"),
                EnsureChannelFirstd(keys=["image", "sc", "label"]),
                Orientationd(keys=["image", "sc", "label"], axcodes="RPI"),
                Spacingd(
                    keys=["image", "sc", "label"],
                    pixdim=self.cfg["pixdim"],
                    mode=(2, 1, 1),
                ),
                # CropForegroundd(keys=["image", "label"], source_key="label", margin=100),
                ResizeWithPadOrCropd(keys=["image", "sc", "label"], spatial_size=self.cfg["spatial_size"],),
                # RandCropByPosNegLabeld(
                #     keys=["image", "label"],
                #     label_key="label",
                #     spatial_size=self.cfg["spatial_size"],
                #     pos=1,
                #     neg=1,
                #     num_samples=4,
                #     image_key="image",
                #     image_threshold=0,
                # ),
                
                # Normalize the intensity of the image
                NormalizeIntensityd(
                    keys=["image"], 
                    nonzero=False, 
                    channel_wise=False
                ),
                # Concatenates the image and the sc
                ConcatItemsd(keys=["sc", "label"], name="outputs"),
                EnsureTyped(keys=["image", "outputs"]),
                # AsDiscreted(
                #     keys=["label"],
                #     num_classes=2,
                #     threshold_values=True,
                #     logit_thresh=0.2,
                # )
            ]
        )
        
        # load the dataset
        dataset = self.cfg["data"]
        logger.info(f"Loading dataset: {dataset}")
        train_files = load_decathlon_datalist(dataset, True, "train")
        val_files = load_decathlon_datalist(dataset, True, "validation")
        test_files = load_decathlon_datalist(dataset, True, "test")
        
        train_cache_rate = 0.5
        self.train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=train_cache_rate, num_workers=16)
        self.val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=0.25, num_workers=16)

        # define test transforms
        transforms_test = val_transforms
        
        # define post-processing transforms for testing; taken (with explanations) from 
        # https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/torch/unet_inference_dict.py#L66
        self.test_post_pred = Compose([
            EnsureTyped(keys=["pred", "label"]),
            Invertd(keys=["pred", "label"], transform=transforms_test, 
                    orig_keys=["image", "label"], 
                    meta_keys=["pred_meta_dict", "label_meta_dict"],
                    nearest_interp=False, to_tensor=True),
            ])
        self.test_ds = CacheDataset(data=test_files, transform=transforms_test, cache_rate=0.1, num_workers=4)


    # --------------------------------
    # DATA LOADERS
    # --------------------------------
    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.cfg["batch_size"], shuffle=True, num_workers=16, 
                            pin_memory=True, persistent_workers=True) 

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=1, shuffle=False, num_workers=16, pin_memory=True, 
                          persistent_workers=True)
    
    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    
    # --------------------------------
    # OPTIMIZATION
    # --------------------------------
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.cfg["weight_decay"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.cfg["max_iterations"])
        return [optimizer], [scheduler]


    # --------------------------------
    # TRAINING
    # --------------------------------
    def training_step(self, batch, batch_idx):

        inputs, labels = batch["image"], batch["outputs"]

        # # print(inputs.shape, labels.shape)
        # input_0 = inputs[0].detach().cpu().squeeze()
        # # print(input_0.shape)
        # label_0 = labels[0].detach().cpu().squeeze()

        # time_0 = datetime.now()

        # # save input 0 in a nifti file
        # input_0_nifti = nib.Nifti1Image(input_0.numpy(), affine=np.eye(4))
        # nib.save(input_0_nifti, f"~/ms_lesion_agnostic/temp/input_0_{time_0}.nii.gz")

        # # save label in a nifti file
        # label_nifti = nib.Nifti1Image(label_0.numpy(), affine=np.eye(4))
        # nib.save(label_nifti, f"~/ms_lesion_agnostic/temp/label_0_{time_0}.nii.gz")
        

        # # check if any label image patch is empty in the batch
        if check_empty_patch(labels) is None:
            # print(f"Empty label patch found. Skipping training step ...")
            return None

        output = self.forward(inputs)   # logits
        # print(f"labels.shape: {labels.shape} \t output.shape: {output.shape}")

        # get probabilities from logits
        output = F.relu(output) / F.relu(output).max() if bool(F.relu(output).max()) else F.relu(output)
        
        # calculate training loss   
        loss = self.loss_function(output, labels)

        # calculate train loss for the sc and the lesion
        loss_sc = self.loss_function(output[:, 0, ...], labels[:, 0, ...])
        loss_lesion = self.loss_function(output[:, 1, ...], labels[:, 1, ...])

        # calculate train dice
        # NOTE: this is done on patches (and not entire 3D volume) because SlidingWindowInference is not used here
        # So, take this dice score with a lot of salt
        train_soft_dice = self.soft_dice_metric(output, labels) 

        # calculate the dice for the sc and the lesion
        train_soft_dice_sc = self.soft_dice_metric(output[:, 0, ...], labels[:, 0, ...])
        train_soft_dice_lesion = self.soft_dice_metric(output[:, 1, ...], labels[:, 1, ...])

        metrics_dict = {
            "loss": loss.cpu(),
            "loss_sc": loss_sc.cpu(),
            "loss_lesion": loss_lesion.cpu(),
            "train_soft_dice": train_soft_dice.detach().cpu(),
            "train_soft_dice_sc": train_soft_dice_sc.detach().cpu(),
            "train_soft_dice_lesion": train_soft_dice_lesion.detach().cpu(),
            "train_number": len(inputs),
            "train_image": inputs[0].detach().cpu().squeeze(),
            "train_gt_sc": labels[0][0].detach().cpu().squeeze(),
            "train_gt_lesion": labels[0][1].detach().cpu().squeeze(),
            "train_pred_sc": output[0][0].detach().cpu().squeeze(),
            "train_pred_lesion": output[0][1].detach().cpu().squeeze(),

        }
        self.train_step_outputs.append(metrics_dict)

        return metrics_dict

    def on_train_epoch_end(self):

        if self.train_step_outputs == []:
            # means the training step was skipped because of empty input patch
            return None
        else:
            train_loss, train_soft_dice = 0, 0
            train_loss_sc, train_loss_lesion = 0, 0
            train_soft_dice_sc, train_soft_dice_lesion = 0, 0
            num_items = len(self.train_step_outputs)
            for output in self.train_step_outputs:
                train_loss += output["loss"].item()
                train_soft_dice += output["train_soft_dice"].item()
                train_loss_sc += output["loss_sc"].item()
                train_loss_lesion += output["loss_lesion"].item()
                train_soft_dice_sc += output["train_soft_dice_sc"].item()
                train_soft_dice_lesion += output["train_soft_dice_lesion"].item()
            
            mean_train_loss = (train_loss / num_items)
            mean_train_soft_dice = (train_soft_dice / num_items)
            mean_train_loss_sc = (train_loss_sc / num_items)
            mean_train_loss_lesion = (train_loss_lesion / num_items)
            mean_train_soft_dice_sc = (train_soft_dice_sc / num_items)
            mean_train_soft_dice_lesion = (train_soft_dice_lesion / num_items)

            wandb_logs = {
                "train_soft_dice": mean_train_soft_dice, 
                "train_loss": mean_train_loss,
                "train_loss_sc": mean_train_loss_sc,
                "train_loss_lesion": mean_train_loss_lesion,
                "train_soft_dice_sc": mean_train_soft_dice_sc,
                "train_soft_dice_lesion": mean_train_soft_dice_lesion
            }
            self.log_dict(wandb_logs)

            # plot the training images
            fig = plot_slices(image=self.train_step_outputs[0]["train_image"],
                              gt=self.train_step_outputs[0]["train_gt_lesion"],
                              pred=self.train_step_outputs[0]["train_pred_lesion"],
                              )
            wandb.log({"training images lesion": wandb.Image(fig)})
            plt.close(fig)

            # plot the training images
            fig2 = plot_slices(image=self.train_step_outputs[0]["train_image"],
                              gt=self.train_step_outputs[0]["train_gt_sc"],
                              pred=self.train_step_outputs[0]["train_pred_sc"],
                              )
            wandb.log({"training images sc": wandb.Image(fig2)})
            plt.close(fig2)

            # free up memory
            self.train_step_outputs.clear()
            wandb_logs.clear()
            


    # --------------------------------
    # VALIDATION
    # --------------------------------    
    def validation_step(self, batch, batch_idx):
        
        inputs, labels = batch["image"], batch["outputs"]

        # NOTE: this calculates the loss on the entire image after sliding window
        outputs = sliding_window_inference(inputs, self.inference_roi_size, mode="gaussian",
                                           sw_batch_size=4, predictor=self.forward, overlap=0.5,) 
        
        # get probabilities from logits
        outputs = F.relu(outputs) / F.relu(outputs).max() if bool(F.relu(outputs).max()) else F.relu(outputs)
        
        # calculate validation loss
        loss = self.loss_function(outputs, labels)
        
        
        # post-process for calculating the evaluation metric
        post_outputs = [self.val_post_pred(i) for i in decollate_batch(outputs)]
        post_labels = [self.val_post_label(i) for i in decollate_batch(labels)]
        val_soft_dice = self.soft_dice_metric(post_outputs[0], post_labels[0])

        hard_preds, hard_labels = (post_outputs[0].detach() > 0.5).float(), (post_labels[0].detach() > 0.5).float()
        val_hard_dice = self.soft_dice_metric(hard_preds, hard_labels)

        # NOTE: there was a massive memory leak when storing cuda tensors in this dict. Hence,
        # using .detach() to avoid storing the whole computation graph
        # Ref: https://discuss.pytorch.org/t/cuda-memory-leak-while-training/82855/2
        metrics_dict = {
            "val_loss": loss.detach().cpu(),
            "val_soft_dice": val_soft_dice.detach().cpu(),
            "val_hard_dice": val_hard_dice.detach().cpu(),
            "val_number": len(post_outputs),
            "val_image_0": inputs[0].detach().cpu().squeeze(),
            "val_gt_0": labels[0].detach().cpu().squeeze(),
            "val_pred_0": post_outputs[0].detach().cpu().squeeze(),
            # "val_image_1": inputs[1].detach().cpu().squeeze(),
            # "val_gt_1": labels[1].detach().cpu().squeeze(),
            # "val_pred_1": post_outputs[1].detach().cpu().squeeze(),
        }
        self.val_step_outputs.append(metrics_dict)
        
        return metrics_dict

    def on_validation_epoch_end(self):

        val_loss, num_items, val_soft_dice, val_hard_dice = 0, 0, 0, 0
        for output in self.val_step_outputs:
            val_loss += output["val_loss"].sum().item()
            val_soft_dice += output["val_soft_dice"].sum().item()
            val_hard_dice += output["val_hard_dice"].sum().item()
            num_items += output["val_number"]
        
        mean_val_loss = (val_loss / num_items)
        mean_val_soft_dice = (val_soft_dice / num_items)
        mean_val_hard_dice = (val_hard_dice / num_items)
                
        wandb_logs = {
            "val_soft_dice": mean_val_soft_dice,
            #"val_hard_dice": mean_val_hard_dice,
            "val_loss": mean_val_loss,
        }

        self.log_dict(wandb_logs)

        # save the best model based on validation dice score
        if mean_val_soft_dice > self.best_val_dice:
            self.best_val_dice = mean_val_soft_dice
            self.best_val_epoch = self.current_epoch
        
        # save the best model based on validation loss
        if mean_val_loss < self.best_val_loss:    
            self.best_val_loss = mean_val_loss
            self.best_val_epoch = self.current_epoch

        logger.info(
            f"\nCurrent epoch: {self.current_epoch}"
            f"\nAverage Soft Dice (VAL): {mean_val_soft_dice:.4f}"
            # f"\nAverage Hard Dice (VAL): {mean_val_hard_dice:.4f}"
            f"\nAverage DiceLoss (VAL): {mean_val_loss:.4f}"
            f"\nBest Average DiceLoss: {self.best_val_loss:.4f} at Epoch: {self.best_val_epoch}"
            f"\n----------------------------------------------------")

        # log on to wandb
        self.log_dict(wandb_logs)

        # # plot the validation images
        # fig0 = plot_slices(image=self.val_step_outputs[0]["val_image_0"],
        #                   gt=self.val_step_outputs[0]["val_gt_0"],
        #                   pred=self.val_step_outputs[0]["val_pred_0"],)
        # wandb.log({"validation images": wandb.Image(fig0)})
        # plt.close(fig0)
        

        # free up memory
        self.val_step_outputs.clear()
        wandb_logs.clear()


    # --------------------------------
    # TESTING
    # --------------------------------
    def test_step(self, batch, batch_idx):
        
        test_input = batch["inputs"]
        # print(batch["label_meta_dict"]["filename_or_obj"][0])
        batch["pred"] = sliding_window_inference(test_input, self.inference_roi_size, 
                                                 sw_batch_size=4, predictor=self.forward, overlap=0.5)

        # normalize the logits
        batch["pred"] = F.relu(batch["pred"]) / F.relu(batch["pred"]).max() if bool(F.relu(batch["pred"]).max()) else F.relu(batch["pred"])

        post_test_out = [self.test_post_pred(i) for i in decollate_batch(batch)]

        # make sure that the shapes of prediction and GT label are the same
        # print(f"pred shape: {post_test_out[0]['pred'].shape}, label shape: {post_test_out[0]['label'].shape}")
        assert post_test_out[0]['pred'].shape == post_test_out[0]['label'].shape
        
        pred, label = post_test_out[0]['pred'].cpu(), post_test_out[0]['label'].cpu()

        # NOTE: Important point from the SoftSeg paper - binarize predictions before computing metrics
        # calculate soft and hard dice here (for quick overview), other metrics can be computed from 
        # the saved predictions using ANIMA
        # 1. Dice Score
        test_soft_dice = self.soft_dice_metric(pred, label)

        # binarizing the predictions 
        pred = (post_test_out[0]['pred'].detach().cpu() > 0.5).float()
        label = (post_test_out[0]['label'].detach().cpu() > 0.5).float()

        # 1.1 Hard Dice Score
        test_hard_dice = self.soft_dice_metric(pred.numpy(), label.numpy())

        metrics_dict = {
            "test_hard_dice": test_hard_dice,
            "test_soft_dice": test_soft_dice,
        }
        self.test_step_outputs.append(metrics_dict)

        return metrics_dict

    def on_test_epoch_end(self):
        
        avg_hard_dice_test, std_hard_dice_test = np.stack([x["test_hard_dice"] for x in self.test_step_outputs]).mean(), \
                                                    np.stack([x["test_hard_dice"] for x in self.test_step_outputs]).std()
        avg_soft_dice_test, std_soft_dice_test = np.stack([x["test_soft_dice"] for x in self.test_step_outputs]).mean(), \
                                                    np.stack([x["test_soft_dice"] for x in self.test_step_outputs]).std()
        
        logger.info(f"Test (Soft) Dice: {avg_soft_dice_test}")
        logger.info(f"Test (Hard) Dice: {avg_hard_dice_test}")
        
        self.avg_test_dice, self.std_test_dice = avg_soft_dice_test, std_soft_dice_test
        self.avg_test_dice_hard, self.std_test_dice_hard = avg_hard_dice_test, std_hard_dice_test
        
        # free up memory
        self.test_step_outputs.clear()

# --------------------------------
# MAIN
# --------------------------------
def main():
    # get the parser
    parser = get_parser()
    args= parser.parse_args()

    # load config file
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Setting the seed
    pl.seed_everything(config["seed"], workers=True)

    # define root path for finding datalists
    dataset_root = config["data"]

    # define optimizer
    optimizer_class = torch.optim.Adam

    wandb.init(project=f'monai-unet-ms-lesion-seg-canproco', config=config)

    logger.info("Defining plans for nnUNet model ...")
    

    # define model
    # TODO: make the model deeper
    # net = UNet(
    #     spatial_dims=3,
    #     in_channels=1,
    #     out_channels=1,
    #     channels=config['unet_channels'],
    #     strides=config['unet_strides'],
    #     kernel_size=3,
    #     up_kernel_size=3,
    #     num_res_units=0,
    #     act='PRELU',
    #     norm=Norm.INSTANCE,
    #     dropout=0.0,
    #     bias=True,
    #     adn_ordering='NDA',
    # )
    # net=UNet(
    #     spatial_dims=3,
    #     in_channels=1,
    #     out_channels=1,
    #     channels=(32, 64, 128, 256),
    #     strides=(2, 2, 2 ),
        
    #     # dropout=0.1
    # )
    net = AttentionUnet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            channels=(32, 64, 128, 256, 512, 1024),
            strides=(2, 2, 2, 2, 2),
            dropout=0.1,
        )
    # net = BasicUNet(spatial_dims=3, features=(32, 64, 128, 256, 32), out_channels=1)

    # net = create_nnunet_from_plans()

    logger.add(os.path.join(config["log_path"], str(datetime.now()) + 'log.txt'), rotation="10 MB", level="INFO")


    # define loss function
    #loss_func = AdapWingLoss(theta=0.5, omega=8, alpha=2.1, epsilon=1, reduction="sum")
    # loss_func = DiceLoss(sigmoid=True, smooth_dr=1e-4)
    loss_func = DiceCELoss(sigmoid=True, smooth_dr=1e-4)
    # loss_func = SoftDiceLoss(smooth=1e-5)
    # NOTE: tried increasing omega and decreasing epsilon but results marginally worse than the above
    # loss_func = AdapWingLoss(theta=0.5, omega=12, alpha=2.1, epsilon=0.5, reduction="sum")
    #logger.info(f"Using AdapWingLoss with theta={loss_func.theta}, omega={loss_func.omega}, alpha={loss_func.alpha}, epsilon={loss_func.epsilon} ...")
    logger.info(f"Using SoftDiceLoss ...")
    # define callbacks
    early_stopping = pl.callbacks.EarlyStopping(
        monitor="val_loss", min_delta=0.00, 
        patience=config["early_stopping_patience"], 
        verbose=False, mode="min")

    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')

    # i.e. train by loading weights from scratch
    pl_model = Model(config, data_root=dataset_root,
                        optimizer_class=optimizer_class, loss_function=loss_func, net=net, 
                        exp_id="test", results_path=config["best_model_path"])
            
    # saving the best model based on validation loss
    checkpoint_callback_loss = pl.callbacks.ModelCheckpoint(
        dirpath=config["best_model_path"], filename='best_model', monitor='val_loss', 
        save_top_k=1, mode="min", save_last=True, save_weights_only=True)
    
    
    logger.info(f"Starting training from scratch ...")
    # wandb logger
    exp_logger = pl.loggers.WandbLogger(
                        name="test",
                        save_dir="/home/GRAMES.POLYMTL.CA/p119007/ms_lesion_agnostic/results",
                        group="test-on-canproco",
                        log_model=True, # save best model using checkpoint callback
                        project='ms-lesion-agnostic',
                        entity='pierre-louis-benveniste',
                        config=config)

    # Saving training script to wandb
    wandb.save("ms-lesion-agnostic/monai/nnunet/config_fake.yml")
    wandb.save("ms-lesion-agnostic/monai/nnunet/train_monai_unet_lightning_regionBased.py")


    # initialise Lightning's trainer.
    trainer = pl.Trainer(
        devices=1, accelerator="gpu",
        logger=exp_logger,
        callbacks=[checkpoint_callback_loss, lr_monitor, early_stopping],
        check_val_every_n_epoch=config["eval_num"],
        max_epochs=config["max_iterations"], 
        precision=32,
        # deterministic=True,
        enable_progress_bar=True) 
        # profiler="simple",)     # to profile the training time taken for each step

    # Train!
    trainer.fit(pl_model)
    logger.info(f" Training Done!") 
    
    # Closing wandb log
    wandb.finish()       


if __name__ == "__main__":
    main()