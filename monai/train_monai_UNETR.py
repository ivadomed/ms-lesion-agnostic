"""
This script is used to train a UNETR model.

It takes as input the config file (a JSON file) 

This script is inspired from : https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/unetr_btcv_segmentation_3d.ipynb

Args: 
    -c: path to the config file

Example:
    python train_monai_nnunet.py -c /path/to/nnunet/config.json

Pierre-Louis Benveniste
"""

import argparse
import json
import os
import sys
import monai
from tqdm import tqdm
import matplotlib.pyplot as plt
import yaml
import numpy as np
import wandb
import time
from loguru import logger

from monai.networks.layers import Norm

os.environ["PYTORCH_USE_CUDA_DSA"] = "1"


#Transforms import
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
    EnsureTyped
    )

# Dataset import
from monai.data import DataLoader, CacheDataset, load_decathlon_datalist, Dataset


# model import
import torch
from monai.networks.nets import UNETR, UNet
from monai.losses import DiceLoss

# For training and validation
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete




def get_parser():
    """
    This function returns the parser for the command line arguments.
    """
    parser = argparse.ArgumentParser(description="Train a nnUNet model using monai")
    parser.add_argument("-c", "--config", help="Path to the config file (.yml file)", required=True)
    return parser


# def validation(model, epoch_iterator_val, config, post_label, post_pred, dice_metric, global_step):
#     model.eval()
#     with torch.no_grad():
#         for batch in epoch_iterator_val:
#             val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
#             val_outputs = model(val_inputs)
#             dice_metric(y_pred=val_outputs, y=val_labels)
#             # val_outputs = sliding_window_inference(val_inputs, config["spatial_size"], 1, model)
#             # val_labels_list = decollate_batch(val_labels)
#             # val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
#             # val_outputs_list = decollate_batch(val_outputs)
#             # val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
#             # dice_metric(y_pred=val_output_convert, y=val_labels_convert)
#             epoch_iterator_val.set_description("Validate (%d / %d Steps)" % (global_step, 10.0))  # noqa: B038
#         # for batch in epoch_iterator_val:
#         #         val_inputs, val_labels = (
#         #             batch["image"].cuda(),
#         #             batch["label"].cuda(),
#         #         )
#         #         # TODO: parametrize this
#         #         roi_size = config["spatial_size"]
#         #         sw_batch_size = 4
#         #         val_outputs = sliding_window_inference(
#         #             val_inputs, roi_size, sw_batch_size, model)

#         #         val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
#         #         val_labels = [post_label(i) for i in decollate_batch(val_labels)]
#         #         # compute metric for current iteration
#         #         dice_metric(y_pred=val_outputs, y=val_labels)
            
#         mean_dice_val = dice_metric.aggregate().item()
#         print("Mean dice val: ", mean_dice_val)
#         dice_metric.reset()

#     return mean_dice_val

# def train(model, config, global_step, train_loader, dice_val_best, global_step_best, loss_function, optimizer, epoch_loss_values, metric_values, val_loader, post_label, post_pred, dice_metric):
#     model.train()
#     epoch_loss = 0
#     step = 0
#     epoch_iterator = tqdm(train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True)
#     for step, batch in enumerate(epoch_iterator):
#         step += 1
#         x, y = (batch["image"].cuda(), batch["label"].cuda())
#         logit_map = model(x)
#         loss = loss_function(logit_map, y)
#         loss.backward()
#         epoch_loss += loss.item()
#         optimizer.step()
#         optimizer.zero_grad()
#         epoch_iterator.set_description(  # noqa: B038
#             "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, config["max_iterations"], loss)
#         )
#         if (global_step % config["eval_num"] == 0 and global_step != 0) or global_step == config["max_iterations"]:
#             epoch_iterator_val = tqdm(val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True)
#             dice_val = validation(model, epoch_iterator_val, config, post_label, post_pred, dice_metric, global_step)
#             epoch_loss /= step
#             epoch_loss_values.append(epoch_loss)
#             metric_values.append(dice_val)
#             if dice_val > dice_val_best:
#                 dice_val_best = dice_val
#                 global_step_best = global_step
#                 torch.save(model.state_dict(), config["best_model_path"])
#                 print(
#                     "Model Was Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(dice_val_best, dice_val)
#                 )
#             else:
#                 print(
#                     "Model Was Not Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
#                         dice_val_best, dice_val
#                     )
#                 )
#         global_step += 1
#     return global_step, dice_val_best, global_step_best


def main():
    """
    Main function of the script.
    """

    # We get the parser and parse the arguments
    parser = get_parser()
    args = parser.parse_args()

    # We load the config file (a yml file)
    # load config file
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    

    ##### ------------------
    # Monai should be installed with pip install monai[all] (to get all readers)
    # We define the trasnformations for training and validation
    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"], reader="NibabelReader"),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RSP"),
            Spacingd(
                keys=["image", "label"],
                pixdim=config["pixdim"],
                mode=(2, 1),
            ),
            ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=config["spatial_size"],),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=config["spatial_size"],
                pos=1,
                neg=1,
                num_samples=4,
                image_key="image",
                image_threshold=0,
            ),
            # Flips the image : left becomes right
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[0],
                prob=0.2,
            ),
            # Flips the image : supperior becomes inferior
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[1],
                prob=0.2,
            ),
            # Flips the image : anterior becomes posterior
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[2],
                prob=0.2,
            ),
            RandAdjustContrastd(
                keys=["image"],
                prob=0.2,
                gamma=(0.5, 4.5),
                invert_image=True,
            ),
            NormalizeIntensityd(
                keys=["image", "label"], 
                nonzero=False, 
                channel_wise=False
            ),
            EnsureTyped(keys=["image", "label"]),
            AsDiscreted(
                keys=["label"],
                num_classes=2,
                threshold_values=True,
                logit_thresh=0.2,
            )
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"], reader="NibabelReader"),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RSP"),
            Spacingd(
                keys=["image", "label"],
                pixdim=config["pixdim"],
                mode=(2, 1),
            ),
            ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=config["spatial_size"],),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=config["spatial_size"],
                pos=1,
                neg=1,
                num_samples=4,
                image_key="image",
                image_threshold=0,
            ),
            NormalizeIntensityd(
                keys=["image", "label"], 
                nonzero=False, 
                channel_wise=False
            ),
            EnsureTyped(keys=["image", "label"]),
            AsDiscreted(
                keys=["label"],
                num_classes=2,
                threshold_values=True,
                logit_thresh=0.2,
            )
        ]
    )

    # Path to data split (JSON file)
    data_split_json_path = config["data"]
    # We load the data lists
    with open(data_split_json_path, "r") as f:
        data = json.load(f)
    train_list = data["train"]
    val_list = data["validation"]
        
    # Path to the output directory
    output_dir = config["output_dir"]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # We load the train and validation data
    logger.info("Loading the training and validation data...")
    train_ds = CacheDataset(
                            data=train_list,
                            transform=train_transforms,
                            cache_rate=0.1,
                            num_workers=2
                            )
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    val_ds = CacheDataset(
                        data=val_list,
                        transform=val_transforms,
                        cache_rate=0.1,
                        num_workers=0,
                        )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)  

    # plot 3 image and save them
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, ax in enumerate(axes):
        img = train_ds[i][0]['image']
        ax.imshow(img[0, 7, :, :], cmap="gray")
        ax.set_title(f"Image {i+1}")
        ax.axis('on')
    plt.savefig(os.path.join(output_dir, "image.png"))



    print("Preparing the UNET model...")
    # we define the device to use
    device = torch.device("cuda:0")

    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        kernel_size=3,
        up_kernel_size=3,
        num_res_units=0,
        act='PRELU',
        norm=Norm.BATCH,
        dropout=0.0,
        bias=True,
        adn_ordering='NDA',
    ).to(device)

    loss_function = DiceLoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    torch.backends.cudnn.benchmark = True

    # initialize wandb
    wandb.init(project=config["experiment_name"], config=config)

    # 🐝 Log gen gradients of the models to wandb
    wandb.watch(model, log_freq=100)

    # 🐝 Add training script as an artifact
    artifact_script = wandb.Artifact(name='training', type='file')
    artifact_script.add_file(local_path=os.path.abspath(__file__), name=os.path.basename(__file__))
    wandb.log_artifact(artifact_script)

    epoch_loss_values = []
    step_loss_values = []
    val_loss_values = []
    best_val_loss = 1000.0

    for epoch in range(config["max_iterations"]):
        logger.info("-" * 10)
        logger.info(f"epoch {epoch + 1}/{config['max_iterations']}")
        model.train()
        epoch_loss = 0
        epoch_cl_loss = 0
        epoch_recon_loss = 0
        step = 0

        for batch_data in train_loader:
            step += 1
            start_time = time.time()

            inputs, gt_input = (
                batch_data["image"].to(device),
                batch_data["label"].to(device),
            )
        
            optimizer.zero_grad()
            output = model(inputs)

            loss = loss_function(output, gt_input)
            loss.detach().cpu()

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            step_loss_values.append(loss.item())


            end_time = time.time()
            logger.info(
                f"{step}/{len(train_list) // train_loader.batch_size}, "
                f"train_loss: {loss.item():.4f}, "
                f"time taken: {end_time-start_time}s"
            )

            wandb.log({"Training/loss": loss.item()})

        epoch_loss /= step

        epoch_loss_values.append(epoch_loss)
        
        # 🐝 log train_loss averaged over epoch to wandb
        wandb.log({"Training/loss_epoch": epoch_loss})


        
        logger.info(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if epoch % config["eval_num"] == 0:
            logger.info("Entering Validation for epoch: {}".format(epoch + 1))
            total_val_loss = 0
            val_step = 0
            model.eval()
            for val_batch in val_loader:
                val_step += 1
                start_time = time.time()
                inputs, gt_input = (
                    val_batch["image"].to(device),
                    val_batch["label"].to(device),
                )
                outputs = model(inputs)
                val_loss = loss_function(outputs, gt_input)
                total_val_loss += val_loss.item()
                end_time = time.time()

            total_val_loss /= val_step
            val_loss_values.append(total_val_loss)

            wandb.log({"Validation loss": total_val_loss})

            logger.info(f"epoch {epoch + 1} Validation avg loss: {total_val_loss:.4f}, " f"time taken: {end_time-start_time}s")

            if total_val_loss < best_val_loss:
                logger.info(f"Saving new model based on validation loss {total_val_loss:.4f}")
                best_val_loss = total_val_loss
                checkpoint = {"epoch": config["max_iterations"], "state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
                torch.save(checkpoint, config["best_model_path"])

    print("Done")


    # # We then train the model
    # post_label = AsDiscrete(to_onehot=2)
    # post_pred = AsDiscrete(argmax=True, to_onehot=2)
    # dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
    # global_step = 0
    # dice_val_best = 0.0
    # global_step_best = 0
    # epoch_loss_values = []
    # metric_values = []
    # while global_step < config["max_iterations"]:
    #     global_step, dice_val_best, global_step_best = train(model, config, global_step, train_loader, dice_val_best, global_step_best,
    #                                                           loss_function, optimizer, epoch_loss_values, metric_values, val_loader, post_label, post_pred, dice_metric)
    # model.load_state_dict(torch.load(config["best_model_path"]))

    # print(f"train completed, best_metric: {dice_val_best:.4f} " f"at iteration: {global_step_best}")


if __name__ == "__main__":
    main()







