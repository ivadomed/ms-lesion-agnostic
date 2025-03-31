"""
This code is used to test the model on a test set.
It uses the class Model which was defined in the file train_monai_unet_lightning.py.
"""
import os
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    NormalizeIntensityd,
    ResizeWithPadOrCropd,
    Invertd,
    EnsureTyped,
    SaveImage,
)
from monai.data import (DataLoader, CacheDataset, load_decathlon_datalist, decollate_batch, Dataset)
from monai.networks.nets import AttentionUnet
import torch
from monai.inferers import sliding_window_inference
import torch.nn.functional as F
from utils.utils import dice_score
import argparse
import yaml
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import numpy as np


def get_parser():
    """
    This function returns the parser for the command line arguments.
    """
    parser = argparse.ArgumentParser(description="Test the model on the test set")
    parser.add_argument("-c", "--config", help="Path to the config file (.yml file)", required=True)
    parser.add_argument("--data-split", help="Data split to use (train, validation, test)", required=True, type=str)
    return parser


def main():
    """
    This function is used to test the model on a test set.

    Args:
        None
    
    Returns:
        None
    """
    # Get the parser
    parser = get_parser()
    args = parser.parse_args()
    
    # Load the config file
    with open(args.config, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    # Device
    DEVICE = "cuda"

    # build output directory
    output_dir = os.path.join(cfg["output_dir"], args.data_split +"_set")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Dict of dice score
    dice_scores = {}
    dice_scores_0_01 = {}
    dice_scores_0_02 = {}
    dice_scores_0_05 = {}
    dice_scores_0_1 = {}
    dice_scores_0_2 = {}
    dice_scores_0_3 = {}
    dice_scores_0_4 = {}
    dice_scores_0_5 = {}
    dice_scores_0_6 = {}
    dice_scores_0_7 = {}
    dice_scores_0_8 = {}
    dice_scores_0_9 = {}
    dice_scores_0_95 = {}
    dice_scores_0_98 = {}
    dice_scores_0_99 = {}


    # Load the data
    test_files = load_decathlon_datalist(cfg["dataset"], True, args.data_split)

    #Create the test transforms
    test_transforms = Compose(
                [
                    LoadImaged(keys=["image", "label"], reader="NibabelReader"),
                    EnsureChannelFirstd(keys=["image", "label"]),
                    Orientationd(keys=["image", "label"], axcodes="RPI"),
                    Spacingd(
                        keys=["image", "label"],
                        pixdim=cfg["pixdim"],
                        mode=(2, 0),
                    ),
                    NormalizeIntensityd(
                        keys=["image"], 
                        nonzero=False, 
                        channel_wise=False
                    ),
                    # ResizeWithPadOrCropd(
                    #     keys=["image", "label"],
                    #     spatial_size=cfg["spatial_size"],
                    # ),
                ]
            )

    # Create the prediction post-processing function
    ## For this to work I had to add cupy-cuda117==10.6.0 to the requirements
    test_post_pred = Compose([
            EnsureTyped(keys=["pred"]),
            Invertd(keys=["pred"], transform=test_transforms, 
                    orig_keys=["image"], 
                    meta_keys=["pred_meta_dict"],
                    nearest_interp=False, to_tensor=True),
            ])

    # Create the data loader
    test_ds = CacheDataset(data=test_files, transform=test_transforms, cache_rate=0.1, num_workers=0)
    test_data_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

    # Load the model
    net = AttentionUnet(
                spatial_dims=3,
                in_channels=1,
                out_channels=1,
                channels=cfg["attention_unet_channels"],
                strides=cfg["attention_unet_strides"],
                dropout=0.1,
    )
    net.to(DEVICE)
    checkpoint = torch.load(cfg["path_to_model"], map_location=torch.device(DEVICE))["state_dict"]
    # NOTE: remove the 'net.' prefix from the keys because of how the model was initialized in lightning
    # https://discuss.pytorch.org/t/missing-keys-unexpected-keys-in-state-dict-when-loading-self-trained-model/22379/14
    for key in list(checkpoint.keys()):
        if 'net.' in key:
            checkpoint[key.replace('net.', '')] = checkpoint[key]
            del checkpoint[key]
        # remove the key loss_function.dice.class_weights because it is not needed
        # I had the error but I don't really know why
        if 'loss_function.dice.class_weight' in key:
            del checkpoint[key]
    net.load_state_dict(checkpoint)
    net.eval()

    # Run inference
    with torch.no_grad():
        for i, batch in enumerate(test_data_loader):
            # get the test input
            test_input = batch["image"].to(DEVICE)

            # run inference            
            batch["pred"] = sliding_window_inference(test_input, cfg["spatial_size"], mode="gaussian",
                                                    sw_batch_size=4, predictor=net, overlap=0.5, progress=False)

            # NOTE: monai's models do not normalize the output, so we need to do it manually
            if bool(F.relu(batch["pred"]).max()):
                batch["pred"] = F.relu(batch["pred"]) / F.relu(batch["pred"]).max() 
            else:
                batch["pred"] = F.relu(batch["pred"])

            # compute the dice score
            dice = dice_score(batch["pred"].cpu(), batch["label"].cpu())

            # post-process the prediction
            post_test_out = [test_post_pred(i) for i in decollate_batch(batch)]
                
            pred = post_test_out[0]['pred'].cpu()
            
            pred_cpu = batch["pred"].cpu()
            label_cpu = batch["label"].cpu()

            # Threshold the prediction and compute the dice score
            pred_0 = pred_cpu.clone()
            pred_0[pred_0 < 0.01] = 0
            pred_0[pred_0 >= 0.01] = 1
            dice = dice_score(pred_0, batch["label"].cpu())
            print(f"For thresh 0 dice score = {dice}")

            pred_0_01 = pred_cpu.clone()
            pred_0_01[pred_0_01 < 0.01] = 0
            pred_0_01[pred_0_01 >= 0.01] = 1
            dice_0_01 = dice_score(pred_0_01, batch["label"].cpu())
            print(f"For thresh 0.01 dice score = {dice_0_01}")

            pred_0_02 = pred_cpu.clone()
            pred_0_02[pred_0_02 < 0.02] = 0
            pred_0_02[pred_0_02 >= 0.02] = 1
            dice_0_02 = dice_score(pred_0_02, batch["label"].cpu())
            print(f"For thresh 0.02 dice score = {dice_0_02}")

            pred_0_05 = pred_cpu.clone()
            pred_0_05[pred_0_05 < 0.05] = 0
            pred_0_05[pred_0_05 >= 0.05] = 1
            dice_0_05 = dice_score(pred_0_05, batch["label"].cpu())
            print(f"For thresh 0.05 dice score = {dice_0_05}")

            pred_0_1 = pred_cpu.clone()
            pred_0_1[pred_0_1 < 0.1] = 0
            pred_0_1[pred_0_1 >= 0.1] = 1
            dice_0_1 = dice_score(pred_0_1, batch["label"].cpu())
            print(f"For thresh 0.1 dice score = {dice_0_1}")

            pred_0_2 = pred_cpu.clone()
            pred_0_2[pred_0_2 < 0.2] = 0
            pred_0_2[pred_0_2 >= 0.2] = 1
            dice_0_2 = dice_score(pred_0_2, batch["label"].cpu())
            print(f"For thresh 0.2 dice score = {dice_0_2}")

            pred_0_3 = pred_cpu.clone()
            pred_0_3[pred_0_3 < 0.3] = 0
            pred_0_3[pred_0_3 >= 0.3] = 1
            dice_0_3 = dice_score(pred_0_3, batch["label"].cpu())
            print(f"For thresh 0.3 dice score = {dice_0_3}")

            pred_0_4 = pred_cpu.clone()
            pred_0_4[pred_0_4 < 0.4] = 0
            pred_0_4[pred_0_4 >= 0.4] = 1
            dice_0_4 = dice_score(pred_0_4, batch["label"].cpu())
            print(f"For thresh 0.4 dice score = {dice_0_4}")

            pred_0_5 = pred_cpu.clone()
            pred_0_5[pred_0_5 < 0.5] = 0
            pred_0_5[pred_0_5 >= 0.5] = 1
            dice_0_5 = dice_score(pred_0_5, batch["label"].cpu())
            print(f"For thresh 0.5 dice score = {dice_0_5}")

            pred_0_6 = pred_cpu.clone()
            pred_0_6[pred_0_6 < 0.6] = 0
            pred_0_6[pred_0_6 >= 0.6] = 1
            dice_0_6 = dice_score(pred_0_6, batch["label"].cpu())
            print(f"For thresh 0.6 dice score = {dice_0_6}")

            pred_0_7 = pred_cpu.clone()
            pred_0_7[pred_0_7 < 0.7] = 0
            pred_0_7[pred_0_7 >= 0.7] = 1
            dice_0_7 = dice_score(pred_0_7, batch["label"].cpu())
            print(f"For thresh 0.7 dice score = {dice_0_7}")

            pred_0_8 = pred_cpu.clone()
            pred_0_8[pred_0_8 < 0.8] = 0
            pred_0_8[pred_0_8 >= 0.8] = 1
            dice_0_8 = dice_score(pred_0_8, batch["label"].cpu())
            print(f"For thresh 0.8 dice score = {dice_0_8}")

            pred_0_9 = pred_cpu.clone()
            pred_0_9[pred_0_9 < 0.9] = 0
            pred_0_9[pred_0_9 >= 0.9] = 1
            dice_0_9 = dice_score(pred_0_9, batch["label"].cpu())
            print(f"For thresh 0.9 dice score = {dice_0_9}")

            pred_0_95 = pred_cpu.clone()
            pred_0_95[pred_0_95 < 0.95] = 0
            pred_0_95[pred_0_95 >= 0.95] = 1
            dice_0_95 = dice_score(pred_0_95, batch["label"].cpu())
            print(f"For thresh 0.95 dice score = {dice_0_95}")

            pred_0_98 = pred_cpu.clone()
            pred_0_98[pred_0_98 < 0.98] = 0
            pred_0_98[pred_0_98 >= 0.98] = 1
            dice_0_98 = dice_score(pred_0_98, batch["label"].cpu())
            print(f"For thresh 0.98 dice score = {dice_0_98}")
        
            pred_0_99 = pred_cpu.clone()
            pred_0_99[pred_0_99 < 0.99] = 0
            pred_0_99[pred_0_99 >= 0.99] = 1
            dice_0_99 = dice_score(pred_0_99, batch["label"].cpu())
            print(f"For thresh 0.99 dice score = {dice_0_99}")
            
            # Get file name
            file_name = test_files[i]["image"].split("/")[-1].split(".")[0]
            print(f"Saving {file_name}: dice score = {dice}")

            # Save the prediction
            pred_saver = SaveImage(
                output_dir=output_dir , output_postfix="pred", output_ext=".nii.gz", 
                separate_folder=False, print_log=False)
            # save the prediction
            pred_saver(pred)

            # Save the dice score
            dice_scores[test_files[i]["image"]] = dice
            dice_scores_0_01[test_files[i]["image"]] = dice_0_01
            dice_scores_0_02[test_files[i]["image"]] = dice_0_02
            dice_scores_0_05[test_files[i]["image"]] = dice_0_05
            dice_scores_0_1[test_files[i]["image"]] = dice_0_1
            dice_scores_0_2[test_files[i]["image"]] = dice_0_2
            dice_scores_0_3[test_files[i]["image"]] = dice_0_3
            dice_scores_0_4[test_files[i]["image"]] = dice_0_4
            dice_scores_0_5[test_files[i]["image"]] = dice_0_5
            dice_scores_0_6[test_files[i]["image"]] = dice_0_6
            dice_scores_0_7[test_files[i]["image"]] = dice_0_7
            dice_scores_0_8[test_files[i]["image"]] = dice_0_8
            dice_scores_0_9[test_files[i]["image"]] = dice_0_9
            dice_scores_0_95[test_files[i]["image"]] = dice_0_95
            dice_scores_0_98[test_files[i]["image"]] = dice_0_98
            dice_scores_0_99[test_files[i]["image"]] = dice_0_99

            test_input.detach()


    # Save the dice scores
    with open(os.path.join(output_dir, "dice_scores.txt"), "w") as f:
        for key, value in dice_scores.items():
            f.write(f"{key}: {value}\n")
    with open(os.path.join(output_dir, "dice_scores_0_01.txt"), "w") as f:
        for key, value in dice_scores_0_01.items():
            f.write(f"{key}: {value}\n")
    with open(os.path.join(output_dir, "dice_scores_0_02.txt"), "w") as f:
        for key, value in dice_scores_0_02.items():
            f.write(f"{key}: {value}\n")
    with open(os.path.join(output_dir, "dice_scores_0_05.txt"), "w") as f:
        for key, value in dice_scores_0_05.items():
            f.write(f"{key}: {value}\n")
    with open(os.path.join(output_dir, "dice_scores_0_1.txt"), "w") as f:
        for key, value in dice_scores_0_1.items():
            f.write(f"{key}: {value}\n")
    with open(os.path.join(output_dir, "dice_scores_0_2.txt"), "w") as f:
        for key, value in dice_scores_0_2.items():
            f.write(f"{key}: {value}\n")
    with open(os.path.join(output_dir, "dice_scores_0_3.txt"), "w") as f:
        for key, value in dice_scores_0_3.items():
            f.write(f"{key}: {value}\n")
    with open(os.path.join(output_dir, "dice_scores_0_4.txt"), "w") as f:
        for key, value in dice_scores_0_4.items():
            f.write(f"{key}: {value}\n")
    with open(os.path.join(output_dir, "dice_scores_0_5.txt"), "w") as f:
        for key, value in dice_scores_0_5.items():
            f.write(f"{key}: {value}\n")
    with open(os.path.join(output_dir, "dice_scores_0_6.txt"), "w") as f:
        for key, value in dice_scores_0_6.items():
            f.write(f"{key}: {value}\n")
    with open(os.path.join(output_dir, "dice_scores_0_7.txt"), "w") as f:
        for key, value in dice_scores_0_7.items():
            f.write(f"{key}: {value}\n")
    with open(os.path.join(output_dir, "dice_scores_0_8.txt"), "w") as f:
        for key, value in dice_scores_0_8.items():
            f.write(f"{key}: {value}\n")
    with open(os.path.join(output_dir, "dice_scores_0_9.txt"), "w") as f:
        for key, value in dice_scores_0_9.items():
            f.write(f"{key}: {value}\n")
    with open(os.path.join(output_dir, "dice_scores_0_95.txt"), "w") as f:
        for key, value in dice_scores_0_95.items():
            f.write(f"{key}: {value}\n")
    with open(os.path.join(output_dir, "dice_scores_0_98.txt"), "w") as f:
        for key, value in dice_scores_0_98.items():
            f.write(f"{key}: {value}\n")
    with open(os.path.join(output_dir, "dice_scores_0_99.txt"), "w") as f:
        for key, value in dice_scores_0_99.items():
            f.write(f"{key}: {value}\n")


if __name__ == "__main__":
    main()