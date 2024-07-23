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
                    ResizeWithPadOrCropd(
                        keys=["image", "label"],
                        spatial_size=cfg["spatial_size"],
                    ),
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
    test_ds = CacheDataset(data=test_files, transform=test_transforms, cache_rate=0.1, num_workers=4)
    test_data_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=1)

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

            # Threshold the prediction
            pred[pred < 0.5] = 0
            
            # Get file name
            file_name = test_files[i]["image"].split("/")[-1].split(".")[0]
            print(f"Saving {file_name}")

            # Save the prediction
            pred_saver = SaveImage(
                output_dir=output_dir , output_postfix="pred", output_ext=".nii.gz", 
                separate_folder=False, print_log=False)
            # save the prediction
            pred_saver(pred)

            # Save the dice score
            dice_scores[test_files[i]["image"]] = dice

            test_input.detach()


    # Save the dice scores
    with open(os.path.join(output_dir, "dice_scores.txt"), "w") as f:
        for key, value in dice_scores.items():
            f.write(f"{key}: {value}\n")


if __name__ == "__main__":
    main()