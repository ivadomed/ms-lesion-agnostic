"""
This file is used to evaluate the predictions of the model on the test set.

Input:
    -pred-folder: Folder containing the predictions of the model on the test set
    -label-folder: Folder containing the images of the test set
    -image-folder: Folder containing the images of the test set
    -levels-folder: Folder containing the vertebral levels of the test set
    -conversion-dict: Dictionary containing the conversion of the predictions to the original labels
    -output-folder: Folder to save the evaluation results

Output:
    None

Example: 
    python evaluate_predictions_per_vert_level.py -pred-folder /path/to/predictions -label-folder /path/to/labels -image-folder /path/to/images -levels-folder /path/to/levels -conversion-dict /path/to/dict -output-folder /path/to/output

Author: Pierre-Louis Benveniste
"""

import os
import numpy as np
import argparse
from pathlib import Path
import json
import nibabel as nib
from tqdm import tqdm
from utils import dice_score, lesion_ppv, lesion_f1_score, lesion_sensitivity


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-pred-folder", required=True, type=str, help="Folder containing the predictions of the model on the test set")
    parser.add_argument("-label-folder", required=True, type=str, help="Folder containing the images of the test set")
    parser.add_argument("-image-folder", required=True, type=str, help="Folder containing the images of the test set")
    parser.add_argument("-levels-folder", required=True, type=str, help="Folder containing the vertebral levels of the test set")
    parser.add_argument("-conversion-dict", required=True, type=str, help="Dictionary containing the conversion of the predictions to the original labels")
    parser.add_argument("-output-folder", required=True, type=str, help="Folder to save the evaluation results")
    return parser.parse_args()


def main():

    # Parse arguments
    args = parse_args()
    pred_folder = args.pred_folder
    label_folder = args.label_folder
    image_folder = args.image_folder
    levels_folder = args.levels_folder
    conversion_dict = args.conversion_dict
    output_folder = args.output_folder

    # Create output folder if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get all the predictions (with rglob)
    predictions = list(Path(pred_folder).rglob("*.nii.gz"))

    # Open the conversion dictionary (its a json file)
    with open(conversion_dict, "r") as f:
        conversion_dict = json.load(f)

    # Dict of global dice score
    global_dice_scores = {}
    global_ppv_scores = {}
    global_f1_scores = {}
    global_sensitivity_scores = {}
    # Dict of per level dice score
    dice_scores = {}
    ppv_scores = {}
    f1_scores = {}
    sensitivity_scores = {}

    # Iterate over the results
    for pred in tqdm(predictions):
        # Get the corresponding image
        label = os.path.join(label_folder, pred.name)
        image = os.path.join(image_folder, pred.name).replace(".nii.gz", "_0000.nii.gz")

        # Get the corresponding vertebral levels
        levels = os.path.join(levels_folder, Path(image).name)
        # If the levels file doesn't exist, skip this prediction
        if not os.path.exists(levels):
            print(f"Levels file does not exist, skipping prediction {pred.name}")
            continue

        # Load the predictions and the label
        pred_data = nib.load(str(pred)).get_fdata()
        label_data = nib.load(str(label)).get_fdata()
        levels_data = nib.load(str(levels)).get_fdata()

        # Compute dice score
        dice = dice_score(pred_data, label_data)
        ppv = lesion_ppv(label_data, pred_data)
        f1 = lesion_f1_score(label_data, pred_data)
        sensitivity = lesion_sensitivity(label_data, pred_data)

        # Get initial image name from conversion dict
        image_name = None
        for original_image in conversion_dict:
            if conversion_dict[original_image] == image:
                image_name = original_image
                break
        
        # Save the dice score
        global_dice_scores[image_name] = dice
        global_ppv_scores[image_name] = ppv
        global_f1_scores[image_name] = f1
        global_sensitivity_scores[image_name] = sensitivity

        # Now we evaluate the predictions for each vertebral level (except 0)
        vert_levels = np.unique(levels_data)

        for level in vert_levels:
            if level == 0 or level == max(vert_levels):
                continue
            # For this test case, the orientation is AIL
            ## Get levels coordinates
            level_mask = levels_data == level
            level_mask_next = levels_data == (level + 1)
            ### Get corresponding coordinates
            level_coords = np.where(level_mask)
            bottom_voxel = int(level_coords[2]) # Because images are in RPI
            level_next_coords = np.where(level_mask_next)
            top_voxel = int(level_next_coords[2]) # Because images are in RPI

            # Crop the predictions and the ground truth

            pred_patch_data = pred_data[:, :, top_voxel:bottom_voxel]
            label_patch_data = label_data[:, :, top_voxel:bottom_voxel]

            # Compute the scores for each chunk
            dice = dice_score(pred_patch_data, label_patch_data)
            ppv = lesion_ppv(label_patch_data, pred_patch_data)
            f1 = lesion_f1_score(label_patch_data, pred_patch_data)
            sensitivity = lesion_sensitivity(label_patch_data, pred_patch_data)

            # Initialize each subdictionary:
            level_key = f"{int(level)}_to_{int(level + 1)}"
            if level_key not in dice_scores:
                dice_scores[level_key] = {}
            if level_key not in ppv_scores:
                ppv_scores[level_key] = {}
            if level_key not in f1_scores:
                f1_scores[level_key] = {}
            if level_key not in sensitivity_scores:
                sensitivity_scores[level_key] = {}
            
            # Save the scores for this level
            dice_scores[level_key][image_name] = dice
            ppv_scores[level_key][image_name] = ppv
            f1_scores[level_key][image_name] = f1
            sensitivity_scores[level_key][image_name] = sensitivity
        
    # Save the results
    with open(os.path.join(output_folder, "dice_scores.txt"), "w") as f:
        for key, value in global_dice_scores.items():
            f.write(f"{key}: {value}\n")
    with open(os.path.join(output_folder, "ppv_scores.txt"), "w") as f:
        for key, value in global_ppv_scores.items():
            f.write(f"{key}: {value}\n")
    with open(os.path.join(output_folder, "f1_scores.txt"), "w") as f:
        for key, value in global_f1_scores.items():
            f.write(f"{key}: {value}\n")
    with open(os.path.join(output_folder, "sensitivity_scores.txt"), "w") as f:
        for key, value in global_sensitivity_scores.items():
            f.write(f"{key}: {value}\n")

    # Now for each vertebral level, save the scores
    for level in dice_scores:
        with open(os.path.join(output_folder, f"dice_scores_{level}.txt"), "w") as f:
            for key, value in dice_scores[level].items():
                f.write(f"{key}: {value}\n")
        with open(os.path.join(output_folder, f"ppv_scores_{level}.txt"), "w") as f:
            for key, value in ppv_scores[level].items():
                f.write(f"{key}: {value}\n")
        with open(os.path.join(output_folder, f"f1_scores_{level}.txt"), "w") as f:
            for key, value in f1_scores[level].items():
                f.write(f"{key}: {value}\n")
        with open(os.path.join(output_folder, f"sensitivity_scores_{level}.txt"), "w") as f:
            for key, value in sensitivity_scores[level].items():
                f.write(f"{key}: {value}\n")

    return None


if __name__ == "__main__":
    main()