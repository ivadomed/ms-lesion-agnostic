"""
This script compares the manual segmentation, the predicted segmentation from model 1 and the predicted segmentation from model 2. 
It calculates the dice score, the lesion sensitivity, the lesion PPV, the lesion F1 score and difference in terms of volume and lesion count.

Input:
    --manual: Folder containing the manual segmentations
    --pred1: Folder containing the predictions of model 1
    --pred2: Folder containing the predictions of model 2
    --image-folder: Folder containing the images of the test set
    --conversion-dict: Dictionary containing the conversion of the predictions to the original labels
    --output-folder: Folder to save the evaluation results

Output:
    None

Example:
    python compare_manual_vs_pred_vs_predpred.py --manual /path/to/manual --pred1 /path/to/pred1 --pred2 /path/to/pred2 --image-folder /path/to/images --conversion-dict /path/to/dict --output-folder /path/to/output

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
import pandas as pd
import skimage


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manual", required=True, type=str, help="Folder containing the manual segmentations")
    parser.add_argument("--pred1", required=True, type=str, help="Folder containing the predictions of model 1")
    parser.add_argument("--pred2", required=True, type=str, help="Folder containing the predictions of model 2")
    parser.add_argument("--image-folder", required=True, type=str, help="Folder containing the images of the test set")
    parser.add_argument("--conversion-dict", required=True, type=str, help="Dictionary containing the conversion of the predictions to the original labels")
    parser.add_argument("--output-folder", required=True, type=str, help="Folder to save the evaluation results")
    return parser.parse_args()


def main():
    # Parse arguments
    args = parse_args()
    pred1_folder = args.pred1
    pred2_folder = args.pred2
    manual_folder = args.manual
    image_folder = args.image_folder
    conversion_dict = args.conversion_dict
    output_folder = args.output_folder

    # Create output folder if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get all the predictions (with rglob)
    predictions1 = list(Path(pred1_folder).rglob("*.nii.gz"))

    # Open the conversion dictionary (its a json file)
    with open(conversion_dict, "r") as f:
        conversion_dict = json.load(f)

    # Create an empty dataframes with the following columns: dice_pred1_manual, dice_pred2_manual, dice_pred1_pred2, ppv_pred1_manual, ppv_pred2_manual, 
    # ppv_pred1_pred2, f1_pred1_manual, f1_pred2_manual, f1_pred1_pred2, sensitivity_pred1_manual, sensitivity_pred2_manual, sensitivity_pred1_pred2, 
    # volume_pred1, volume_pred2, volume_manual, lesion_count_pred1, lesion_count_pred2, lesion_count_manual
    dataframe = pd.DataFrame(columns=["dice_pred1_manual", "dice_pred2_manual", "dice_pred1_pred2", "ppv_pred1_manual", "ppv_pred2_manual", "ppv_pred1_pred2", 
                                      "f1_pred1_manual", "f1_pred2_manual", "f1_pred1_pred2", "sensitivity_pred1_manual", "sensitivity_pred2_manual", 
                                      "sensitivity_pred1_pred2", "volume_pred1", "volume_pred2", "volume_manual", "lesion_count_pred1", "lesion_count_pred2", 
                                      "lesion_count_manual"])

    # Iterate over the results
    for pred1 in tqdm(predictions1):

        ## Get the corresponding image
        manual = os.path.join(manual_folder, pred1.name)
        image = os.path.join(image_folder, pred1.name).replace(".nii.gz", "_0000.nii.gz")
        pred2 = os.path.join(pred2_folder, pred1.name)

        # Load the data
        pred1_data = nib.load(str(pred1)).get_fdata()
        pred2_data = nib.load(str(pred2)).get_fdata()
        manual_data = nib.load(str(manual)).get_fdata()

        # Find original file name
        original_image = None
        for image_name in conversion_dict:
            if conversion_dict[image_name] == image:
                original_image = image_name
                break

        # Compute dice score
        dice_pred1_manual = dice_score(pred1_data, manual_data)
        dice_pred2_manual = dice_score(pred2_data, manual_data)
        dice_pred1_pred2 = dice_score(pred1_data, pred2_data)

        # Compute lesion PPV
        ppv_pred1_manual = lesion_ppv(manual_data, pred1_data)
        ppv_pred2_manual = lesion_ppv(manual_data, pred2_data)
        ppv_pred1_pred2 = lesion_ppv(pred1_data, pred2_data)

        # Compute lesion F1 score
        f1_pred1_manual = lesion_f1_score(manual_data, pred1_data)
        f1_pred2_manual = lesion_f1_score(manual_data, pred2_data)
        f1_pred1_pred2 = lesion_f1_score(pred1_data, pred2_data)

        # Compute lesion sensitivity
        sensitivity_pred1_manual = lesion_sensitivity(manual_data, pred1_data)
        sensitivity_pred2_manual = lesion_sensitivity(manual_data, pred2_data)
        sensitivity_pred1_pred2 = lesion_sensitivity(pred1_data, pred2_data)

        # Compute volume
        voxel_volume = nib.load(str(pred1)).header.get_zooms()
        voxel_volume =voxel_volume[0]*voxel_volume[1]*voxel_volume[2]
        volume_pred1 = np.sum(pred1_data) * voxel_volume
        volume_pred2 = np.sum(pred2_data) * voxel_volume
        volume_manual = np.sum(manual_data) * voxel_volume

        # Count the number of lesions
        _, lesion_count_pred1 = skimage.measure.label(pred1_data, connectivity=2, return_num=True)
        _, lesion_count_pred2 = skimage.measure.label(pred2_data, connectivity=2, return_num=True)
        _, lesion_count_manual = skimage.measure.label(manual_data, connectivity=2, return_num=True)

        # Save the results
        dataframe.loc[original_image] = [dice_pred1_manual, dice_pred2_manual, dice_pred1_pred2, ppv_pred1_manual, ppv_pred2_manual, ppv_pred1_pred2, 
                                          f1_pred1_manual, f1_pred2_manual, f1_pred1_pred2, sensitivity_pred1_manual, sensitivity_pred2_manual, 
                                          sensitivity_pred1_pred2, volume_pred1, volume_pred2, volume_manual, lesion_count_pred1, lesion_count_pred2, 
                                          lesion_count_manual]

    # Save the results
    dataframe.to_csv(os.path.join(output_folder, "evaluation_results.csv"))


if __name__ == "__main__":
    main()