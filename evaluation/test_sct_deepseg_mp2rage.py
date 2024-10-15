"""
This file is used to compute the performances of the sct_deepseg_mp2rage model on the test set.
The input file is the msd dataset which stores the test set. 
It outputs the following metrics: Dice score, lesion_ppv, lesion sensitivity, lesion F1 score.

Input: 
    --msd-data-path: path to the msd dataset
    --output-path: path to the output folder

Output for each model:
    - dice_score.csv
    - lesion_ppv.csv
    - lesion_sensitivity.csv
    - lesion_f1_score.csv

Example:
    python test_sct_models.py --msd-data-path /path/to/msd-dataset --output-path /path/to/output-folder

Author: Pierre-Louis Benveniste
"""

import os
import argparse
import json
import nibabel as nib
from utils import dice_score, lesion_ppv, lesion_f1_score, lesion_sensitivity
import tqdm


def parse_arguments():
    parser = argparse.ArgumentParser(description='Compute the performances of the three SCT models on the test set')
    parser.add_argument('--msd-data-path', type=str, help='Path to the msd dataset')
    parser.add_argument('--output-path', type=str, help='Path to the output folder')
    args = parser.parse_args()
    return args


def main():

    args = parse_arguments()
    # Get the arguments
    msd_data_path = args.msd_data_path
    output_path = args.output_path

    # Open the msd dataset
    with open(msd_data_path) as f:
        data = json.load(f)
    
    data_sub = data['test']

    # Create the output folders
    path_sct_deepseg_mp2rage = os.path.join(output_path, 'sct_deepseg_mp2rage')
    os.makedirs(path_sct_deepseg_mp2rage, exist_ok=True)

    # Initialize the metrics
    dice_scores_mp2rage = {}
    ppv_scores_mp2rage = {}
    f1_scores_mp2rage = {}
    sensitivity_scores_mp2rage = {}

    # Iterate over the test set subjects
    for i in tqdm.tqdm(data_sub):

        # Get the subject name
        sub_name = i['image'].split('/')[-1]

        print("Performing predictions for subject: ", sub_name)

        pred_deepseg_mp2rage = os.path.join(path_sct_deepseg_mp2rage, sub_name.replace('.nii.gz', '_lesionseg.nii.gz'))
        
        ############################################
        # Perform the predictions
        ## sct_deepseg -t seg_ms_lesion_mp2rage
        assert os.system(f"sct_deepseg -i {i['image']} -task seg_ms_lesion_mp2rage -o {pred_deepseg_mp2rage}") ==0

        ############################################
        # Compute the metrics
        # We use the ground truth lesion mask
        gt = i['label']
        gt_data = nib.load(str(gt)).get_fdata()

        # sct_deepseg -t seg_ms_lesion_mp2rage
        pred_mp2rage_data = nib.load(str(pred_deepseg_mp2rage)).get_fdata()

        dice_mp2rage = dice_score(pred_mp2rage_data, gt_data)
        ppv_mp2rage = lesion_ppv(gt_data, pred_mp2rage_data)
        f1_mp2rage = lesion_f1_score(gt_data, pred_mp2rage_data)
        sensitivity_mp2rage = lesion_sensitivity(gt_data, pred_mp2rage_data)

        dice_scores_mp2rage[i['image']] = dice_mp2rage
        ppv_scores_mp2rage[i['image']] = ppv_mp2rage
        f1_scores_mp2rage[i['image']] = f1_mp2rage
        sensitivity_scores_mp2rage[i['image']] = sensitivity_mp2rage

    # Save the metrics
    # sct_deepseg -t seg_ms_lesion_mp2rage
    with open(os.path.join(output_path, 'dice_score_mp2rage.txt'), 'w') as f:
        for key, value in dice_scores_mp2rage.items():
            f.write(f"{key}: {value}\n")
    with open(os.path.join(output_path, 'ppv_score_mp2rage.txt'), 'w') as f:
        for key, value in ppv_scores_mp2rage.items():
            f.write(f"{key}: {value}\n")
    with open(os.path.join(output_path, 'f1_score_mp2rage.txt'), 'w') as f:
        for key, value in f1_scores_mp2rage.items():
            f.write(f"{key}: {value}\n")
    with open(os.path.join(output_path, 'sensitivity_score_mp2rage.txt'), 'w') as f:
        for key, value in sensitivity_scores_mp2rage.items():
            f.write(f"{key}: {value}\n")
    
    return  None


if __name__ == '__main__':
    main()