"""
This file is used to compute the performances of the sct_deepseg PSIR-STIR model on the test set.
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
    path_sct_deepseg_stir_psir = os.path.join(output_path, 'sct_deepseg_stir_psir')
    os.makedirs(path_sct_deepseg_stir_psir, exist_ok=True)

    # Initialize the metrics
    dice_scores_psir = {}
    ppv_scores_psir = {}
    f1_scores_psir = {}
    sensitivity_scores_psir = {}

    # Iterate over the test set subjects
    for i in tqdm.tqdm(data_sub):

        # Get the subject name
        sub_name = i['image'].split('/')[-1]

        print("Performing predictions for subject: ", sub_name)

        pred_deepseg_stir_psir = os.path.join(path_sct_deepseg_stir_psir, sub_name.replace('.nii.gz', '_lesionseg.nii.gz'))
        
        ############################################
        # Perform the predictions
        ## sct_deepseg -t seg_sc_ms_lesion_stir_psir
        if i['contrast'] == 'PSIR':
            assert os.system(f"sct_deepseg -i {i['image']} -task seg_sc_ms_lesion_stir_psir -c psir -o {pred_deepseg_stir_psir}") ==0
        else: 
            assert os.system(f"sct_deepseg -i {i['image']} -task seg_sc_ms_lesion_stir_psir -c stir -o {pred_deepseg_stir_psir}") ==0

        # This methods produces the spinal cord seg which we don't want
        assert os.system(f"rm {pred_deepseg_stir_psir.replace('.nii.gz', '_sc_seg.nii.gz')}") ==0
        # We also rename the file to remove the _lesion_seg same for json file
        assert os.system(f"mv {pred_deepseg_stir_psir.replace('.nii.gz', '_lesion_seg.nii.gz')} {pred_deepseg_stir_psir}") ==0
        assert os.system(f"mv {pred_deepseg_stir_psir.replace('.nii.gz', '_lesion_seg.json')} {pred_deepseg_stir_psir.replace('.nii.gz', '.json')}") ==0

        ############################################
        # Compute the metrics
        # We use the ground truth lesion mask
        gt = i['label']
        gt_data = nib.load(str(gt)).get_fdata()

        # sct_deepseg -t seg_sc_ms_lesion_stir_psir
        pred_psir_data = nib.load(str(pred_deepseg_stir_psir)).get_fdata()

        dice_psir = dice_score(pred_psir_data, gt_data)
        ppv_psir = lesion_ppv(gt_data, pred_psir_data)
        f1_psir = lesion_f1_score(gt_data, pred_psir_data)
        sensitivity_psir = lesion_sensitivity(gt_data, pred_psir_data)

        dice_scores_psir[i['image']] = dice_psir
        ppv_scores_psir[i['image']] = ppv_psir
        f1_scores_psir[i['image']] = f1_psir
        sensitivity_scores_psir[i['image']] = sensitivity_psir

    # Save the metrics
    # sct_deepseg -t seg_sc_ms_lesion_stir_psir
    with open(os.path.join(output_path, 'dice_score_psir.txt'), 'w') as f:
        for key, value in dice_scores_psir.items():
            f.write(f"{key}: {value}\n")
    with open(os.path.join(output_path, 'ppv_score_psir.txt'), 'w') as f:
        for key, value in ppv_scores_psir.items():
            f.write(f"{key}: {value}\n")
    with open(os.path.join(output_path, 'f1_score_psir.txt'), 'w') as f:
        for key, value in f1_scores_psir.items():
            f.write(f"{key}: {value}\n")
    with open(os.path.join(output_path, 'sensitivity_score_psir.txt'), 'w') as f:
        for key, value in sensitivity_scores_psir.items():
            f.write(f"{key}: {value}\n")
    
    return  None


if __name__ == '__main__':
    main()