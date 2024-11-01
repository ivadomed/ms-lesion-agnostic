"""
This file is used to compute the performances of sct_deepseg psir-stir model on the external test set.
The input path is the path to the repository containing the datasets.
It outputs the following metrics: Dice score, lesion_ppv, lesion sensitivity, lesion F1 score.

Input: 
    --input-folder: path to the folder containing the datasets
    --output-path: path to the output folder

Output for each model:
    - dice_score.csv
    - lesion_ppv.csv
    - lesion_sensitivity.csv
    - lesion_f1_score.csv

Example:
    python test_sct_models.py --input-folder /path/to/datasets --output-path /path/to/output-folder

Author: Pierre-Louis Benveniste
"""

import os
import argparse
import json
import nibabel as nib
from utils import dice_score, lesion_ppv, lesion_f1_score, lesion_sensitivity
import tqdm
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def parse_arguments():
    parser = argparse.ArgumentParser(description='Compute the performances of the three SCT models on the test set')
    parser.add_argument('--input-folder', type=str, help='Path to the folder containing the datasets')
    parser.add_argument('--output-path', type=str, help='Path to the output folder')
    args = parser.parse_args()
    return args


def main():

    args = parse_arguments()
    # Get the arguments
    input_folder = args.input_folder
    output_path = args.output_path

    # Path to corresponding datasets
    basel_2018_path = os.path.join(input_folder, 'ms-basel-2018')
    basel_2020_path = os.path.join(input_folder, 'ms-basel-2020')

    # We list all files which have a segmentation using rglob
    basel_2018_label_files = list(Path(basel_2018_path).rglob('*_lesion-manual.nii.gz'))
    basel_2020_label_files = list(Path(basel_2020_path).rglob('*_lesion-manual.nii.gz'))
    label_files = basel_2018_label_files + basel_2020_label_files

    # We create the output folder
    path_sct_deepseg_stir_psir = os.path.join(output_path, 'sct_deepseg_psir-stir')
    os.makedirs(path_sct_deepseg_stir_psir, exist_ok=True)

    # Initilize a dataframe to store the results with the following structure: image_path, contrast, orientation, dice_score, lesion_ppv, lesion_sensitivity, lesion_f1_score
    metrics_results = []

    # Iterate over the label files
    for label_file in tqdm.tqdm(label_files):
        # Find the corresponding image file
        image_file = str(label_file).replace('_lesion-manual.nii.gz', '.nii.gz').replace('derivatives/labels/','')
        # Check if images exists  
        assert (os.path.exists(image_file))
        
        # Get the subject name
        sub_name = image_file.split('/')[-1]
        contrast = sub_name.split('_')[-1].split('.')[0]
        if 'acq-sag' in sub_name:
            orientation = 'sag'
        elif 'acq-ax' in sub_name:
            orientation = 'ax'
        else:
            orientation = 'iso'
        pred_deepseg_stir_psir = os.path.join(path_sct_deepseg_stir_psir, sub_name.replace('.nii.gz', '_lesionseg.nii.gz'))

        print("Performing predictions for subject: ", sub_name)

        ## sct_deepseg -t seg_sc_ms_lesion_stir_psir
        if contrast == 'T1w':
            assert os.system(f"sct_deepseg -i {image_file} -task seg_sc_ms_lesion_stir_psir -c psir -o {pred_deepseg_stir_psir}") ==0
        else: 
            assert os.system(f"sct_deepseg -i {image_file} -task seg_sc_ms_lesion_stir_psir -c stir -o {pred_deepseg_stir_psir}") ==0

        # This methods produces the spinal cord seg which we don't want
        assert os.system(f"rm {pred_deepseg_stir_psir.replace('.nii.gz', '_sc_seg.nii.gz')}") ==0
        # We also rename the file to remove the _lesion_seg same for json file
        assert os.system(f"mv {pred_deepseg_stir_psir.replace('.nii.gz', '_lesion_seg.nii.gz')} {pred_deepseg_stir_psir}") ==0
        assert os.system(f"mv {pred_deepseg_stir_psir.replace('.nii.gz', '_lesion_seg.json')} {pred_deepseg_stir_psir.replace('.nii.gz', '.json')}") ==0

        ############################################
        # Compute the metrics
        # We use the ground truth lesion mask
        gt_data = nib.load(str(label_file)).get_fdata()

        # sct_deepseg_lesion
        pred_deepseg_stir_psir_data = nib.load(str(pred_deepseg_stir_psir)).get_fdata()

        dice = dice_score(pred_deepseg_stir_psir_data, gt_data)
        ppv = lesion_ppv(gt_data, pred_deepseg_stir_psir_data)
        f1 = lesion_f1_score(gt_data, pred_deepseg_stir_psir_data)
        sensitivity = lesion_sensitivity(gt_data, pred_deepseg_stir_psir_data)

        # Add the results to the dataset
        metrics_results.append([image_file, contrast, orientation, dice, ppv, sensitivity, f1])

    # Format the dataset to a pandas datafram
    metrics_results = pd.DataFrame(metrics_results, columns=['image_path', 'contrast', 'orientation', 'dice_score', 'ppv_score', 'sensitivity_score', 'f1_score'])
    contrast_counts = metrics_results['contrast'].value_counts()
    metrics_results['contrast_count'] = metrics_results['contrast'].apply(lambda x: x + f' (n={contrast_counts[x]})')
    orientation_counts = metrics_results['orientation'].value_counts()
    metrics_results['orientation_count'] = metrics_results['orientation'].apply(lambda x: x + f' (n={orientation_counts[x]})')

    # Plot the results
    # plot a violin plot per contrast for dice scores
    plt.figure(figsize=(20, 10))
    plt.grid(True)
    sns.violinplot(x='contrast_count', y='dice_score', data=metrics_results)
    # y ranges from -0.2 to 1.2
    plt.ylim(-0.2, 1.2)
    plt.title('Dice scores per contrast')
    plt.show()

    # # Save the plot
    plt.savefig(path_sct_deepseg_stir_psir + '/dice_scores_contrast.png')
    print(f"Saved the dice plot in {path_sct_deepseg_stir_psir}")

    # plot a violin plot per contrast for ppv scores
    plt.figure(figsize=(20, 10))
    plt.grid(True)
    sns.violinplot(x='contrast_count', y='ppv_score', data=metrics_results)
    # y ranges from -0.2 to 1.2
    plt.ylim(-0.2, 1.2)
    plt.title('PPV scores per contrast')
    plt.show()

    # # Save the plot
    plt.savefig(path_sct_deepseg_stir_psir + '/ppv_scores_contrast.png')
    print(f"Saved the ppv plot in {path_sct_deepseg_stir_psir}")

    # plot a violin plot per contrast for f1 scores
    plt.figure(figsize=(20, 10))
    plt.grid(True)
    sns.violinplot(x='contrast_count', y='f1_score', data=metrics_results)
    # y ranges from -0.2 to 1.2
    plt.ylim(-0.2, 1.2)
    plt.title('F1 scores per contrast')
    plt.show()

    # # Save the plot
    plt.savefig(path_sct_deepseg_stir_psir + '/f1_scores_contrast.png')
    print(f"Saved the F1 plot in {path_sct_deepseg_stir_psir}")

    # plot a violin plot per contrast for f1 scores
    plt.figure(figsize=(20, 10))
    plt.grid(True)
    sns.violinplot(x='contrast_count', y='sensitivity_score', data=metrics_results)
    # y ranges from -0.2 to 1.2
    plt.ylim(-0.2, 1.2)
    plt.title('Sensitivity scores per contrast')
    plt.show()

    # # Save the plot
    plt.savefig(path_sct_deepseg_stir_psir + '/sensitivity_scores_contrast.png')
    print(f"Saved the sensitivity plot in {path_sct_deepseg_stir_psir}")

    # Print the mean dice score per contrast and std
    print("\nDice score per contrast (mean ± std)")
    dice_stats = metrics_results.groupby('contrast_count')['dice_score'].agg(['mean', 'std'])
    for contrast, row in dice_stats.iterrows():
        print(f"{contrast}: {row['mean']:.4f} ± {row['std']:.4f}")

    # Print the mean ppv score per contrast and std
    print("\nPPV score per contrast (mean ± std)")
    ppv_stats = metrics_results.groupby('contrast_count')['ppv_score'].agg(['mean', 'std'])
    for contrast, row in ppv_stats.iterrows():
        print(f"{contrast}: {row['mean']:.4f} ± {row['std']:.4f}")
    
    # Print the mean f1 score per contrast and std
    print("\nF1 score per contrast (mean ± std)")
    f1_stats = metrics_results.groupby('contrast_count')['f1_score'].agg(['mean', 'std'])
    for contrast, row in f1_stats.iterrows():
        print(f"{contrast}: {row['mean']:.4f} ± {row['std']:.4f}")

    # Print the mean sensitivity score per contrast and std
    print("\nSensitivity score per contrast (mean ± std)")
    sensitivity_stats = metrics_results.groupby('contrast_count')['sensitivity_score'].agg(['mean', 'std'])
    for contrast, row in sensitivity_stats.iterrows():
        print(f"{contrast}: {row['mean']:.4f} ± {row['std']:.4f}")

    # save the metrics
    metrics_results.to_csv(os.path.join(path_sct_deepseg_stir_psir, 'metrics_results.csv'), index=False)

    return  None


if __name__ == '__main__':
    main()