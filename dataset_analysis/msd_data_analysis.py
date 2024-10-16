"""
This file was created to analyze the msd dataset used for training and testing our dataset. 
It takes as input the msd dataset and analysis the properties of the dataset.

Input:
    --msd-data-path
    --dataset-path
    --output-folder

Output:
    None

Example:
    python dataset_analysis/msd_data_analysis.py --msd-data-path /path/to/msd/data --dataset-path /path/to/folder/of/datasets --output-folder /path/to/output/folder

Author: Pierre-Louis Benveniste
"""

import argparse
import os
import json
import nibabel as nib
import numpy as np
from pathlib import Path
from image import Image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--msd-data-path', type=str, required=True, help='Path to the MSD dataset')
    parser.add_argument('--output-folder', type=str, required=True, help='Path to the output folder')
    parser.add_argument('--dataset-path', type=str, required=True, help='Path to the folder containing the datasets')
    return parser.parse_args()


def main():
    # Parse arguments
    args = parse_args()
    msd_data_path = args.msd_data_path
    output_folder = args.output_folder
    dataset_path = args.dataset_path
    
    # Build the output folder
    os.makedirs(output_folder, exist_ok=True)

    # Load the dataset
    with open(msd_data_path, 'r') as f:
        msd_data = json.load(f)
    
    # Get data
    data = msd_data['train'] + msd_data['validation'] + msd_data['test']

    print("Number of images: ", len(data))
    print("Number of images for training: ", (msd_data['numTraining']))
    print("Number of images for validation: ", (msd_data['numValidation']))
    print("Number of images for testing: ", (msd_data['numTest']))

    # Count the number of images per countrast
    contrast_count = {}
    for image in data:
        contrast = image['contrast']
        if contrast not in contrast_count:
            contrast_count[contrast] = 0
        contrast_count[contrast] += 1
    
    print("Number of images per contrast: ", contrast_count)

    # Count the number of images per orientation
    orientation_count = {}
    for image in data:
        orientation = image['orientation']
        if orientation not in orientation_count:
            orientation_count[orientation] = 0
        orientation_count[orientation] += 1
    
    print("Number of images per orientation: ", orientation_count)

    # Now we will look at the average resolution of the images
    ## Iterate over the images
    resolutions = []
    for image in data:
        image_reoriented = Image(image['image']).change_orientation('RPI')
        resolution = image_reoriented.dim[4:7]
        resolution = [float(res) for res in resolution]
        resolutions.append(resolution)
        
    print("Average resolution: ", np.mean(resolutions, axis=0))
    print("Std resolution: ", np.std(resolutions, axis=0))
    print("Median resolution: ", np.median(resolutions, axis=0))

    print("-------------------------------------")

    #############################################################
    # Now we can look at the external testing dataset ms-basel-2018 and ms-basel-2020
    path_ms_basel_2018 = os.path.join(dataset_path, 'ms-basel-2018')
    path_ms_basel_2020 = os.path.join(dataset_path, 'ms-basel-2020')

    # We count all the segmentation files in both using rglob
    ms_basel_2018 = list(Path(path_ms_basel_2018).rglob("*lesion-manual.nii.gz"))
    ms_basel_2020 = list(Path(path_ms_basel_2020).rglob("*lesion-manual.nii.gz"))

    list_contrast_ms_basel_2018 = [str(image).split("/")[-1].split('_')[-2] for image in ms_basel_2018]

    print("Number of images in ms-basel-2018: ", len(ms_basel_2018))
    print("Contrast in ms-basel-2018: ", set(list_contrast_ms_basel_2018))
    # Print the number of each contrast
    contrast_count_ms_basel_2018 = {}
    for contrast in set(list_contrast_ms_basel_2018):
        contrast_count_ms_basel_2018[contrast] = list_contrast_ms_basel_2018.count(contrast)
    print("Number of images per contrast in ms-basel-2018: ", contrast_count_ms_basel_2018)

    list_contrast_ms_basel_2020 = [str(image).split("/")[-1].split('_')[-2] for image in ms_basel_2020]

    print("Number of images in ms-basel-2020: ", len(ms_basel_2020))
    print("Contrast in ms-basel-2020: ", set(list_contrast_ms_basel_2020))
    # Print the number of each contrast
    contrast_count_ms_basel_2020 = {}
    for contrast in set(list_contrast_ms_basel_2020):
        contrast_count_ms_basel_2020[contrast] = list_contrast_ms_basel_2020.count(contrast)
    print("Number of images per contrast in ms-basel-2020: ", contrast_count_ms_basel_2020)

    print("-------------------------------------")
    #############################################################
    # Now we can look at the external testing dataset umass*
    path_umass_1 = os.path.join(dataset_path, 'umass-ms-ge-hdxt1.5')
    path_umass_2 = os.path.join(dataset_path, 'umass-ms-ge-pioneer3')
    path_umass_3 = os.path.join(dataset_path, 'umass-ms-siemens-espree1.5')
    path_umass_4 = os.path.join(dataset_path, 'umass-ms-ge-excite1.5')

    # We count all the segmentation files in both using rglob
    umass_1 = list(Path(path_umass_1).rglob("*.nii.gz"))
    umass_2 = list(Path(path_umass_2).rglob("*.nii.gz"))
    umass_3 = list(Path(path_umass_3).rglob("*.nii.gz"))
    umass_4 = list(Path(path_umass_4).rglob("*.nii.gz"))

    # Concatenate all the lists
    umass = umass_1 + umass_2 + umass_3 + umass_4

    # Remove files containing ''SHA256E' in the name
    umass = [image for image in umass if 'SHA256E' not in str(image)]

    list_contrast_umass = [str(image).split("/")[-1].split('_')[-1].split('.')[0] for image in umass]

    print("Number of images in umass: ", len(umass))
    print("Contrast in umass: ", set(list_contrast_umass))
    # Print the number of each contrast
    contrast_count_umass = {}
    for contrast in set(list_contrast_umass):
        contrast_count_umass[contrast] = list_contrast_umass.count(contrast)
    print("Number of images per contrast in umass: ", contrast_count_umass)

    return None


if __name__ == '__main__':
    main()