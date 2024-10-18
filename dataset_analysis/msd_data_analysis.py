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
from tqdm import tqdm

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
    for image in tqdm(data):
        image_reoriented = Image(image['image']).change_orientation('RPI')
        resolution = image_reoriented.dim[4:7]
        resolution = [float(res) for res in resolution]
        resolutions.append(resolution)
        
    print("Average resolution: ", np.mean(resolutions, axis=0))
    print("Std resolution: ", np.std(resolutions, axis=0))
    print("Median resolution: ", np.median(resolutions, axis=0))
    print("Minimum pixel dimension: ", np.min(resolutions))
    print("Maximum pixel dimension: ", np.max(resolutions))

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

    # Now we will look at the average resolution of the images
    basel_data = ms_basel_2018 + ms_basel_2020
    ## Iterate over the images
    resolutions_basel = []
    orientation_basel = []
    for image in tqdm(basel_data):
        image_reoriented = Image(str(image)).change_orientation('RPI')
        resolution = image_reoriented.dim[4:7]
        resolution = [float(res) for res in resolution]
        resolutions_basel.append(resolution)
        if np.allclose(resolution, resolution[0], atol=1e-3):
            orientation = 'iso'
        elif np.argmax(resolution) == 0:
            orientation = 'sag'
        # Elif, the lowest arg is 1 then the orientation is coronal
        elif np.argmax(resolution) == 1:
            orientation = 'cor'
        # Else the orientation is axial
        else:
            orientation = 'ax'
        orientation_basel.append(orientation)
        
    print("Average resolution: ", np.mean(resolutions_basel, axis=0))
    print("Std resolution: ", np.std(resolutions_basel, axis=0))
    print("Median resolution: ", np.median(resolutions_basel, axis=0))
    print("Minimum pixel dimension: ", np.min(resolutions_basel))
    print("Maximum pixel dimension: ", np.max(resolutions_basel))

    # Count the number of images per orientation
    orientation_count_basel = {}
    for orientation in set(orientation_basel):
        orientation_count_basel[orientation] = orientation_basel.count(orientation)
    print("Number of images per orientation in basel: ", orientation_count_basel)

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

    # Remove : sub-ms1115_ses-01_acq-ax_ce-gad_T1w.nii.gz
    umass = [image for image in umass if 'sub-ms1115_ses-01_acq-ax_ce-gad_T1w' not in str(image)]
    # Remove: sub-ms1098_ses-01_acq-ax_ce-gad_T1w.nii.gz
    umass = [image for image in umass if 'sub-ms1098_ses-01_acq-ax_ce-gad_T1w' not in str(image)]
    # Remove: sub-ms1234_ses-03_acq-ax_ce-gad_T1w.nii.gz
    umass = [image for image in umass if 'sub-ms1234_ses-03_acq-ax_ce-gad_T1w' not in str(image)]

    list_contrast_umass = [str(image).split("/")[-1].split('_')[-1].split('.')[0] for image in umass]

    print("Number of images in umass: ", len(umass))
    print("Contrast in umass: ", set(list_contrast_umass))
    # Print the number of each contrast
    contrast_count_umass = {}
    for contrast in set(list_contrast_umass):
        contrast_count_umass[contrast] = list_contrast_umass.count(contrast)
    print("Number of images per contrast in umass: ", contrast_count_umass)

    # Now we will look at the average resolution of the images
    ## Iterate over the images
    resolutions_umass = []
    orientation_umass = []
    for image in tqdm(umass):
        image_reoriented = Image(str(image)).change_orientation('RPI')
        resolution = image_reoriented.dim[4:7]
        resolution = [float(res) for res in resolution]
        resolutions_umass.append(resolution)
        if np.allclose(resolution, resolution[0], atol=1e-3):
            orientation = 'iso'
        elif np.argmax(resolution) == 0:
            orientation = 'sag'
        # Elif, the lowest arg is 1 then the orientation is coronal
        elif np.argmax(resolution) == 1:
            orientation = 'cor'
            print(image)
        # Else the orientation is axial
        else:
            orientation = 'ax'
        orientation_umass.append(orientation)
        
    print("Average resolution: ", np.mean(resolutions_umass, axis=0))
    print("Std resolution: ", np.std(resolutions_umass, axis=0))
    print("Median resolution: ", np.median(resolutions_umass, axis=0))
    print("Minimum pixel dimension: ", np.min(resolutions_umass))
    print("Maximum pixel dimension: ", np.max(resolutions_umass))

    # Count the number of images per orientation
    orientation_count_umass = {}
    for orientation in set(orientation_umass):
        orientation_count_umass[orientation] = orientation_umass.count(orientation)
    print("Number of images per orientation in umass: ", orientation_count_umass)

    print("-------------------------------------")

    #############################################################
    # Now we can look at the external testing dataset ms-nmo-beijing
    path_beijing = os.path.join(dataset_path, 'ms-nmo-beijing')
    
    beijing = list(Path(path_beijing).rglob("*/anat/*.nii.gz"))

    # Keep only T1w images
    beijing = [image for image in beijing if 'T1w' in str(image)]
    # Remove Localizer images
    beijing = [image for image in beijing if 'Localizer' not in str(image)]
    beijing = [image for image in beijing if 'localizer' not in str(image)]



    list_contrast_beijing = [str(image).split("/")[-1].split('_')[-1].split('.')[0] for image in beijing]

    print("Number of images in beijing: ", len(beijing))
    print("Contrast in beijing: ", set(list_contrast_beijing))
    # Print the number of each contrast
    contrast_count_beijing = {}
    for contrast in set(list_contrast_beijing):
        contrast_count_beijing[contrast] = list_contrast_beijing.count(contrast)
    print("Number of images per contrast in beijing: ", contrast_count_beijing)

    # Now we will look at the average resolution of the images
    ## Iterate over the images
    resolutions_beijing = []
    orientations_beijing = []
    for image in tqdm(beijing):
        image_reoriented = Image(str(image)).change_orientation('RPI')
        resolution = image_reoriented.dim[4:7]
        resolution = [float(res) for res in resolution]
        resolutions_beijing.append(resolution)
        if np.allclose(resolution, resolution[0], atol=1e-3):
            orientation = 'iso'
        elif np.argmax(resolution) == 0:
            orientation = 'sag'
        # Elif, the lowest arg is 1 then the orientation is coronal
        elif np.argmax(resolution) == 1:
            orientation = 'cor'
            print(image)
        # Else the orientation is axial
        else:
            orientation = 'ax'
        orientations_beijing.append(orientation)

        
    print("Average resolution: ", np.mean(resolutions_beijing, axis=0))
    print("Std resolution: ", np.std(resolutions_beijing, axis=0))
    print("Median resolution: ", np.median(resolutions_beijing, axis=0))
    print("Minimum pixel dimension: ", np.min(resolutions_beijing))
    print("Maximum pixel dimension: ", np.max(resolutions_beijing))

    # Count the number of images per orientation
    orientation_count_beijing = {}
    for orientation in set(orientations_beijing):
        orientation_count_beijing[orientation] = orientations_beijing.count(orientation)
    print("Number of images per orientation in beijing: ", orientation_count_beijing)


    return None


if __name__ == '__main__':
    main()