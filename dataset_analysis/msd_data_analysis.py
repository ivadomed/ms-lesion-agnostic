"""
This file was created to analyze the msd dataset used for training and testing our dataset. 
It takes as input the msd dataset and analysis the properties of the dataset.

Input:
    --msd-data-path: path to the msd dataset in json format
    --output-folder: path to the output folder where the analysis will be saved

Output:
    None

Example:
    python dataset_analysis/msd_data_analysis.py --msd-data-path /path/to/msd/data --output-folder /path/to/output/folder

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
from loguru import logger
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--msd-data-path', type=str, required=True, help='Path to the MSD dataset')
    parser.add_argument('--output-folder', type=str, required=True, help='Path to the output folder')
    return parser.parse_args()


def main():
    # Parse arguments
    args = parse_args()
    msd_data_path = args.msd_data_path
    output_folder = args.output_folder
    
    # Build the output folder
    os.makedirs(output_folder, exist_ok=True)

    # Load the dataset
    with open(msd_data_path, 'r') as f:
        msd_data = json.load(f)
    
    # Get data
    data = msd_data['train'] + msd_data['validation'] + msd_data['test'] + msd_data['externalValidation']

    # Create the logger file
    log_file = os.path.join(output_folder, f'{Path(msd_data_path).name.split(".json")[0]}_analysis.txt')
    # Clear the log file
    with open(log_file, 'w') as f:
        f.write('')
    logger.add(log_file)

    # Log some basic stuff
    logger.info(f"MSD dataset: {Path(msd_data_path)}")
    logger.info(f"Number of images: {len(data)}")
    logger.info(f"Number of images for training: {(msd_data['numTraining'])}")
    logger.info(f"Number of images for validation: {(msd_data['numValidation'])}")
    logger.info(f"Number of images for testing: {(msd_data['numTest'])}")
    logger.info(f"Number of images for external validation: {(msd_data['numExternalValidation'])}")

    # In the dataset replace all the MEGRE contrasts by T2star
    for image in data:
        if image['contrast'] == 'MEGRE':
            image['contrast'] = 'T2star'

    # Count the number of images per contrast
    contrast_count = {}
    for image in data:
        contrast = image['contrast']
        if contrast not in contrast_count:
            contrast_count[contrast] = 0
        contrast_count[contrast] += 1
    logger.info(f"Number of images per contrast: {contrast_count}")

    # Count the number of images per site
    site_count = {}
    for image in data:
        site = image['site']
        if site not in site_count:
            site_count[site] = 0
        site_count[site] += 1
    logger.info(f"Number of images per site: {site_count}")

    # Count the acquisition of images
    acquisition_count = {}
    for image in data:
        acquisition = image['acquisition']
        if acquisition not in acquisition_count:
            acquisition_count[acquisition] = 0
        acquisition_count[acquisition] += 1
    logger.info(f"Number of images per acquisition: {acquisition_count}")

    # Now we count the number of subjects
    subjects = []
    for image in data:
        sub = image['image'].split('/')[-1].split('_')[0]
        dataset = image['site']
        subject = dataset + '/' + sub
        subjects.append(subject)
    logger.info(f"Number of subjects: {len(set(subjects))}")   

    # Print the number of sites:
    logger.info(f"Number of sites: {len(set([image['site'] for image in data]))}")

    # Now we will look at the average resolution of the images
    resolutions = []
    for image in tqdm(data):
        resolutions.append(image['resolution'])
    logger.info(f"Average resolution (RPI): {np.mean(resolutions, axis=0)}")
    logger.info(f"Std resolution (RPI): {np.std(resolutions, axis=0)}")
    logger.info(f"Median resolution (RPI): {np.median(resolutions, axis=0)}")
    logger.info(f"Minimum pixel dimension (RPI): {np.min(resolutions)}")
    logger.info(f"Maximum pixel dimension (RPI): {np.max(resolutions)}")

    # Now we count the field strength of the images
    field_strength = []
    count_field_strength = {}
    for image in tqdm(data):
        sidecar = image['image'].replace('.nii.gz', '.json')
        # if the sidecar does not exist, we skip the image
        if not os.path.exists(sidecar):
            continue
        with open(sidecar, 'r') as f:
            try:
                metadata = json.load(f)  # Remplacez 'response' par votre source de données
            except json.JSONDecodeError as e:
                continue
        # if field "MagneticFieldStrength" does not exist, we skip the image
        if "MagneticFieldStrength" not in metadata:
            continue
        field_strength.append(metadata["MagneticFieldStrength"])
        # Count the field strength
        if metadata["MagneticFieldStrength"] not in count_field_strength:
            count_field_strength[metadata["MagneticFieldStrength"]] = 0
        count_field_strength[metadata["MagneticFieldStrength"]] += 1
    logger.info(f"Field strength for MSD dataset: {set(field_strength)}")
    logger.info(f"Count of field strength for MSD dataset: {count_field_strength}")

    logger.info("-------------------------------------")

    # Now we want to display the following table : 
    # | Site      | Contrast | Acquisition | Orientation | Count | Avg resolution (RPI) | Number of subjects |
    # |-----------|----------|-------------|-------------|-------|----------------------|--------------------|
    # | canproco  | PSIR     | 2D          | sagittal    | 100   | 0.1x0.1x0.5          | 100                |
    # | canproco  | PSIR     | 2D          | axial       | 65    | 0.1x0.1x0.5          | 65                 |
    # | canproco  | STIR     | 2D          | sagittal    | 200   | 0.5x0.5x0.6          | 200                |
    # | canproco  | /        | /           | /           | 365   | 0.4x0.4x0.55         | 200                |

    # Create a pandas DataFrame to store the data
    df = pd.DataFrame(columns=['Site', 'Contrast', 'Acquisition', 'Orientation', 'Count', 'Avg resolution (R-L)', 'Avg resolution (P-A)', 'Avg resolution (I-S)', 'Number of subjects'])
    ## Add the data to the DataFrame
    for image in data:
        dataset = image['site']
        contrast = image['contrast']
        # For acquisition, if 3D ok, if sag or axial, then 2D
        if image['acquisition'] == '3D':
            acquisition = '3D'
        elif image['acquisition'] in ['sag', 'ax']:
            acquisition = '2D'
        # for orientation, if axial or sagittal, then we keep it, else we put /
        if image['acquisition'] in ['ax', 'sag']:
            orientation = image['acquisition']
        else:
            orientation = '/'
        resolution = image['resolution']
        sub = image['image'].split('/')[-1].split('_')[0]
        subject = dataset + '/' + sub
        # Add the data to the DataFrame
        new_row = {
            'Site': dataset,
            'Contrast': contrast,
            'Acquisition': acquisition,
            'Orientation': orientation,
            'Count': 1,
            'Avg resolution (R-L)': resolution[0],
            'Std resolution (R-L)': resolution[0],
            'Avg resolution (P-A)': resolution[1],
            'Std resolution (P-A)': resolution[1],
            'Avg resolution (I-S)': resolution[2],
            'Std resolution (I-S)': resolution[2],
            'Number of subjects': subject
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    # Group the DataFrame by Site, Contrast, Acquisition, Orientation and sum the Count and Number of subjects and average the Avg resolution (RPI)
    df_grouped = df.groupby(['Site', 'Contrast', 'Acquisition', 'Orientation']).agg({
        'Count': 'sum',
        'Avg resolution (R-L)': 'mean',
        'Std resolution (R-L)': 'std',
        'Avg resolution (P-A)': 'mean',
        'Std resolution (P-A)': 'std',
        'Avg resolution (I-S)': 'mean',
        'Std resolution (I-S)': 'std',
        'Number of subjects': 'nunique'
    })
    # Reset the index
    df_grouped = df_grouped.reset_index()
    # Log the DataFrame
    logger.info("DataFrame with the number of images per site, contrast, acquisition and orientation:")
    logger.info(df_grouped.to_string(index=False))

    return None


if __name__ == '__main__':
    main()