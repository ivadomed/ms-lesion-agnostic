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
import numpy as np
from pathlib import Path
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
        data = json.load(f)
    
    # Create the logger file
    log_file = os.path.join(output_folder, f'{Path(msd_data_path).name.split(".json")[0]}_analysis.txt')
    # Clear the log file
    with open(log_file, 'w') as f:
        f.write('')
    logger.add(log_file)

    # Log some basic stuff
    logger.info(f"MSD dataset: {Path(msd_data_path)}")
    logger.info(f"Number of images: {len(data)}")

    # Count the number of images per contrast
    contrast_count = {}
    for image in data:
        image = data[image]
        contrast = image['contrast']
        if contrast not in contrast_count:
            contrast_count[contrast] = 0
        contrast_count[contrast] += 1
    logger.info(f"Number of images per contrast: {contrast_count}")

    # Count the number of images per site
    site_count = {}
    for image in data:
        image = data[image]
        site = image['site']
        if site not in site_count:
            site_count[site] = 0
        site_count[site] += 1
    logger.info(f"Number of images per site: {site_count}")

    # We also count the number of subjects per site
    subjects_per_site = {}
    for image in data:
        image = data[image]
        dataset = image['site']
        sub = image['subject_id']
        subject = dataset + '/' + sub
        if dataset not in subjects_per_site:
            subjects_per_site[dataset] = set()
        subjects_per_site[dataset].add(subject)
    # Convert the sets to counts
    for site in subjects_per_site:
        subjects_per_site[site] = len(subjects_per_site[site])
    logger.info(f"\n Number of subjects per site: {subjects_per_site}")

    # Create a pandas DataFrame to store the data
    df = pd.DataFrame(columns=['Site', 'Contrast', 'Acquisition', 'Orientation', 'Count', 'Avg resolution (R-L)', 'Avg resolution (P-A)', 'Avg resolution (I-S)', 'Number of subjects'])
    ## Add the data to the DataFrame
    for image in data:
        image = data[image]
        dataset = image['site']
        contrast = image['contrast']
        acquisition = image['acquisition']
        orientation = image['orientation']
        resolution = image['resolution']
        field_strength = image['field_strength']
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
            'Number of subjects': subject,
            'Field strength': field_strength
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    # Group the DataFrame by Site, Contrast, Acquisition, Orientation and sum the Count and Number of subjects and average the Avg resolution (RPI)
    df_grouped = df.groupby(['Site', 'Contrast', 'Acquisition', 'Orientation', 'Field strength']).agg({
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
    # We add the number of subjects per site
    subjects_per_site_series = pd.Series(subjects_per_site, name='# Participants')
    df_grouped = df_grouped.merge(subjects_per_site_series, left_on='Site', right_index=True, how='left')
    # Reorder the columns
    df_grouped = df_grouped[['Site','# Participants','Field strength', 'Contrast', 'Acquisition', 'Orientation', 'Avg resolution (R-L)', 'Std resolution (R-L)', 'Avg resolution (P-A)', 'Std resolution (P-A)', 'Avg resolution (I-S)', 'Std resolution (I-S)', 'Count']]
    # Log the DataFrame
    logger.info("DataFrame with the number of images per site, contrast, acquisition, orientation and field strength:")
    logger.info(df_grouped.to_string(index=False))

    # Also save the DataFrame to a csv file
    csv_file = os.path.join(output_folder, 'csv_file.csv')
    df_grouped.to_csv(csv_file, index=False)

    return None


if __name__ == '__main__':
    main()