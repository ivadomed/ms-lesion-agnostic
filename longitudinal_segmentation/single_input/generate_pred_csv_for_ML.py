"""
This code performs lesion mapping from baseline to follow-up MRI scans using a machine learning model.

Input:
    -i: msd dataset path
    -o: output folder where to store the lesion matching results

Output:
    None

Authors: Pierre-Louis Benveniste
"""
import os
import argparse
import json
import sys
from pathlib import Path
import pandas as pd
from loguru import logger
from datetime import date
from tqdm import tqdm
# Import the functions from utils in parent folder
file_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.abspath(os.path.join(file_path, ".."))
sys.path.insert(0, root_path)
from utils import segment_sc, segment_lesions, get_centerline, get_levels, label_lesion_seg
from single_input.map_lesions_unregistered import label_centerline, analyze_lesions, compute_lesion_location

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-msd', type=str, required=True, help='Path to the input MSD dataset')
    parser.add_argument('-o', '--output-folder', type=str, required=True, help='Path to the output folder where lesion matching results will be stored')
    return parser.parse_args()


def build_dataset(input_msd_dataset, output_folder):
    """
    This function is used to build the dataset for training the model and lesion mapping.
    For each lesion in the baseline scan and follow-up scan, we add them to the dataset alongside their coordinates and volume.
    Input:
        - input_msd_dataset: Path to the MSD dataset folder.
        - output_folder: Path to the folder where the dataset will be saved.

    Output:
        None
    """
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Add a logger
    logger.add(os.path.join(output_folder, f'logger_{str(date.today())}.log'))

    # Load the msd dataset
    with open(input_msd_dataset, 'r') as f:
        msd_data = json.load(f)
    data = msd_data['data']

    dataset = []

    # Remove subjects that are not from Toronto site
    data = {k: v for k, v in data.items() if 'tor' in k}

    # Loop through each subject
    for subject in tqdm(data):

        # Build subject output folder
        subject_output_folder = os.path.join(output_folder, subject)
        os.makedirs(subject_output_folder, exist_ok=True)
        # Build a temp folder for intermediate results
        temp_folder = os.path.join(subject_output_folder, 'temp')
        os.makedirs(temp_folder, exist_ok=True)
        # Build a qc folder
        qc_folder = os.path.join(temp_folder, 'qc')
        os.makedirs(qc_folder, exist_ok=True)

        # Define timepoints
        timepoint1 = "ses-M0"
        timepoint2 = "ses-M12"

        ## We extract image name
        image1 = data[subject][timepoint1][0]
        image2 = data[subject][timepoint2][0]

        # Build the outputs paths
        lesion_seg_1 = os.path.join(temp_folder, 'lesion_seg_1.nii.gz')
        lesion_seg_2 = os.path.join(temp_folder, 'lesion_seg_2.nii.gz')
        labeled_mask1 = os.path.join(subject_output_folder, Path(image1).name.replace('.nii.gz', '_lesion-seg-labeled.nii.gz'))
        labeled_mask2 = os.path.join(subject_output_folder, Path(image2).name.replace('.nii.gz', '_lesion-seg-labeled.nii.gz'))
        sc_seg_1 = os.path.join(temp_folder, 'sc_seg_1.nii.gz')
        sc_seg_2 = os.path.join(temp_folder, 'sc_seg_2.nii.gz')
        centerline_1 = os.path.join(temp_folder, 'centerline_1.nii.gz')
        centerline_2 = os.path.join(temp_folder, 'centerline_2.nii.gz')
        levels_1 = os.path.join(temp_folder, 'levels_1.nii.gz')
        levels_2 = os.path.join(temp_folder, 'levels_2.nii.gz')
        labeled_centerline_1 = os.path.join(temp_folder, 'labeled_centerline_1.nii.gz')
        labeled_centerline_2 = os.path.join(temp_folder, 'labeled_centerline_2.nii.gz')

        # # Now for both timepoint, we compute lesion volume and coordinates in the anatomical space
        # # Segment the spinal cord
        # segment_sc(image1, sc_seg_1)
        # segment_sc(image2, sc_seg_2)
        # # Get the centerline
        # get_centerline(sc_seg_1, centerline_1)
        # get_centerline(sc_seg_2, centerline_2)
        # # Get the levels
        # get_levels(image1, levels_1)
        # get_levels(image2, levels_2)
        # # segment lesions
        # segment_lesions(image1, sc_seg_1, qc_folder, lesion_seg_1, test_time_aug=True)
        # segment_lesions(image2, sc_seg_2, qc_folder, lesion_seg_2, test_time_aug=True)
        # # Label the lesion segmentation
        label_lesion_seg(lesion_seg_1, labeled_mask1)
        label_lesion_seg(lesion_seg_2, labeled_mask2)

        # # We label the centerline
        # label_centerline(centerline_1, levels_1, labeled_centerline_1)
        # label_centerline(centerline_2, levels_2, labeled_centerline_2)


        # For each lesion in the labeled mask at timepoint1, we get its coordinates and volume
        lesion_analysis_1 = analyze_lesions(labeled_mask1)
        lesion_analysis_1 = compute_lesion_location(lesion_analysis_1, labeled_mask1, sc_seg_1, labeled_centerline_1, levels_1)
        lesion_analysis_2 = analyze_lesions(labeled_mask2)
        lesion_analysis_2 = compute_lesion_location(lesion_analysis_2, labeled_mask2, sc_seg_2, labeled_centerline_2, levels_2)
        
        # Now we add the lesions to the dataset
        for lesion_id, lesion_info in lesion_analysis_1.items():
            dataset.append({
                'subject': subject,
                'timepoint': timepoint1,
                'group': None,
                'lesion_id': lesion_id,
                'z': lesion_info['centerline_z'],
                'r': lesion_info['radius_mm'],
                'theta': lesion_info['theta'],
                'volume': lesion_info['volume_mm3'],
                'diameter_RL': lesion_info['diameter_RL_mm'],
                'diameter_AP': lesion_info['diameter_AP_mm'],
                'diameter_SI': lesion_info['diameter_SI_mm']
            })
    
        for lesion_id, lesion_info in lesion_analysis_2.items():
            dataset.append({
                'subject': subject,
                'timepoint': timepoint2,
                'group': None,
                'lesion_id': lesion_id,
                'z': lesion_info['centerline_z'],
                'r': lesion_info['radius_mm'],
                'theta': lesion_info['theta'],
                'volume': lesion_info['volume_mm3'],
                'diameter_RL': lesion_info['diameter_RL_mm'],
                'diameter_AP': lesion_info['diameter_AP_mm'],
                'diameter_SI': lesion_info['diameter_SI_mm']
            })
        
    # Convert to pandas dataframe
    dataset = pd.DataFrame(dataset)
    # Save the dataset
    dataset_path = os.path.join(output_folder, 'lesion_dataset_pred_lesion.csv')
    dataset.to_csv(dataset_path, index=False)
    logger.info(f"Dataset saved to {dataset_path}")
        
    return None


if __name__ == "__main__":  
    args = parse_args()
    input_msd_dataset = args.input_msd
    output_folder = args.output_folder

    # Build the dataset
    build_dataset(input_msd_dataset, output_folder)