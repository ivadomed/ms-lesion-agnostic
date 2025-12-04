"""
This code performs lesion mapping from baseline to follow-up MRI scans using a machine learning model.

Input:
    -i: msd dataset path
    -pred: path to the folder containing the predicted segmentations (SC, lesions, centerline, levels ...)
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
    parser.add_argument('-pred', '--pred', type=str, required=True, help='Path to the folder containing the predicted segmentations')
    parser.add_argument('-o', '--output-folder', type=str, required=True, help='Path to the output folder where lesion matching results will be stored')
    return parser.parse_args()


def build_dataset(input_msd_dataset, pred_folder, output_folder):
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

        # Initialize file names for lesions, sc and disc levels at both timepoints
        image_1_name = Path(image1).name
        image_2_name = Path(image2).name
        # Initialize the names
        sc_seg_1 = os.path.join(pred_folder, image_1_name.replace('.nii.gz', '_sc-seg.nii.gz'))
        sc_seg_2 = os.path.join(pred_folder, image_2_name.replace('.nii.gz', '_sc-seg.nii.gz'))
        centerline_1 = os.path.join(pred_folder, image_1_name.replace('.nii.gz', '_centerline.nii.gz'))
        centerline_2 = os.path.join(pred_folder, image_2_name.replace('.nii.gz', '_centerline.nii.gz'))
        lesion_seg_1 =  os.path.join(pred_folder, image_1_name.replace('.nii.gz', '_lesion-seg.nii.gz'))
        lesion_seg_2 =  os.path.join(pred_folder, image_2_name.replace('.nii.gz', '_lesion-seg.nii.gz'))
        levels_1 = os.path.join(pred_folder, image_1_name.replace('.nii.gz', '_levels.nii.gz'))
        levels_2 = os.path.join(pred_folder, image_2_name.replace('.nii.gz', '_levels.nii.gz'))
        registered_image2_to_1 = os.path.join(temp_folder, image_2_name.replace('.nii.gz', '_registered_to_' + image_1_name))
        warping_field_img2_to_1 = os.path.join(temp_folder, image_2_name.replace('.nii.gz', '_warp_to_' + image_1_name))
        inv_warping_field_img2_to_1 = os.path.join(temp_folder, image_2_name.replace('.nii.gz', '_inv_warp_to_' + image_1_name))
        lesion_seg_2_reg = os.path.join(temp_folder, image_2_name.replace('.nii.gz', '_lesion-seg-reg.nii.gz'))
        labeled_lesion_seg_1 = os.path.join(output_folder, image_1_name.replace('.nii.gz', '_lesion-seg-labeled.nii.gz'))
        labeled_lesion_seg_2 = os.path.join(output_folder, image_2_name.replace('.nii.gz', '_lesion-seg-labeled.nii.gz'))
        labeled_lesion_seg_2_reg = os.path.join(temp_folder, image_2_name.replace('.nii.gz', '_lesion-seg-reg-labeled.nii.gz'))

        # We label the centerline
        labeled_centerline_1 = os.path.join(temp_folder, image_1_name.replace('.nii.gz', '_labeled-centerline.nii.gz'))
        labeled_centerline_2 = os.path.join(temp_folder, image_2_name.replace('.nii.gz', '_labeled-centerline.nii.gz'))
        label_centerline(centerline_1, levels_1, labeled_centerline_1)
        label_centerline(centerline_2, levels_2, labeled_centerline_2)

        # For each lesion in the labeled mask at timepoint1, we get its coordinates and volume
        lesion_analysis_1 = analyze_lesions(labeled_lesion_seg_1)
        lesion_analysis_1 = compute_lesion_location(lesion_analysis_1, labeled_lesion_seg_1, sc_seg_1, labeled_centerline_1, levels_1)
        lesion_analysis_2 = analyze_lesions(labeled_lesion_seg_2)
        lesion_analysis_2 = compute_lesion_location(lesion_analysis_2, labeled_lesion_seg_2, sc_seg_2, labeled_centerline_2, levels_2)
        
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
    pred_folder = args.pred
    output_folder = args.output_folder

    # Build the dataset
    build_dataset(input_msd_dataset, output_folder)