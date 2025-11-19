"""
This code evaluates the lesion mapping results between two timepoints.
It computes segmentation metrics of the predicted segmentations against the ground truth.
It also computes lesion matching metrics based on the lesion mapping results and the ground truth mapping.

Input:
    -i : path to the input MSD dataset
    -p : path to the folder where predictions were stored
    -o : path to the output folder where evaluation results will be stored

Output:
    None
"""
import os
import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-msd', type=str, required=True, help='Path to the input MSD dataset')
    parser.add_argument('-p', '--predictions-folder', type=str, required=True, help='Path to the folder where predictions were stored')
    parser.add_argument('-o', '--output-folder', type=str, required=True, help='Path to the output folder where evaluation results will be stored')
    return parser.parse_args()


def main():
    args = parse_args()
    input_msd_dataset = args.input_msd
    predictions_folder = args.predictions_folder
    output_folder = args.output_folder

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Load the msd dataset
    with open(input_msd_dataset, 'r') as f:
        msd_data = json.load(f)
    data = msd_data['data']

    # We run evaluation now on all the data
    for subject in data:
        subject_id = subject
        # Initialize the timepoints and images
        timepoint1 = "ses-M0"
        timepoint2 = "ses-M12"
        input_image1 = data[subject][timepoint1][0]
        input_image2 = data[subject][timepoint2][0]
        # We build the path to the labeled predicted segmentations, lesion mapping file
        predicted_lesion_seg_1 = os.path.join(predictions_folder, subject_id, Path(input_image1).name.replace('.nii.gz', '_lesion-seg_labeled.nii.gz'))
        predicted_lesion_seg_2 = os.path.join(predictions_folder, subject_id, Path(input_image2).name.replace('.nii.gz', '_lesion-seg_labeled.nii.gz'))
        pred_lesion_mapping_file = os.path.join(predictions_folder, subject_id, 'lesion_mapping.json')
        # Build path to the GT segmentations and GT lesion mapping file
        gt_lesion_seg_1 = input_image1.replace('canproco', 'canproco/derivatives/labels-ms-spinal-cord-only').replace('.nii.gz', '_lesion-manual-labeled.nii.gz')
        gt_lesion_seg_2 = input_image2.replace('canproco', 'canproco/derivatives/labels-ms-spinal-cord-only').replace('.nii.gz', '_lesion-manual-labeled.nii.gz')
        gt_lesion_mapping_file = str(Path(gt_lesion_seg_1).parent).replace('ses-M0/anat', 'lesion-mapping.json')
        # print the file names
        print(f"Evaluating subject {subject_id}")
        print(f"Predicted lesion seg timepoint 1: {predicted_lesion_seg_1}")
        print(f"Predicted lesion seg timepoint 2: {predicted_lesion_seg_2}")
        print(f"GT lesion seg timepoint 1: {gt_lesion_seg_1}")
        print(f"GT lesion seg timepoint 2: {gt_lesion_seg_2}")
        print(f"Predicted lesion mapping file: {pred_lesion_mapping_file}")
        print(f"GT lesion mapping file: {gt_lesion_mapping_file}")


        break
    return None


if __name__ == "__main__":
    main()