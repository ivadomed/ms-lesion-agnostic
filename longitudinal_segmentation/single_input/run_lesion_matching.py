"""
This file runs one of the 3 methods for lesion matching between two timepoints.
It runs on all images of the msd dataset provided as input.

Input:
    -i: msd dataset path
    -o: output folder where to store the lesion matching results
    -m: method to use for lesion matching. Choices are:
        - registered_with_CoM
        - registered_with_IoU
        - unregistered
    -gt: whether to use ground truth lesion segmentations (default: False)

Output:
    None

Authors: Pierre-Louis Benveniste
"""
import os
import argparse
import json
from tqdm import tqdm
from map_lesions_registered_with_CoM import map_lesions_registered_with_CoM
from map_lesions_registered_with_IoU import map_lesions_registered_with_IoU
from map_lesions_unregistered import map_lesions_unregistered


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-msd', type=str, required=True, help='Path to the input MSD dataset')
    parser.add_argument('-o', '--output-folder', type=str, required=True, help='Path to the output folder where lesion matching results will be stored')
    parser.add_argument('-m', '--method', type=str, required=True, choices=['registered_with_CoM', 'registered_with_IoU', 'unregistered'], help='Method to use for lesion matching')
    parser.add_argument('-gt', '--ground-truth', action='store_true', help='Whether to use ground truth lesion segmentations')
    return parser.parse_args()


def main():
    args = parse_args()
    input_msd_dataset = args.input_msd
    output_folder = args.output_folder
    method = args.method

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Load the msd dataset
    with open(input_msd_dataset, 'r') as f:
        msd_data = json.load(f)
    data = msd_data['data']

    # We run lesion matching now on all the data
    for subject in tqdm(data):
        subject_id = subject
        # Initialize the timepoints and images
        timepoint1 = "ses-M0"
        timepoint2 = "ses-M12"
        input_image1 = data[subject][timepoint1][0]
        input_image2 = data[subject][timepoint2][0]
        # Build subject output folder
        subject_output_folder = os.path.join(output_folder, subject_id)
        os.makedirs(subject_output_folder, exist_ok=True)

        if method == 'registered_with_CoM':
            lesion_mapping = map_lesions_registered_with_CoM(input_image1, input_image2, subject_output_folder, GT_lesion=args.ground_truth)
        elif method == 'registered_with_IoU':
            lesion_mapping = map_lesions_registered_with_IoU(input_image1, input_image2, subject_output_folder, GT_lesion=args.ground_truth)
        elif method == 'unregistered':
            lesion_mapping = map_lesions_unregistered(input_image1, input_image2, subject_output_folder, GT_lesion=args.ground_truth)
        
        # Save the lesion mapping in a json file
        mapping_output_file = os.path.join(subject_output_folder, 'lesion_mapping.json')
        with open(mapping_output_file, 'w') as f:
            json.dump(lesion_mapping, f, indent=4)

    return None


if __name__ == "__main__":
    main()