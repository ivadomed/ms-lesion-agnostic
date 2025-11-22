"""
This file is used to generate ground truth labeled lesion segmentations and the corresponding lesion mapping files.

Input:
    -i : path to the input MSD dataset

Output:
    None

Author: Pierre-Louis Benveniste
"""
import os
import argparse
import json
from pathlib import Path
import sys
# Import the functions from utils in parent folder
file_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.abspath(os.path.join(file_path, ".."))
sys.path.insert(0, root_path)
from utils import label_lesion_seg
from single_input.map_lesions_registered_with_IoU import map_lesions_registered_with_IoU


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-msd', type=str, required=True, help='Path to the input MSD dataset')
    return parser.parse_args()


def main():
    args = parse_args()
    input_msd_dataset = args.input_msd

    # Load the msd dataset
    with open(input_msd_dataset, 'r') as f:
        msd_data = json.load(f)
    data = msd_data['data']

    # We run generation now on all the data
    for subject in data:
        subject_id = subject
        # Initialize the timepoints and images
        timepoint1 = "ses-M0"
        timepoint2 = "ses-M12"
        input_image1 = data[subject][timepoint1][0]
        input_image2 = data[subject][timepoint2][0]
        # Build path to the GT segmentations and GT lesion mapping file
        gt_lesion_seg_1 = input_image1.replace('canproco', 'canproco/derivatives/labels-ms-spinal-cord-only').replace('.nii.gz', '_lesion-manual.nii.gz')
        gt_lesion_seg_2 = input_image2.replace('canproco', 'canproco/derivatives/labels-ms-spinal-cord-only').replace('.nii.gz', '_lesion-manual.nii.gz')
        # Initialize the output files
        labeled_gt_lesion_seg_1 = gt_lesion_seg_1.replace('_lesion-manual.nii.gz', '_lesion-manual-labeled.nii.gz')
        labeled_gt_lesion_seg_2 = gt_lesion_seg_2.replace('_lesion-manual.nii.gz', '_lesion-manual-labeled.nii.gz')
        gt_lesion_mapping_file = str(Path(gt_lesion_seg_1).parent).replace('ses-M0/anat', 'lesion-mapping.json')
        
        # We label the GT lesion segmentations
        label_lesion_seg(gt_lesion_seg_1, labeled_gt_lesion_seg_1)
        label_lesion_seg(gt_lesion_seg_2, labeled_gt_lesion_seg_2)

        # Create an output folder for lesion mapping results
        output_folder = str(Path(gt_lesion_seg_1).parent).replace('ses-M0/anat', 'output')
        os.makedirs(output_folder, exist_ok=True)

        # Generate the lesion mapping file for this subject using reg IoU
        
        lesion_mapping = map_lesions_registered_with_IoU(input_image1, input_image2, output_folder=output_folder, IoU_threshold=0.2, lesion_seg_1=labeled_gt_lesion_seg_1, lesion_seg_2=labeled_gt_lesion_seg_2)

        # Save the lesion mapping in a json file
        with open(gt_lesion_mapping_file, 'w') as f:
            json.dump(lesion_mapping, f, indent=4)

    return None


if __name__ == "__main__":
    main()