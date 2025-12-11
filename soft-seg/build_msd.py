"""
This file builds the MSD dataset for the soft segmentation tasks.
It uses data from ns-ucsf-2025.

Arguments:
    -i: Path to the ms-ucsf-2025 BIDS dataset
    -o: Output directory

Output:
    None

Author: Pierre-Louis Benveniste
"""
import os
import argparse
from pathlib import Path
from tqdm import tqdm
import json


def parse_args():
    parser = argparse.ArgumentParser(description='Build MSD dataset for soft segmentation tasks')
    parser.add_argument('-i', '--input', type=str, required=True, help='Path to the ms-ucsf-2025 BIDS dataset')
    parser.add_argument('-o', '--output', type=str, required=True, help='Output directory')
    return parser.parse_args()


def main():

    # Parse arguments
    args = parse_args()
    input_msd = args.input
    output_dir = args.output
    
    # Build the output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize the output dict
    dict_subs_sessions = {}

    # List all the subjects in the input directory
    subjects = [d for d in os.listdir(input_msd) if os.path.isdir(os.path.join(input_msd, d))]
    subjects = [s for s in subjects if s.startswith('sub-')]
    subjects.sort()

    count_session = 0
    count_subject = 0
    # Loop through each subject
    for subject in tqdm(subjects):
        count_subject += 1
        # For each subject we get the sessions
        sessions = [d for d in os.listdir(os.path.join(input_msd, subject))]
        sessions = [s for s in sessions if s.startswith('ses-')]
        print(f"Processing subject {subject} with sessions {sessions}")
        # Initialize the subject in the dictionary
        dict_subs_sessions[subject] = {}
        # Loop through each session
        for session in sessions:
            count_session += 1
            session_path = os.path.join(input_msd, subject, session, 'anat')
            # Get all images in the session directory
            images = list(Path(session_path).rglob('*.nii.gz'))
            images = [str(img) for img in images if 'SHA256' not in str(img)]
            images.sort()  # Sort the images for consistency
            # Add the images to the dictionary
            dict_subs_sessions[subject][session] = {}
            dict_subs_sessions[subject][session]['images'] = images

    print(f"Number of subjects: {count_subject}")
    print(f"Number of sessions: {count_session}")

    # Create the dict for the MSD dataset
    msd_dataset = {}
    msd_dataset['name'] = 'dataset_for_soft_segmentation'
    msd_dataset['description'] = 'This is an msd dataset for the evaluation of soft-segmentation methods'
    msd_dataset['modality'] = {"0": "MRI"}
    msd_dataset['number_of_subjects'] = count_subject
    msd_dataset['number_of_sessions'] = count_session
    msd_dataset['data'] = dict_subs_sessions

    # We save the json file
    json_path = os.path.join(output_dir, f'dataset_soft_seg.json')
    with open(json_path, 'w') as f:
        json.dump(msd_dataset, f, indent=4)
    

if __name__ == "__main__":
    main()