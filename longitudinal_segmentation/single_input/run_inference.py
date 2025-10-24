"""
This code runs inference wiht SCT on the msd dataset.
The output segmentations are stored in an output folder per subject.
It segments the spinal cord, the lesions and computes the centerline.

Input:
    -i : path to the input MSD dataset
    -o : path to the output folder where segmentations will be stored

Output:
    None

Example usage:
    python longitudinal_segmentation/late_fusion/run_inference.py -i /path/to/msd_dataset -o /path/to/output_folder

Authors: Pierre-Louis Benveniste
"""
import os
import argparse
import json
from pathlib import Path
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-msd', type=str, required=True, help='Path to the input MSD dataset')
    parser.add_argument('-o', '--output-folder', type=str, required=True, help='Path to the output folder where segmentations will be stored')
    return parser.parse_args()


def main():
    args = parse_args()
    input_msd_dataset = args.input_msd
    output_folder = args.output_folder

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    # Build a QC folder in the output folder
    qc_folder = os.path.join(output_folder, 'QC')
    os.makedirs(qc_folder, exist_ok=True)

    # Load the msd dataset
    with open(input_msd_dataset, 'r') as f:
        msd_data = json.load(f)
    data = msd_data['data']

    # Structure is the following:
    # "sub-mon126": {
    #         "ses-M0": [
    #             "/home/plbenveniste/net/longitudinal_ms/data/canproco/sub-mon126/ses-M0/anat/sub-mon126_ses-M0_PSIR.nii.gz"
    #         ],
    #         "ses-M12": [
    #             "/home/plbenveniste/net/longitudinal_ms/data/canproco/sub-mon126/ses-M12/anat/sub-mon126_ses-M12_PSIR.nii.gz"
    #         ]
    #     },

    # We run inference now on all the data
    for subject in tqdm(data):
        subject_id = subject
        # Create output folder for the subject
        subject_output_folder = os.path.join(output_folder, subject_id)
        os.makedirs(subject_output_folder, exist_ok=True)
        # For each session
        for session in data[subject]:
            # Build folder
            session_output_folder = os.path.join(subject_output_folder, session)
            os.makedirs(session_output_folder, exist_ok=True)
            session_id = session
            # For each image in the session (usually only one)
            for image in data[subject][session]:
                input_image_path = image
                # We want to segment the spinal cord
                output_sc_seg_path = os.path.join(session_output_folder, Path(input_image_path).name.replace('.nii.gz', '_sc_seg.nii.gz'))
                assert os.system(f'SCT_USE_GPU=1 sct_deepseg spinalcord -i {input_image_path} -o {output_sc_seg_path} ') == 0
                # For each image, we want the lesion segmentation
                output_lesion_seg_path = os.path.join(session_output_folder, Path(input_image_path).name.replace('.nii.gz', '_seg.nii.gz'))
                assert os.system(f'SCT_USE_GPU=1 sct_deepseg lesion_ms -i {input_image_path} -o {output_lesion_seg_path} -qc {qc_folder} -qc-plane Sagittal -qc-seg {output_sc_seg_path} ') == 0
                # For each image, we also want the vertebral levels
                output_vertebrae_path = os.path.join(session_output_folder, Path(input_image_path).name.replace('.nii.gz', '_levels.nii.gz'))
                assert os.system(f'SCT_USE_GPU=1 sct_deepseg totalspineseg -i {input_image_path} -step1-only 1 -o {output_vertebrae_path} ') == 0
                # Finally we also want the cord centerline
                output_centerline_path = os.path.join(session_output_folder, Path(input_image_path).name.replace('.nii.gz', '_centerline.nii.gz'))
                assert os.system(f'sct_get_centerline -i  {output_sc_seg_path} -method fitseg -o {output_centerline_path}') == 0

    print("Inference completed.")
        

if __name__ == "__main__":
    main()