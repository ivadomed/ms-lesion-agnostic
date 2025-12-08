"""
This code creates maximum intensity projections (MIP) of label images along the axial axis for visualization purposes.
It is related to this issue: https://github.com/ivadomed/ms-lesion-agnostic/issues/88

Input:
    -i : path the msd dataset
    -o : path to the output folder where MIP projections will be stored

Output:
    None

Author: Pierre-Louis Benveniste
"""
import os
import sys
import argparse
from pathlib import Path
from loguru import logger
import nibabel as nib
import numpy as np
import json
# Import the functions from utils in parent folder
file_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.abspath(os.path.join(file_path, ".."))
sys.path.insert(0, root_path)
from utils import get_levels, keep_common_levels_only


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-msd', type=str, required=True, help='Path to the input MSD dataset')
    parser.add_argument('-o', '--output_folder', type=str, required=True, help='Path to the output folder where MIP projections will be stored')
    return parser.parse_args()


def main():
    args = parse_args()
    output_folder = args.output_folder
    input_msd = args.input_msd

    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Load the msd dataset
    with open(input_msd, 'r') as f:
        msd_data = json.load(f)
    data = msd_data['data']

    for subject in data:
        image1_path = data[subject]['ses-M0'][0]
        image2_path = data[subject]['ses-M12'][0]
        # Build label file path
        label1_path = image1_path.replace('canproco', 'canproco/derivatives/labels-ms-spinal-cord-only').replace('.nii.gz', '_lesion-manual.nii.gz')
        label2_path = image2_path.replace('canproco', 'canproco/derivatives/labels-ms-spinal-cord-only').replace('.nii.gz', '_lesion-manual.nii.gz')

        # Build output path for subject
        subject_output_folder = os.path.join(output_folder, subject)
        os.makedirs(subject_output_folder, exist_ok=True)

        # Segment both levels
        levels1 = os.path.join(subject_output_folder, 'levels_ses-M0.nii.gz')
        levels2 = os.path.join(subject_output_folder, 'levels_ses-M12.nii.gz')
        get_levels(image1_path, levels1)
        get_levels(image2_path, levels2)
        keep_common_levels_only(levels1, levels2)

        # Register img2 to img1
        ## Build output warping field path
        warp_2_to_1_path = os.path.join(subject_output_folder, f'warp_ses-M12_to_ses-M0.nii.gz')
        registered_image2_to_1 = os.path.join(subject_output_folder, f'ses-M12_registered_to_ses-M0.nii.gz')
        parameter = "step=0,type=label,dof=Tx_Ty_Tz:step=1,type=im,algo=dl"
        assert os.system(f"sct_register_multimodal -i {image2_path} -d {image1_path} -param '{parameter}' -ilabel {levels2} -dlabel {levels1} -o {registered_image2_to_1} -owarp {warp_2_to_1_path} ") == 0, "Registration failed"

        # Apply warping field to label2
        registered_label2 = os.path.join(subject_output_folder, f'lesion-manual_ses-M12_registered_to_ses-M0.nii.gz')
        assert os.system(f"sct_apply_transfo -i {label2_path} -d {image1_path} -w {warp_2_to_1_path} -o {registered_label2} -x nn") == 0, "Applying warping field to label failed"

        # Now we load both segmentations and create MIP projections along R-L axis
        label1_data = nib.load(label1_path).get_fdata()
        registered_label2_data = nib.load(registered_label2).get_fdata()
        # Find the R-L direction
        label1_img = nib.load(label1_path)
        registered_label2_img = nib.load(registered_label2)
        label1_ornt = nib.orientations.aff2axcodes(label1_img.affine)
        label2_ornt = nib.orientations.aff2axcodes(registered_label2_img.affine)
        rl_axis_1 = label1_ornt.index('L') if 'L' in label1_ornt else label1_ornt.index('R')
        rl_axis_2 = label2_ornt.index('L') if 'L' in label2_ornt else label2_ornt.index('R')
        # Create MIP projections
        mip_label1 = np.max(label1_data, axis=rl_axis_1)
        mip_registered_label2 = np.max(registered_label2_data, axis=rl_axis_2)
        # Build output paths
        mip_label1_path = os.path.join(subject_output_folder, f'ses-M0_lesion-manual_MIP_RL.nii.gz')
        mip_registered_label2_path = os.path.join(subject_output_folder, f'ses-M12_registered_to_ses-M0_lesion-manual_MIP_RL.nii.gz')
        # Save MIP projections
        mip_label1_img = nib.Nifti1Image(mip_label1, affine=label1_img.affine)
        mip_registered_label2_img = nib.Nifti1Image(mip_registered_label2, affine=registered_label2_img.affine)
        nib.save(mip_label1_img, mip_label1_path)
        nib.save(mip_registered_label2_img, mip_registered_label2_path)
        # we create a SC seg which is the full slice with only value 1 for QC purposes
        sc_seg1 = np.ones(label1_data.shape)
        sc_seg1_path = os.path.join(subject_output_folder, f'ses-M0_sc_seg.nii.gz')
        sc_seg1_img = nib.Nifti1Image(sc_seg1, affine=label1_img.affine)
        nib.save(sc_seg1_img, sc_seg1_path)
        
        # Generate the QC
        qc_folder = os.path.join(output_folder, "qc")
        os.makedirs(qc_folder, exist_ok=True)
        assert os.system(f'sct_qc -i {mip_label1_path} -p sct_deepseg_lesion -d {mip_registered_label2_path} -s {sc_seg1_path} -plane sagittal -qc {qc_folder} ') == 0, "Failed to generate QC"

    return None


if __name__ == "__main__":
    main()