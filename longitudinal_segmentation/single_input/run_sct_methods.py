"""
This file runs the different SCT commands to provide the segmentations of the SC, lesions, centerline and registration between two timepoints.
It runs on all images of the msd dataset provided as input.

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
from tqdm import tqdm
from pathlib import Path
import sys
# Import the functions from utils in parent folder
file_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.abspath(os.path.join(file_path, ".."))
sys.path.insert(0, root_path)
from utils import segment_sc, segment_lesions, get_levels, keep_common_levels_only, compute_lesion_CoM, label_lesion_seg, correct_labeling, get_centerline


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-msd', type=str, required=True, help='Path to the input MSD dataset')
    parser.add_argument('-o', '--output-folder', type=str, required=True, help='Path to the output folder where lesion matching results will be stored')
    return parser.parse_args()


def main():
    args = parse_args()
    input_msd_dataset = args.input_msd
    output_folder = args.output_folder

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
        # Build a QC folder
        qc_folder = os.path.join(output_folder, 'qc')
        os.makedirs(qc_folder, exist_ok=True)
        # Build subject output folder
        subject_output_folder = os.path.join(output_folder, subject_id)
        os.makedirs(subject_output_folder, exist_ok=True)
        # Build the a temp folder
        temp_folder = os.path.join(subject_output_folder, 'temp')
        os.makedirs(temp_folder, exist_ok=True)
        
        image_1_name = Path(input_image1).name
        image_2_name = Path(input_image2).name
        # Copy image to the subject output folder
        copy_image1 = os.path.join(subject_output_folder, image_1_name)
        copy_image2 = os.path.join(subject_output_folder, image_2_name)
        assert os.system(f"cp {input_image1} {copy_image1}") == 0, "Failed to copy image 1"
        assert os.system(f"cp {input_image2} {copy_image2}") == 0, "Failed to copy image 2"
        # SC segmentations
        sc_seg_1 = os.path.join(subject_output_folder, image_1_name.replace('.nii.gz', '_sc-seg.nii.gz'))
        sc_seg_2 = os.path.join(subject_output_folder, image_2_name.replace('.nii.gz', '_sc-seg.nii.gz'))
        levels_1 = os.path.join(subject_output_folder, image_1_name.replace('.nii.gz', '_levels.nii.gz'))
        levels_2 = os.path.join(subject_output_folder, image_2_name.replace('.nii.gz', '_levels.nii.gz'))
        levels_1_common = os.path.join(subject_output_folder, image_1_name.replace('.nii.gz', '_levels-common.nii.gz'))
        levels_2_common = os.path.join(subject_output_folder, image_2_name.replace('.nii.gz', '_levels-common.nii.gz'))
        # Centerline segmentations
        centerline_1 = os.path.join(subject_output_folder, image_1_name.replace('.nii.gz', '_centerline.nii.gz'))
        centerline_2 = os.path.join(subject_output_folder, image_2_name.replace('.nii.gz', '_centerline.nii.gz'))
        # Initialize file name for registered image 2
        registered_image2_to_1 = os.path.join(subject_output_folder, image_2_name.replace('.nii.gz', '_registered_to_' + image_1_name))
        warping_field_img2_to_1 = os.path.join(subject_output_folder, image_2_name.replace('.nii.gz', '_warp_to_' + image_1_name))
        inv_warping_field_img2_to_1 = os.path.join(subject_output_folder, image_2_name.replace('.nii.gz', '_inv_warp_to_' + image_1_name))
        # Initialize file name for labeled lesion segmentations
        lesion_seg_1 = os.path.join(subject_output_folder, image_1_name.replace('.nii.gz', '_lesion-seg.nii.gz'))
        lesion_seg_2 = os.path.join(subject_output_folder, image_2_name.replace('.nii.gz', '_lesion-seg.nii.gz'))
        # Initialize file name for lesion segmentation of image 2 registered to image 1
        lesion_seg_2_reg = os.path.join(subject_output_folder, image_2_name.replace('.nii.gz', '_lesion-seg-reg.nii.gz'))
        # Labeled lesion segmentations
        labeled_lesion_seg_1 = os.path.join(subject_output_folder, image_1_name.replace('.nii.gz', '_lesion-seg-labeled.nii.gz'))
        labeled_lesion_seg_2 = os.path.join(subject_output_folder, image_2_name.replace('.nii.gz', '_lesion-seg-labeled.nii.gz'))
        labeled_lesion_seg_2_reg = os.path.join(subject_output_folder, image_2_name.replace('.nii.gz', '_lesion-seg-reg-labeled.nii.gz'))

        # Run the SCT methods
        segment_sc(input_image1, sc_seg_1)
        segment_sc(input_image2, sc_seg_2)
        # Get centerlines
        get_centerline(sc_seg_1, centerline_1)
        get_centerline(sc_seg_2, centerline_2)
        # Get vertebral levels
        get_levels(input_image1, levels_1)
        get_levels(input_image2, levels_2)
        # Segment lesions
        segment_lesions(input_image1, sc_seg_1, qc_folder, lesion_seg_1, test_time_aug=True)
        segment_lesions(input_image2, sc_seg_2, qc_folder, lesion_seg_2, test_time_aug=True)
        # Keep only common levels
        keep_common_levels_only(levels_1, levels_2, levels_1_common, levels_2_common)
        # Register image 2 to image 1
        parameter = "step=0,type=label,dof=Tx_Ty_Tz:step=1,type=im,algo=dl"
        assert os.system(f"sct_register_multimodal -i {input_image2} -d {input_image1} -param '{parameter}' -ilabel {levels_2_common} -dlabel {levels_1_common} -o {registered_image2_to_1} -owarp {warping_field_img2_to_1} -owarpinv {inv_warping_field_img2_to_1} -dseg {sc_seg_1} -qc {qc_folder} ") == 0, "Registration failed"
        # # We warp the lesion segmentation of image 2 to image 1 space using a linear interpolation
        assert os.system(f"sct_apply_transfo -i {lesion_seg_2} -d {input_image1} -w {warping_field_img2_to_1} -o {lesion_seg_2_reg} -x nn") == 0, "Failed to warp lesion segmentation of image 2"
        # Label the lesion segmentations
        label_lesion_seg(lesion_seg_1, labeled_lesion_seg_1)
        label_lesion_seg(lesion_seg_2, labeled_lesion_seg_2)
        label_lesion_seg(lesion_seg_2_reg, labeled_lesion_seg_2_reg)

    return None


if __name__ == "__main__":
    main()