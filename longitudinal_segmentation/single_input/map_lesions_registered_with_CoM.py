"""
This code compares lesion segmentations from two timepoints for a single subject using registration to match lesions.
The lesion matching is based on Hungarian algorithm applied to the center of mass (CoM) of lesions.

Input:
    -i1 : path to the input image at timepoint 1
    -i2 : path to the input image at timepoint 2
    -o : path to the output folder where comparison results will be stored

Output:
    None

Author: Pierre-Louis Benveniste
"""
import os
import argparse
from pathlib import Path
from loguru import logger
from datetime import date
import nibabel as nib
from scipy import ndimage
import numpy as np
from scipy.optimize import linear_sum_assignment
import json
import sys
# Import the functions from utils in parent folder
file_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.abspath(os.path.join(file_path, ".."))
sys.path.insert(0, root_path)
from utils import segment_sc, segment_lesions, get_levels, keep_common_levels_only, compute_lesion_CoM, label_lesion_seg, correct_labeling


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i1', '--input_image1', type=str, required=True, help='Path to the input image at timepoint 1')
    parser.add_argument('-i2', '--input_image2', type=str, required=True, help='Path to the input image at timepoint 2')
    parser.add_argument('-o', '--output_folder', type=str, required=True, help='Path to the output folder where comparison results will be stored')
    return parser.parse_args()


def compute_lesion_mapping(lesion_1_CoM, lesion_2_CoM):
    """
    This function computes the lesion mapping between two timepoints based on the CoM of lesions.

    Inputs:
        lesion_1_CoM : dict, mapping from lesion indices to their CoM at timepoint 1
        lesion_2_reg_CoM : dict, mapping from lesion indices to their CoM at timepoint 2 (registered to timepoint 1)

    Outputs:
        lesion_mapping_forward : dict, mapping from lesion indices from baseline to follow-up
    """
    # Compute cost matrix based on Euclidean distance between CoMs
    cost_matrix = np.zeros((len(lesion_1_CoM), len(lesion_2_CoM)))
    for i, (lesion1_id, com1) in enumerate(lesion_1_CoM.items()):
        for j, (lesion2_id, com2) in enumerate(lesion_2_CoM.items()):
            cost_matrix[i, j] = np.linalg.norm(np.array(com1) - np.array(com2))
    # Apply Hungarian algorithm to find optimal assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Create a lesion mapping report
    lesion_mapping = {}
    for i, j in zip(row_ind, col_ind):
        lesion1_id = int(list(lesion_1_CoM.keys())[i])
        lesion2_id = int(list(lesion_2_CoM.keys())[j])
        lesion_mapping[lesion1_id] = lesion2_id
    
    return lesion_mapping


def map_lesions_registered_with_CoM(input_image1, input_image2, output_folder):
    """
    This function performs lesion mapping between two timepoints using registered images and lesion matching based on the center of mass of lesions.

    Inputs:
        input_image1 : path to the input image at timepoint 1
        input_image2 : path to the input image at timepoint 2
        output_folder : path to the output folder where comparison results will be stored

    Outputs:
        lesion_mapping_1_to_2 : dict, mapping from lesion indices from timepoint 1 to timepoint 2 based on CoM
    """
    # Build output directory
    os.makedirs(output_folder, exist_ok=True)
    # Build temporary directory
    temp_folder = os.path.join(output_folder, "temp")
    os.makedirs(temp_folder, exist_ok=True)
    # Build QC folder
    qc_folder = os.path.join(output_folder, "qc")
    os.makedirs(qc_folder, exist_ok=True)

    # Build logger
    logger.add(os.path.join(output_folder, f'logger_{str(date.today())}.log'))

    # Copy both images to the output folder
    assert os.system(f'cp {input_image1} {output_folder}/') == 0, "Failed to copy image 1"
    assert os.system(f'cp {input_image2} {output_folder}/') == 0, "Failed to copy image 2"

    # Initialize file names for lesions, sc and disc levels at both timepoints
    image_1_name = Path(input_image1).name
    lesion_seg_1 = os.path.join(temp_folder, image_1_name.replace('.nii.gz', '_lesion-seg.nii.gz'))
    sc_seg_1 = os.path.join(temp_folder, image_1_name.replace('.nii.gz', '_sc-seg.nii.gz'))
    levels_1 = os.path.join(temp_folder, image_1_name.replace('.nii.gz', '_levels.nii.gz'))
    image_2_name = Path(input_image2).name
    lesion_seg_2 = os.path.join(temp_folder, image_2_name.replace('.nii.gz', '_lesion-seg.nii.gz'))
    sc_seg_2 = os.path.join(temp_folder, image_2_name.replace('.nii.gz', '_sc-seg.nii.gz'))
    levels_2 = os.path.join(temp_folder, image_2_name.replace('.nii.gz', '_levels.nii.gz'))
    # Initialize file name for registered image 2
    registered_image2_to_1 = os.path.join(temp_folder, image_2_name.replace('.nii.gz', '_registered_to_' + image_1_name))
    warping_field_img2_to_1 = os.path.join(temp_folder, image_2_name.replace('.nii.gz', '_warp_to_' + image_1_name))
    inv_warping_field_img2_to_1 = os.path.join(temp_folder, image_2_name.replace('.nii.gz', '_inv_warp_to_' + image_1_name))
    # Initialize file name for lesion segmentation of image 2 registered to image 1
    lesion_seg_2_reg = os.path.join(temp_folder, image_2_name.replace('.nii.gz', '_registered.nii.gz'))
    # Initialize file name for labeled lesion segmentations
    labeled_lesion_seg_1 = os.path.join(output_folder, image_1_name.replace('.nii.gz', '_lesion-seg-labeled.nii.gz'))
    labeled_lesion_seg_2 = os.path.join(output_folder, image_2_name.replace('.nii.gz', '_lesion-seg-labeled.nii.gz'))
    labeled_lesion_seg_2_reg = os.path.join(temp_folder, image_2_name.replace('.nii.gz', '_lesion-seg-registered-labeled.nii.gz'))

    # Segment the spinal cord
    segment_sc(input_image1, sc_seg_1)
    segment_sc(input_image2, sc_seg_2)
    # Segment the lesions
    segment_lesions(input_image1, sc_seg_1, qc_folder, lesion_seg_1, test_time_aug=True, soft_ms_lesion=True)
    segment_lesions(input_image2, sc_seg_2, qc_folder, lesion_seg_2, test_time_aug=True, soft_ms_lesion=True)
    # Get the levels
    get_levels(input_image1, levels_1)
    get_levels(input_image2, levels_2)

    # In the context of this registration, we need to only have common levels between both timepoints
    keep_common_levels_only(levels_1, levels_2)

    # Register image 2 to image 1
    parameter = "step=0,type=label,dof=Tx_Ty_Tz:step=1,type=im,algo=dl"
    assert os.system(f"sct_register_multimodal -i {input_image2} -d {input_image1} -param '{parameter}' -ilabel {levels_2} -dlabel {levels_1} -o {registered_image2_to_1} -owarp {warping_field_img2_to_1} -owarpinv {inv_warping_field_img2_to_1} -dseg {sc_seg_1} -qc {qc_folder} ") == 0, "Registration failed"

    # We warp the lesion segmentation of image 2 to image 1 space using a linear interpolation
    assert os.system(f"sct_apply_transfo -i {lesion_seg_2} -d {input_image1} -w {warping_field_img2_to_1} -o {lesion_seg_2_reg} -x linear") == 0, "Failed to warp lesion segmentation of image 2"
    
    # We label the lesion segmentation files
    label_lesion_seg(lesion_seg_1, labeled_lesion_seg_1)
    label_lesion_seg(lesion_seg_2, labeled_lesion_seg_2)
    label_lesion_seg(lesion_seg_2_reg, labeled_lesion_seg_2_reg)

    # Compute lesion Center-of-Mass (CoM) on the image 1, image 2, and the registered image 2
    lesion_1_CoM = compute_lesion_CoM(labeled_lesion_seg_1)
    lesion_2_CoM = compute_lesion_CoM(labeled_lesion_seg_2)
    lesion_2_reg_CoM = compute_lesion_CoM(labeled_lesion_seg_2_reg)

    # We perform lesion matching based on the CoM using the Hungarian algorithm
    lesion_mapping_1_to_reg2 = compute_lesion_mapping(lesion_1_CoM, lesion_2_reg_CoM)
    logger.info(f"Lesion mapping from timepoint 1 to timepoint 2 based on CoM:\n{lesion_mapping_1_to_reg2}")

    # Then we register back to image 2 space the corrected lesion segmentation
    # Initialize file name for corrected lesion segmentation of image 2 registered back to image 2
    reg_back_labeled_lesion_seg_2 = os.path.join(temp_folder, image_2_name.replace('.nii.gz', '_lesion-seg-registered-back-labeled.nii.gz'))
    assert os.system(f"sct_apply_transfo -i {labeled_lesion_seg_2_reg} -d {input_image2} -w {inv_warping_field_img2_to_1} -o {reg_back_labeled_lesion_seg_2} -x linear") == 0, "Failed to warp back corrected lesion segmentation to image 2 space"
    
    # Now we compute the CoM of the registered back lesions
    lesion_2_reg_back_CoM = compute_lesion_CoM(reg_back_labeled_lesion_seg_2)
    # We perform lesion matching of reg back lesion 2 and lesion 2
    lesion_mapping_regback2_to_2 = compute_lesion_mapping(lesion_2_reg_back_CoM, lesion_2_CoM)
    logger.info(f"Lesion mapping from timepoint 2 registered back to timepoint 2 to timepoint 2 based on CoM:\n{lesion_mapping_regback2_to_2}")

    # Then we build the full mapping from timepoint 1 to timepoint 2
    full_mapping_1_to_2 = {}
    for lesion1_id, lesion2_reg_id in lesion_mapping_1_to_reg2.items():
        if lesion2_reg_id in lesion_mapping_regback2_to_2:
            lesion2_id = lesion_mapping_regback2_to_2[lesion2_reg_id]
            full_mapping_1_to_2[lesion1_id] = [lesion2_id]
    logger.info(f"Full lesion mapping from timepoint 1 to timepoint 2 based on CoM:\n{full_mapping_1_to_2}")

    # # Remove temporary folder
    # assert os.system(f'rm -rf {temp_folder}') == 0, "Failed to remove temporary folder"

    return full_mapping_1_to_2


if __name__ == "__main__":
    args = parse_args()
    map_lesions_registered_with_CoM(args.input_image1, args.input_image2, args.output_folder)