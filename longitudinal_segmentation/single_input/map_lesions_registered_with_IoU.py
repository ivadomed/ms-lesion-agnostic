"""
This code compares lesion segmentations from two timepoints for a single subject using registration to match lesions.
The lesion matching is performed based on the IoU (Intersection over Union) between lesions.

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


def map_lesions_registered_with_IoU(input_image1, input_image2, output_folder, IoU_threshold=0.05):
    """
    This function performs lesion mapping between two timepoints using registered images and lesion matching based on the center of mass of lesions.

    Inputs:
        input_image1 : path to the input image at timepoint 1
        input_image2 : path to the input image at timepoint 2
        output_folder : path to the output folder where comparison results will be stored

    Outputs:
        None
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
    sc_seg_1 = os.path.join(temp_folder, image_1_name.replace('.nii.gz', '_sc_seg.nii.gz'))
    levels_1 = os.path.join(temp_folder, image_1_name.replace('.nii.gz', '_levels.nii.gz'))
    image_2_name = Path(input_image2).name
    lesion_seg_2 = os.path.join(temp_folder, image_2_name.replace('.nii.gz', '_lesion-seg.nii.gz'))
    sc_seg_2 = os.path.join(temp_folder, image_2_name.replace('.nii.gz', '_sc_seg.nii.gz'))
    levels_2 = os.path.join(temp_folder, image_2_name.replace('.nii.gz', '_levels.nii.gz'))
    # Initialize file name for registered image 2
    registered_image2_to_1 = os.path.join(temp_folder, image_2_name.replace('.nii.gz', '_registered_to_' + image_1_name))
    warping_field_img2_to_1 = os.path.join(temp_folder, image_2_name.replace('.nii.gz', '_warp_to_' + image_1_name))
    inv_warping_field_img2_to_1 = os.path.join(temp_folder, image_2_name.replace('.nii.gz', '_inv_warp_to_' + image_1_name))
    # Initialize file name for lesion segmentation of image 2 registered to image 1
    lesion_seg_2_reg = os.path.join(temp_folder, image_2_name.replace('.nii.gz', '_registered.nii.gz'))
    # Initialize file name for labeled lesion segmentations
    labeled_lesion_seg_1 = os.path.join(temp_folder, image_1_name.replace('.nii.gz', '_lesion-seg_labeled.nii.gz'))
    labeled_lesion_seg_2 = os.path.join(temp_folder, image_2_name.replace('.nii.gz', '_lesion-seg_labeled.nii.gz'))
    labeled_lesion_seg_2_reg = os.path.join(temp_folder, image_2_name.replace('.nii.gz', '_lesion-seg_registered_labeled.nii.gz'))

    # # Segment the spinal cord
    # segment_sc(input_image1, sc_seg_1)
    # segment_sc(input_image2, sc_seg_2)
    # # Segment the lesions
    # segment_lesions(input_image1, sc_seg_1, qc_folder, lesion_seg_1, test_time_aug=True)
    # segment_lesions(input_image2, sc_seg_2, qc_folder, lesion_seg_2, test_time_aug=True)
    # # Get the levels
    # get_levels(input_image1, levels_1)
    # get_levels(input_image2, levels_2)

    # In the context of this registration, we need to only have common levels between both timepoints
    keep_common_levels_only(levels_1, levels_2)

    # # Register image 2 to image 1
    # parameter = "step=0,type=label,dof=Tx_Ty_Tz:step=1,type=im,algo=dl"
    # assert os.system(f"sct_register_multimodal -i {input_image2} -d {input_image1} -param '{parameter}' -ilabel {levels_2} -dlabel {levels_1} -o {registered_image2_to_1} -owarp {warping_field_img2_to_1} -owarpinv {inv_warping_field_img2_to_1} -dseg {sc_seg_1} -qc {qc_folder} ") == 0, "Registration failed"

    # # We warp the lesion segmentation of image 2 to image 1 space using a linear interpolation
    # assert os.system(f"sct_apply_transfo -i {lesion_seg_2} -d {input_image1} -w {warping_field_img2_to_1} -o {lesion_seg_2_reg} -x nn") == 0, "Failed to warp lesion segmentation of image 2"
    
    # We label the lesion segmentation files
    label_lesion_seg(lesion_seg_1, labeled_lesion_seg_1)
    label_lesion_seg(lesion_seg_2, labeled_lesion_seg_2)
    label_lesion_seg(lesion_seg_2_reg, labeled_lesion_seg_2_reg)

    # Now we perform lesion matching betweemn timepoint 1 and timepoint 2 registered to timepoint 1 based on IoU
    ## If two lesions have IoU > threshold, they are considered the same lesion
    img_lesion_1 = nib.load(labeled_lesion_seg_1)
    data_lesion_1 = img_lesion_1.get_fdata()
    img_lesion_2_reg = nib.load(labeled_lesion_seg_2_reg)
    data_lesion_2_reg = img_lesion_2_reg.get_fdata()
    n_lesions_1 = int(data_lesion_1.max())
    n_lesions_2 = int(data_lesion_2_reg.max())
    logger.info(f"Number of lesions at timepoint 1: {n_lesions_1}")
    logger.info(f"Number of lesions at timepoint 2: {n_lesions_2}")
    # For each lesion at timepoint 1, compute IoU with each lesion at timepoint 2
    IoU_matrix = np.zeros((n_lesions_1, n_lesions_2))
    for i in range(1, n_lesions_1 + 1):
        lesion_1_mask = (data_lesion_1 == i)
        for j in range(1, n_lesions_2 + 1):
            lesion_2_mask = (data_lesion_2_reg == j)
            intersection = np.logical_and(lesion_1_mask, lesion_2_mask).sum()
            union = np.logical_or(lesion_1_mask, lesion_2_mask).sum()
            if union > 0:
                IoU = intersection / union
            else:
                IoU = 0
            IoU_matrix[i-1, j-1] = IoU
    # print the lesion IoU matrix
    logger.info(f"IoU matrix between lesions at timepoint 1 and timepoint 2:\n{IoU_matrix}")
    # Now we use the Hungarian algorithm to find the best matching between lesions based on IoU
    cost_matrix = 1 - IoU_matrix  # We want to maximize IoU, so we minimize 1 - IoU
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    # Now we filter the matches based on the IoU threshold
    lesion_id_mapping = {} # old lesion id at timepoint 2 to new lesion id at timepoint 1
    for i, j in zip(row_ind, col_ind):
        if IoU_matrix[i, j] >= IoU_threshold:
            lesion_id_mapping[int(j + 1)] = int(i + 1)  # lesion ids are 1-indexed
    logger.info(f"Lesion ID mapping from timepoint 2 to timepoint 1 based on IoU threshold of {IoU_threshold}: {lesion_id_mapping}")
    ## ALSO NEED TO LOOK AT HOW TO MAP LESIONS IN THE OTHER DIRECTION ALSO

    return None


if __name__ == "__main__":
    args = parse_args()
    map_lesions_registered_with_IoU(args.input_image1, args.input_image2, args.output_folder)