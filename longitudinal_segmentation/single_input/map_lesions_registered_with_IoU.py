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


def compute_IoU_matrix(data_lesion_1, data_lesion_2):
    """
    This function computes the IoU matrix between lesions in two lesion segmentation images.

    Inputs:
        data_lesion_1 : numpy array of the first lesion segmentation image
        data_lesion_2 : numpy array of the second lesion segmentation image
    Outputs:
        IoU_matrix : numpy array of shape (n_lesions_1, n_lesions_2) containing the IoU values between each pair of lesions
    """
    n_lesions_1 = len(np.unique(data_lesion_1)) - 1  # exclude background
    n_lesions_2 = len(np.unique(data_lesion_2)) - 1  # exclude background
    IoU_matrix = np.zeros((n_lesions_1, n_lesions_2))
    for i in range(1, n_lesions_1 + 1):
        lesion_1_mask = (data_lesion_1 == i)
        for j in range(1, n_lesions_2 + 1):
            lesion_2_mask = (data_lesion_2 == j)
            intersection = np.logical_and(lesion_1_mask, lesion_2_mask).sum()
            union = np.logical_or(lesion_1_mask, lesion_2_mask).sum()
            if union > 0:
                IoU = intersection / union
            else:
                IoU = 0
            IoU_matrix[i-1, j-1] = IoU
    return IoU_matrix


def compute_lesion_mapping(IoU_matrix, IoU_threshold):
    """
    This function computes the lesion mapping between two timepoints based on the IoU matrix and a given threshold.

    Inputs:
        IoU_matrix : numpy array of shape (n_lesions_1, n_lesions_2) containing the IoU values between each pair of lesions
        IoU_threshold : float, threshold for IoU to consider a match

    Outputs:
        lesion_mapping_forward : dict, mapping from lesion indices from baseline to follow-up
    """
    n_lesions_1, n_lesions_2 = IoU_matrix.shape
    lesion_mapping_forward = {}
    for i in range(n_lesions_1):
        lesion_mapping_forward[i+1] = []
        for j in range(n_lesions_2):
            if IoU_matrix[i, j] >= IoU_threshold:
                lesion_mapping_forward[i+1].append(j+1)
    return lesion_mapping_forward


def map_lesions_registered_with_IoU(input_image1, input_image2, output_folder, IoU_threshold=1e-5, lesion_seg_1_input=None, lesion_seg_2_input=None):
    """
    This function performs lesion mapping between two timepoints using registered images and lesion matching based on the center of mass of lesions.

    Inputs:
        input_image1 : path to the input image at timepoint 1
        input_image2 : path to the input image at timepoint 2
        output_folder : path to the output folder where comparison results will be stored
        lesion_seg_1_input : path to the lesion segmentation at timepoint 1 (optional)
        lesion_seg_2_input : path to the lesion segmentation at timepoint 2 (optional)

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
    if lesion_seg_1_input is None:
        lesion_seg_1 = os.path.join(temp_folder, image_1_name.replace('.nii.gz', '_lesion-seg.nii.gz'))
    sc_seg_1 = os.path.join(temp_folder, image_1_name.replace('.nii.gz', '_sc-seg.nii.gz'))
    levels_1 = os.path.join(temp_folder, image_1_name.replace('.nii.gz', '_levels.nii.gz'))
    image_2_name = Path(input_image2).name
    if lesion_seg_2_input is None:
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
    if lesion_seg_1_input is None:
        segment_lesions(input_image1, sc_seg_1, qc_folder, lesion_seg_1, test_time_aug=True, soft_ms_lesion=True)
    else:
        lesion_seg_1 = lesion_seg_1_input
    if lesion_seg_2_input is None:
        segment_lesions(input_image2, sc_seg_2, qc_folder, lesion_seg_2, test_time_aug=True, soft_ms_lesion=True)
    else:
        lesion_seg_2 = lesion_seg_2_input
    # Get the levels
    get_levels(input_image1, levels_1)
    get_levels(input_image2, levels_2)

    # In the context of this registration, we need to only have common levels between both timepoints
    keep_common_levels_only(levels_1, levels_2)

    # Register image 2 to image 1
    parameter = "step=0,type=label,dof=Tx_Ty_Tz:step=1,type=im,algo=dl"
    assert os.system(f"sct_register_multimodal -i {input_image2} -d {input_image1} -param '{parameter}' -ilabel {levels_2} -dlabel {levels_1} -o {registered_image2_to_1} -owarp {warping_field_img2_to_1} -owarpinv {inv_warping_field_img2_to_1} -dseg {sc_seg_1} -qc {qc_folder} ") == 0, "Registration failed"

    # # We warp the lesion segmentation of image 2 to image 1 space using a linear interpolation
    assert os.system(f"sct_apply_transfo -i {lesion_seg_2} -d {input_image1} -w {warping_field_img2_to_1} -o {lesion_seg_2_reg} -x linear") == 0, "Failed to warp lesion segmentation of image 2"
    
    # We label the lesion segmentation files
    label_lesion_seg(lesion_seg_1, labeled_lesion_seg_1)
    label_lesion_seg(lesion_seg_2, labeled_lesion_seg_2)
    label_lesion_seg(lesion_seg_2_reg, labeled_lesion_seg_2_reg)

    # Now we perform lesion matching betweemn timepoint 1 and timepoint 2 registered to timepoint 1 based on IoU
    data_lesion_1 = nib.load(labeled_lesion_seg_1).get_fdata()
    data_lesion_2_reg = nib.load(labeled_lesion_seg_2_reg).get_fdata()
    # For each lesion at timepoint 1, compute IoU with each lesion at timepoint 2
    IoU_matrix = compute_IoU_matrix(data_lesion_1, data_lesion_2_reg)
    logger.info(f"IoU matrix between lesions at timepoint 1 and timepoint 2:\n{IoU_matrix}")
    # We map lesions in both directions based on the IoU threshold
    lesion_mapping_1_to_2 = compute_lesion_mapping(IoU_matrix, IoU_threshold)    
    # print the lesion mapping
    logger.info(f"Lesion mapping based on IoU threshold of {IoU_threshold}:\n{lesion_mapping_1_to_2}")

    # We reg back the labeled lesion segmentation of timepoint 2 to timepoint 2 space
    lesion_seg_2_reg_back_labeled = os.path.join(temp_folder, image_2_name.replace('.nii.gz', '_lesion-seg-registered-back-labeled.nii.gz'))
    assert os.system(f"sct_apply_transfo -i {labeled_lesion_seg_2_reg} -d {input_image2} -w {inv_warping_field_img2_to_1} -o {lesion_seg_2_reg_back_labeled} -x linear") == 0, "Failed to warp back lesion segmentation of image 2"

    # Then we match the labeled lesion segmentation of timepoint 2 reg back to timepoint 2 space to have consistent lesion labels
    data_lesion_reg_back = nib.load(lesion_seg_2_reg_back_labeled).get_fdata()
    data_lesion_2 = nib.load(labeled_lesion_seg_2).get_fdata()
    # Compute IoU between lesions in both images
    IoU_matrix_reg_back = compute_IoU_matrix(data_lesion_reg_back, data_lesion_2)
    logger.info(f"IoU matrix between lesions at timepoint 2 reg back and timepoint 2:\n{IoU_matrix_reg_back}")
    # We map lesions in both directions based on the IoU threshold
    lesion_mapping_reg_back_to_2 = compute_lesion_mapping(IoU_matrix_reg_back, IoU_threshold)
    # print the lesion mapping
    logger.info(f"Lesion mapping from reg back to timepoint 2 based on IoU threshold of {IoU_threshold}:\n{lesion_mapping_reg_back_to_2}")

    # Now we compute the full mapping from timepoint 1 to timepoint 2
    full_mapping_1_to_2 = {}
    for lesion_1, mapped_lesions_2 in lesion_mapping_1_to_2.items():
        full_mapped_lesions_2 = []
        for lesion_2 in mapped_lesions_2:
            if lesion_2 in  lesion_mapping_reg_back_to_2:
                full_mapped_lesions_2.extend(lesion_mapping_reg_back_to_2[lesion_2])
        full_mapping_1_to_2[lesion_1] = full_mapped_lesions_2

    logger.info(f"Full lesion mapping from timepoint 1 to timepoint 2 based on IoU threshold of {IoU_threshold}:\n{full_mapping_1_to_2}")

    # Remove the temporary folder
    # assert os.system(f'rm -rf {temp_folder}') == 0, "Failed to remove temporary folder"

    return full_mapping_1_to_2


if __name__ == "__main__":
    args = parse_args()
    lesion_mapping_1_to_2 = map_lesions_registered_with_IoU(args.input_image1, args.input_image2, args.output_folder)