"""
This code compares lesion segmentations from two timepoints for a single subject using registration to match lesions.

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
from scipy.interpolate import interp1d
from scipy.optimize import linear_sum_assignment
import json
import sys
# Import the functions from utils in parent folder
file_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.abspath(os.path.join(file_path, ".."))
sys.path.insert(0, root_path)
from utils import segment_sc, segment_lesions, get_levels


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i1', '--input_image1', type=str, required=True, help='Path to the input image at timepoint 1')
    parser.add_argument('-i2', '--input_image2', type=str, required=True, help='Path to the input image at timepoint 2')
    parser.add_argument('-o', '--output_folder', type=str, required=True, help='Path to the output folder where comparison results will be stored')
    return parser.parse_args()


def keep_common_levels_only(levels_1, levels_2):
    """
    This function keeps only the common disc levels between two level segmentations.
    I could have done this using 'sct_label_utils -remove-reference'

    Inputs:
        levels_1 : path to the levels segmentation at timepoint 1
        levels_2 : path to the levels segmentation at timepoint 2

    Outputs:
        None
    """
    # Load levels
    img_levels_1 = nib.load(levels_1)
    data_levels_1 = img_levels_1.get_fdata()
    img_levels_2 = nib.load(levels_2)
    data_levels_2 = img_levels_2.get_fdata()
    # Find common levels
    common_levels = np.intersect1d(np.unique(data_levels_1), np.unique(data_levels_2))
    common_levels = common_levels[common_levels != 0]  # Exclude background
    # Keep only common levels
    new_data_levels_1 = np.zeros(data_levels_1.shape)
    new_data_levels_2 = np.zeros(data_levels_2.shape)
    for level in common_levels:
        new_data_levels_1[data_levels_1 == level] = level
        new_data_levels_2[data_levels_2 == level] = level
    # Save new levels
    new_img_levels_1 = nib.Nifti1Image(new_data_levels_1, img_levels_1.affine, img_levels_1.header)
    nib.save(new_img_levels_1, levels_1)
    new_img_levels_2 = nib.Nifti1Image(new_data_levels_2, img_levels_2.affine, img_levels_2.header)
    nib.save(new_img_levels_2, levels_2)

    return None


def compute_lesion_CoM(lesion_seg):
    """
    This function computes the Center-of-Mass (CoM) of lesions in the lesion segmentation.

    Inputs:
        lesion_seg : path to the lesion segmentation
    
    Outputs:
        lesion_CoM : list of CoM coordinates for each lesion
    """
    # Load lesion segmentation
    img_lesion = nib.load(lesion_seg)
    data_lesion = img_lesion.get_fdata()
    # Label connected components (lesions)
    labeled_lesions, num_lesions = ndimage.label(data_lesion)
    lesion_CoM = {}
    for lesion_id in range(1, num_lesions + 1):
        lesion_mask = (labeled_lesions == lesion_id)
        if np.sum(lesion_mask) == 0:
            continue
        com = ndimage.center_of_mass(lesion_mask)
        lesion_CoM[lesion_id] = com

    return lesion_CoM


def map_lesions_registered_with_CoM(input_image1, input_image2, output_folder):
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

    # We warp the lesion segmentation of image 2 to image 1 space using a linear interpolation
    assert os.system(f"sct_apply_transfo -i {lesion_seg_2} -d {input_image1} -w {warping_field_img2_to_1} -o {lesion_seg_2_reg} -x linear") == 0, "Failed to warp lesion segmentation of image 2"
    
    # WE SHOULD FIRST CREATE LABELED LESION SEGS FOR ALL 3 SEGMENTATIONS (IMAGE 1, IMAGE 2, IMAGE 2 REGISTERED)

    # Compute lesion Center-of-Mass (CoM) on the image 1, image 2, and the registered image 2
    lesion_1_CoM = compute_lesion_CoM(lesion_seg_1)
    lesion_2_CoM = compute_lesion_CoM(lesion_seg_2)
    lesion_2_reg_CoM = compute_lesion_CoM(lesion_seg_2_reg)

    # We perform lesion matching based on the CoM using the Hungarian algorithm
    cost_matrix = np.zeros((len(lesion_1_CoM), len(lesion_2_reg_CoM)))
    for i, (lesion1_id, com1) in enumerate(lesion_1_CoM.items()):
        for j, (lesion2_id, com2) in enumerate(lesion_2_reg_CoM.items()):
            cost_matrix[i, j] = np.linalg.norm(np.array(com1) - np.array(com2))
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    print("Lesion mapping (lesion ID at timepoint 1 -> lesion ID at timepoint 2):")
    for i, j in zip(row_ind, col_ind):
        lesion1_id = list(lesion_1_CoM.keys())[i]
        lesion2_id = list(lesion_2_reg_CoM.keys())[j]
        distance = cost_matrix[i, j]
        print(f"Lesion {lesion1_id} -> Lesion {lesion2_id} (Distance: {distance:.2f} voxels)")
    
    # Save the

    



if __name__ == "__main__":
    args = parse_args()
    map_lesions_registered_with_CoM(args.input_image1, args.input_image2, args.output_folder)