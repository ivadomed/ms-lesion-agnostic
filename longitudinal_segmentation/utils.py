"""
This file contains utility functions for longitudinal segmentation with single input images.

Author: Pierre-Louis Benveniste
"""
import os
from loguru import logger
import nibabel as nib
from scipy import ndimage
import numpy as np


def segment_sc(input_image, output_sc_seg):
    """
    This function segments the spinal cord from the input image and saves the segmentation.

    Inputs:
        input_image : path to the input image
        output_sc_seg : path to save the spinal cord segmentation
    Outputs:
        None
    """
    # Placeholder implementation
    logger.info(f"Segmenting spinal cord in {input_image} and saving to {output_sc_seg}")

    # Use SCT with GPU:
    assert os.system(f"SCT_USE_GPU=1 sct_deepseg spinalcord -i {input_image} -o {output_sc_seg}") == 0, "Spinal cord segmentation failed"
    
    return None


def segment_lesions(input_image, input_sc_seg, qc_folder, output_lesion_seg, test_time_aug=False, soft_ms_lesion=False):
    """
    This function segments lesions from the input image and saves the segmentation.

    Inputs:
        input_image : path to the input image
        input_sc_seg : path to the spinal cord segmentation
        qc_folder : path to the quality control folder
        output_lesion_seg : path to save the lesion segmentation

    Outputs:
        None
    """
    # Placeholder implementation
    logger.info(f"Segmenting lesions in {input_image} and saving to {output_lesion_seg}")

    # Use SCT with GPU:
    assert os.system(f"SCT_USE_GPU=1 sct_deepseg lesion_ms -i {input_image} {'-test-time-aug' if test_time_aug else ''} {'-soft-ms-lesion' if soft_ms_lesion else ''} -o {output_lesion_seg} -qc {qc_folder} -qc-plane Sagittal -qc-seg {input_sc_seg}") == 0, "Lesion segmentation failed"
    
    return None


def get_centerline(input_sc_seg, output_centerline):
    """
    This function computes the centerline from the spinal cord segmentation.

    Inputs:
        input_sc_seg : path to the spinal cord segmentation
        output_centerline : path to save the centerline
    """
    # Placeholder implementation
    logger.info(f"Computing centerline from {input_sc_seg} and saving to {output_centerline}")

    # Use SCT with GPU:
    assert os.system(f"sct_get_centerline -i  {input_sc_seg} -method fitseg -o {output_centerline}") == 0, "Centerline computation failed"

    return None


def get_levels(input_image, output_levels):
    """
    This function computes the vertebral levels from the input image.

    Inputs:
        input_image : path to the input image
        output_levels : path to save the vertebral levels
    """
    # Placeholder implementation
    logger.info(f"Computing vertebral levels from {input_image} and saving to {output_levels}")

    # Build a temporary folder for the levels
    temp_folder = os.path.join(os.path.dirname(output_levels), "temp_levels")
    os.makedirs(temp_folder, exist_ok=True)
    temp_output = os.path.join(temp_folder, 'output.nii.gz')

    # Use SCT with GPU:
    assert os.system(f"SCT_USE_GPU=1 sct_deepseg totalspineseg -i {input_image} -step1-only 1 -o {temp_output} ") == 0, "Vertebral levels computation failed"

    # Then we only copy the output levels files to the desired output folder
    level_output_file = os.path.join(temp_output.replace('.nii.gz', '_step1_levels.nii.gz'))
    assert os.system(f'cp {level_output_file} {output_levels}') == 0, "Failed to copy vertebral levels file"

    # Clean up temporary folder
    os.system(f'rm -rf {temp_folder}')

    return None


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


def compute_lesion_CoM(labeled_lesion_seg):
    """
    This function computes the Center-of-Mass (CoM) of lesions in the lesion segmentation.

    Inputs:
        labeled_lesion_seg : path to the labeled lesion segmentation
    
    Outputs:
        lesion_CoM : list of CoM coordinates for each lesion
    """
    # Load lesion segmentation
    img_lesion = nib.load(labeled_lesion_seg)
    data_lesion = img_lesion.get_fdata()
    # Initialize dictionary to store CoM
    lesion_CoM = {}
    # Get unique lesion labels
    lesions = np.unique(data_lesion)
    lesions = lesions[lesions != 0]  # Exclude background
    for lesion_id in lesions:
        lesion_mask = (data_lesion == lesion_id)
        if np.sum(lesion_mask) == 0:
            continue
        com = ndimage.center_of_mass(lesion_mask)
        lesion_CoM[lesion_id] = com

    return lesion_CoM


def label_lesion_seg(lesion_seg, output_labeled_lesion_seg):
    """
    This function labels the lesion segmentation.

    Inputs:
        lesion_seg : path to the lesion segmentation
        output_labeled_lesion_seg : path to the output labeled lesion segmentation

    Outputs:
        None
    """
    # Load lesion segmentation
    img_lesion = nib.load(lesion_seg)
    data_lesion = img_lesion.get_fdata()
    # Connectivity for labeling (this is to have connectivity in 3D even through corners)
    s = ndimage.generate_binary_structure(3,3)
    # Label connected components (lesions)
    labeled_lesions, num_lesions = ndimage.label(data_lesion, structure=s)
    # Save labeled lesion segmentation
    labeled_img = nib.Nifti1Image(labeled_lesions, img_lesion.affine, img_lesion.header)
    nib.save(labeled_img, output_labeled_lesion_seg)

    return None


def correct_labeling(labeled_lesion_seg, lesion_id_mapping):
    """
    This function corrects the labeling of lesions in the registered lesion segmentation based on the matching.
    The lesion_id_mapping is a dictionary mapping old lesion IDs to new lesion IDs.

    Inputs:
        labeled_lesion_seg : path to the labeled lesion segmentation to be corrected
        lesion_id_mapping : dictionary mapping old lesion IDs to new lesion IDs
    Outputs:
        None
    """
    # Load lesion segmentation
    img_lesion = nib.load(labeled_lesion_seg)
    data_lesion = img_lesion.get_fdata()
    # Create a new array for corrected labels
    corrected_data = np.zeros(data_lesion.shape)
    for old_id, new_id in lesion_id_mapping.items():
        corrected_data[data_lesion == old_id] = int(new_id)
    # Save corrected lesion segmentation
    corrected_img = nib.Nifti1Image(corrected_data, img_lesion.affine, img_lesion.header)
    nib.save(corrected_img, labeled_lesion_seg)

    return None