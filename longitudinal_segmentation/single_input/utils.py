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


def segment_lesions(input_image, input_sc_seg, qc_folder, output_lesion_seg):
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
    assert os.system(f"SCT_USE_GPU=1 sct_deepseg lesion_ms -i {input_image} -o {output_lesion_seg} -qc {qc_folder} -qc-plane Sagittal -qc-seg {input_sc_seg}") == 0, "Lesion segmentation failed"
    
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


def analyze_lesions(lesion_seg, sc_seg, centerline, levels, output_labeled_lesion_seg):
    """
    This function analyzes the lesions given the lesion segmentation, spinal cord segmentation, centerline, and vertebral levels.

    Inputs:
        lesion_seg : path to the lesion segmentation
        sc_seg : path to the spinal cord segmentation
        centerline : path to the centerline
        levels : path to the vertebral levels
    
    Outputs:
        analysis_results : results of the lesion analysis
    """
    # Placeholder implementation
    logger.info(f"Analyzing lesions in {lesion_seg}")

    # Initialize the analysis results
    analysis_results = {}

    # We load the segmentation and look at each lesion (connected component)
    lesion_data = nib.load(lesion_seg).get_fdata()
    # Label the connected components
    lbl_data, num_lesion = ndimage.label(lesion_data)
    # Compute the center of mass for each lesion
    labels = [i+1 for i in range(num_lesion)]
    h = ndimage.center_of_mass(lesion_data, lbl_data, labels)
    # Store the results
    analysis_results['num_lesions'] = num_lesion
    analysis_results['lesions'] = {}
    for label, CoM in zip(labels, h):
        analysis_results['lesions'][f'{label}'] = {}
        analysis_results['lesions'][f'{label}']['center_of_mass'] = CoM

    # For each lesion, we compute its volume
    voxel_volume = np.prod(nib.load(lesion_seg).header.get_zooms())
    for label in labels:
        lesion_data = (lbl_data == label).astype(np.uint8)
        lesion_volume = np.sum(lesion_data) * voxel_volume  # in mm^3
        analysis_results['lesions'][f'{label}']['volume_mm3'] = lesion_volume

    # We save a labeled lesion segmentation
    labeled_lesion_img = nib.Nifti1Image(lbl_data, nib.load(lesion_seg).affine)
    nib.save(labeled_lesion_img, output_labeled_lesion_seg)

    return analysis_results