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
from longitudinal_segmentation.single_input.map_lesions_unregistered import segment_sc, segment_lesions


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i1', '--input_image1', type=str, required=True, help='Path to the input image at timepoint 1')
    parser.add_argument('-i2', '--input_image2', type=str, required=True, help='Path to the input image at timepoint 2')
    parser.add_argument('-o', '--output_folder', type=str, required=True, help='Path to the output folder where comparison results will be stored')
    return parser.parse_args()


def map_lesions_registered(input_image1, input_image2, output_folder):
    """
    This function performs lesion mapping between two timepoints.

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

    # Segment the lesions, the sc, the centerline and the discs at both timepoints
    image_1_name = Path(input_image1).name
    lesion_seg_1 = os.path.join(temp_folder, image_1_name.replace('.nii.gz', '_lesion-seg.nii.gz'))
    sc_seg_1 = os.path.join(output_folder, image_1_name.replace('.nii.gz', '_sc_seg.nii.gz'))
    centerline_1 = os.path.join(temp_folder, image_1_name.replace('.nii.gz', '_centerline.nii.gz'))
    levels_1 = os.path.join(temp_folder, image_1_name.replace('.nii.gz', '_levels.nii.gz'))
    image_2_name = Path(input_image2).name
    lesion_seg_2 = os.path.join(temp_folder, image_2_name.replace('.nii.gz', '_lesion-seg.nii.gz'))
    sc_seg_2 = os.path.join(output_folder, image_2_name.replace('.nii.gz', '_sc_seg.nii.gz'))
    centerline_2 = os.path.join(temp_folder, image_2_name.replace('.nii.gz', '_centerline.nii.gz'))
    levels_2 = os.path.join(temp_folder, image_2_name.replace('.nii.gz', '_levels.nii.gz'))

    # Segment the spinal cord
    segment_sc(input_image1, sc_seg_1)
    segment_sc(input_image2, sc_seg_2)
    # Segment the lesions
    segment_lesions(input_image1, sc_seg_1, qc_folder, lesion_seg_1)
    segment_lesions(input_image2, sc_seg_2, qc_folder, lesion_seg_2)


if __name__ == "__main__":
    args = parse_args()
    map_lesions_registered(args.input_image1, args.input_image2, args.output_folder)
