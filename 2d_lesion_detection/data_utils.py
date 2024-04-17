"""
Functions for data preprocessing, to generate a YOLO format dataset from a BIDS database.
"""

import logging
import os
from pathlib import Path

import cv2
import torch
import numpy as np
import nibabel as nib
import scipy.ndimage as ndi
from torchvision.ops import masks_to_boxes
from skimage.exposure import equalize_adapthist


def nifti_to_png(nifti_path:Path, output_dir:Path, spinal_cord_path:Path=None, slice_list:list=None):
    """
    Converts a nifti volume into slices along the sagittal plane and
    saves them as png files in specified output_dir

    If spinal_cord_path is given, only slices that contain part of the spinal cord are saved and slice_list is ignored.
    If slice_list is given and spinal_cord_path is None, only slices in the given list are saved. slice_list should contain ints.

    Not suitable for segmentations (instead use nifti_seg_to_png) because 
    of intensity normalization between 0 and 255
    
    png images are named <nifti filename>_<slice number>
    For example, if nifti file sub-cal056_ses-M12_STIR.nii.gz is given,
    the first slice will be named sub-cal056_ses-M12_STIR_0.png

    Args:
        nifti_path (pathlib.Path) : path to nifti file
        output_dir (pathlib.Path) : path to the directory where png slices will be saved
        spinal_cord_path (pathlib.Path) : path to the spinal cord segmentation file (optional)

    adapted from https://neuraldatascience.io/8-mri/nifti.html#plot-a-series-of-slices
    """
    # Make output directory if it doesn't already exist
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = nifti_path.stem[:-len(".nii")]

    volume = nib.load(nifti_path)
    vol_data = volume.get_fdata() # np array

    if spinal_cord_path is None:
        sc_seg_data = None
    else:
        sc_seg = nib.load(spinal_cord_path)
        sc_seg_data = sc_seg.get_fdata()

    # Normalize pixel intensity from 0 to 255 
    vol_data = (vol_data - np.amin(vol_data)) * (255 / (np.amax(vol_data) - np.amin(vol_data)))
    vol_data = np.round(vol_data)

    n_slice = vol_data.shape[2]

    for i in range(n_slice):
        # if spinal cord segmentation is given, check if slice contains spinal cord
        if not sc_seg_data is None:
            sc_seg_slice = sc_seg_data[:, :, i]

            if sc_seg_slice.max() == 1:
                # slice contains spinal cord
                output_path = os.path.join(str(output_dir), f"{filename}_{i}.png")
                img_slice = np.clip(ndi.rotate(vol_data[:, :, i], 90) / 255.0, 0, 1) # make sure it's between 0 and 1 for histogram equalization
                
                cv2.imwrite(output_path, equalize_adapthist(img_slice)*255) # equalize histogram
            else:
                assert(sc_seg_slice.max() == 0)
        else:
            # if no segmentation is given
            # save slice if slice is in slice_list OR if slice_list is None
            if slice_list is None or i in slice_list:
                output_path = os.path.join(str(output_dir), f"{filename}_{i}.png")
                cv2.imwrite(output_path, ndi.rotate(vol_data[:, :, i], 90))


def mask_to_bbox(mask:np.ndarray) -> "np.ndarray|None":
    """
    Extracts bounding box coordinates for each object in a binary tensor

    Bounding box coordinates are in format (x_center, y_center, width, height)
    with normalized cooordinates (between 0 and 1)

    Args:
        mask (np.ndarray): binary mask to get bboxes from

    Returns:
        boxes_array (np.ndarray|None): array containing bounding box coordinates for each object
            if no object is detected, None is returned

    """
    # Check if we have a binary mask
    try:
        assert np.all(np.logical_or(mask == 0, mask == 1))
    except AssertionError as e:
        logging.warning(f"{e}: A binary mask is expected, but given mask is not.")

    width = mask.shape[1]
    height = mask.shape[0]

    # Separate each object
    labeled_array, num_labels = ndi.label(mask)

    if num_labels == 0: # No objects
        return None

    # List to store bounding boxes for each lesion
    boxes = []

    # Loop over each labeled region
    for label in range(1, num_labels + 1):
        # Create a boolean mask for the current object
        obj_mask = labeled_array == label

        # Compute the bounding box for the current object
        # Returns format (x1, y1, x2, y2) in pixels
        obj_box = masks_to_boxes(torch.from_numpy(obj_mask).unsqueeze(0))[0]

        # Add to list
        boxes.append(obj_box)

    # Convert list of tensors to an array
    boxes_array = np.array([box.numpy() for box in boxes])

    # Convert coordinates format to (x_center, y_center, width, height) and normalize
    boxes_array = convert_bboxes_format(boxes_array, width, height)

    return boxes_array


def convert_bboxes_format(bboxes:np.ndarray, img_width:int, img_height:int) -> np.ndarray:
    """
    Converts bounding box format from (x1, y1, x2, y2) in pixels to 
    (x_center, y_center, width, height) normalized between 0 and 1

    Args:
        bboxes (np.ndarray): Bounding box to convert
        img_width (int): Corresponding image width (px)
        img_height (int): Corresponding image height (px)

    Returns:
        (np.ndarray): converted coordinates  
    """    
    # Extract coordinates from the input array
    x_start, y_start, x_end, y_end = bboxes[:,0], bboxes[:,1], bboxes[:,2], bboxes[:,3]

    # Calculate center coordinates
    x_center = (x_start + x_end) / 2
    y_center = (y_start + y_end) / 2

    # Calculate width and height of boxes
    width = x_end - x_start
    height = y_end - y_start

    # Normalize
    x_center = x_center/img_width
    width = width/img_width
    y_center = y_center/img_height
    height = height/img_height

    # Stack the converted coordinates and sizes into a single array
    return np.stack((x_center, y_center, width, height), axis=-1)


def labels_from_nifti(nifti_labels_path:Path, output_dir:Path)->Path:
    """
    Creates txt files containing bounding box coordinates for each slice in a nifti segmentation
    If no bounding box is found for a given slice, no txt file is created

    txt filenames correspond to the nifti name:
    For example, if sub-cal056_ses-M12_STIR_lesion-manual.nii.gz is given as input, the txt file for the 
    first slice will be named sub-cal056_ses-M12_STIR_0.txt

    Args:
        nifti_labels_path (pathlib.Path): Path to the nifti file containing the segmentation
        output_dir (pathlib.Path): Path to the directory where txt files will be saved
    """
    # Make output directory if it doesn't already exist
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = nifti_labels_path.stem[:-len("_lesion_manual.nii")]

    # get nifti volume as numpy array
    volume = nib.load(nifti_labels_path)
    vol_array = volume.get_fdata()

    # For each slice, extract bounding boxes and save to txt file
    n_slice = vol_array.shape[2]
    for i in range(n_slice):
        slice_array = np.round(ndi.rotate(vol_array[:, :, i], 90))

        # Get bounding boxes
        boxes_array = mask_to_bbox(slice_array)

        if not boxes_array is None:
            # Add a column for class 0
            # Since we only have one type of object to detect (lesion)
            column_of_zeros = np.zeros((boxes_array.shape[0], 1))
            boxes_array = np.concatenate((column_of_zeros, boxes_array), axis=1)

            # Save to output directory
            output_path = os.path.join(str(output_dir), filename + f"_{i}.txt")
            np.savetxt(output_path, boxes_array, fmt = ['%g', '%.6f', '%.6f', '%.6f', '%.6f'])

        else:
            # If no object is found, no txt file is generated
            pass