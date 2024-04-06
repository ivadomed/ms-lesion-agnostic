"""
Unit tests for functions used in data_utils.py 
"""
import os
import logging
import tempfile
from pathlib import Path

import numpy as np
import nibabel as nib

from data_utils import nifti_to_png, mask_to_bbox, convert_bboxes_format

def test_nifti_to_png():
    """
    Makes sure that only the correct slices are saved, but does not check the content of the saved pngs
    """
    # Create nifti image
    image_data = np.zeros((3, 3, 3))
    image_data[:,:,0] = image_data[:,:,0] * 500
    image_data[:,:,1] = image_data[:,:,1] * 1000

    # Create spinal cord seg mask
    # Adding white pixel to slices 0 and 2
    sc_data = np.zeros((3, 3, 3))
    sc_data[1,1,0] = 1
    sc_data[1,1,2] = 1

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save nifti
        image_path = Path(tmpdir)/'image.nii.gz'
        nifti_image = nib.Nifti1Image(image_data, affine=np.eye(4))
        nib.save(nifti_image, image_path)

        sc_seg_path = Path(tmpdir)/'sc_seg.nii.gz'
        nifti_sc_seg = nib.Nifti1Image(sc_data, affine=np.eye(4))
        nib.save(nifti_sc_seg, sc_seg_path)

        # With sc_seg -- saved slices should be 0 and 2
        nifti_to_png(image_path, Path(tmpdir)/'with_sc_seg', sc_seg_path)
        txt_names = os.listdir(Path(tmpdir)/'with_sc_seg')

        assert len(txt_names) == 2
        assert "image_0.png" in txt_names
        assert "image_2.png" in txt_names
        assert "image_1.png" not in txt_names

        # With list of slices to save
        nifti_to_png(image_path, Path(tmpdir)/'with_slice_list', slice_list=[1,2])
        txt_names = os.listdir(Path(tmpdir)/'with_slice_list')

        assert len(txt_names) == 2
        assert "image_1.png" in txt_names
        assert "image_2.png" in txt_names
        assert "image_0.png" not in txt_names


def test_mask_to_bbox_warning(caplog):
    """
    Give a non binary mask to mask_to_bbox(), a warning should be logged
    """
    mask = np.zeros((3, 3))
    mask[1,:] = 3

    caplog.set_level(logging.WARNING)
    mask_to_bbox(mask)

    # Make sure a warning was logged
    assert len(caplog.records) == 1

    # Check content of the warning
    warning = caplog.records[0]
    assert "A binary mask is expected" in warning.message


def test_mask_to_bbox():
    """
    Give a mask to mask_to_bbox() and check that bbox is correct
    """
    mask = np.array([[0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0],
                     [0, 1, 1, 0, 0, 0],
                     [0, 1, 1, 0, 1, 0],
                     [0, 1, 1, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0]])

    expected = np.array([[1.5, 3, 1, 2],[4, 3, 0, 0]])/6
    assert np.allclose(mask_to_bbox(mask), expected, atol=0.001)


def test_convert_bboxes_format():
    """
    Convert a bounding box in x1,y1,x2,y2 format to
    x_center,y_center,width,height normalized
    """
    xyxy = np.array([[1, 1, 4, 3]])
    xywh = np.array([[2.5, 2, 3, 2]])

    width = 10
    height = 5

    xywhn = np.array([[0, 0, 0, 0]], dtype=np.float64)
    xywhn[0][0] = xywh[0][0]/width
    xywhn[0][1] = xywh[0][1]/height
    xywhn[0][2] = xywh[0][2]/width
    xywhn[0][3] = xywh[0][3]/height

    assert np.array_equal(convert_bboxes_format(xyxy, width, height), xywhn)

