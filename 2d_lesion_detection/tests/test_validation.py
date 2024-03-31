"""
Unit tests for functions used in validation.py and yolo_inference.py
"""

import torch
from torch import tensor

from yolo_inference import merge_overlapping_boxes, intersection_over_smallest_area
from validation import confusion_matrix, convert_bbox_format_to_corners, get_png_from_txt

def test_merge_overlapping_boxes():
    """
    Define two boxes and merge them with two different threshold values.
    Make sure merged boxes are as expected.
    """
    # Define two overlapping boxes (x1,y1,x2,y2)
    box1= tensor([2,2,6,6], dtype=torch.int32)
    box2= tensor([3,3,6,7], dtype=torch.int32)

    boxes = torch.stack((box1,box2))

    # Get merged boxes with two different IoUs
    merged_25 = merge_overlapping_boxes(boxes, 0.25)
    merged_75 = merge_overlapping_boxes(boxes, 0.80)

    # Boxes should be merged with IoU 25
    # but not with IoU 50
    assert torch.equal(merged_25[0], tensor([2,2,6,7], dtype=torch.int32))
    assert torch.equal(merged_75[0], boxes[0])
    assert torch.equal(merged_75[1], boxes[1])


def test_intersection_over_smallest_area():
    """
    Define 3 boxes and make sure intersection over smallest area values are 
    as expected.
    """
    # Define two overlapping boxes (x1,y1,x2,y2)
    box1= tensor([2,2,6,6], dtype=torch.int32)
    box2= tensor([3,3,6,7], dtype=torch.int32)
    box3= tensor([6,2,8,8], dtype=torch.int32)

    iosa12 = intersection_over_smallest_area(box1, box2)
    iosa13 = intersection_over_smallest_area(box1, box3)

    assert iosa12 == 0.75
    assert iosa13 == 0


def test_confusion_matrix():
    """
    Define two lists of bounding boxes: ground truth (labels) and predictions (preds)
    Calculate tp, fn, fp values with different thresholds and make sure values are as expected.
    """
    labels= tensor([[1,1,5,6], [1,7,3,9], [6,4,8,7]], dtype=torch.int32)
    preds= tensor([[1,2,6,7], [7,1,9,5], [7,8,9,9], [2,8,4,9]], dtype=torch.int32)

    # With 0.5 iou threshold
    tp, fn, fp = confusion_matrix(labels, preds, 0.5)
    assert tp == 1
    assert fn == 2
    assert fp == 3

    # With 0.1 iou threshold
    tp, fn, fp = confusion_matrix(labels, preds, 0.1)
    assert tp == 2
    assert fn == 1
    assert fp == 2

    # With 0.6 iou threshold
    tp, fn, fp = confusion_matrix(labels, preds, 0.6)
    assert tp == 0
    assert fn == 3
    assert fp == 4


def test_convert_bbox_format_to_corners():
    """
    Define image size and corresponding coordinates in center format
    Convert to corner format and make sure result is as expected

    Repeat for smaller boxes
    """
    img_width = 10
    img_height = 20
    center_coords = torch.tensor([[0,               # class
                                   4/img_width,     # x_center
                                   7/img_height,    # y_center
                                   4/img_width,     # width
                                   6/img_height]])  # height
    
    corners = convert_bbox_format_to_corners(center_coords, img_width, img_height)

    assert torch.equal(corners, tensor([[2,4,6,10]]))

    img_height = 5
    img_width = 7
    center_coords = torch.tensor([[0,
                                   1.5/img_width,
                                   2/img_height,
                                   1/img_width,
                                   2/img_height]])
    
    corners = convert_bbox_format_to_corners(center_coords, img_width, img_height)

    assert torch.equal(corners, tensor([[1,1,2,3]]))


def test_get_png_from_txt():
    """
    Define a labels folder path 
    Get corresponding images folder

    Repeat with a label txt file
    The corresponding images path should be a png file
    """
    # If input is a folder
    label_folder = "~/data/yolo_training/dataset_1/labels/test"

    assert get_png_from_txt(label_folder) == "~/data/yolo_training/dataset_1/images/test"

    # If input is a file
    txt_file = "~/data/yolo_training/dataset_1/labels/test/sub-cal080_ses-M0_STIR_2.txt"

    assert get_png_from_txt(txt_file) == "~/data/yolo_training/dataset_1/images/test/sub-cal080_ses-M0_STIR_2.png"

