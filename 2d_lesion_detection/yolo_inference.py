"""
Functions to predict slice-wise MS lesion positions using a trained YOLOv8 model.

Also performs post-processing by taking all slice predictions for a volume and 
merging boxes that overlap # TODO this feature should be moved to validation.py

Dataset should be formatted in YOLO format as defined in pre-processing.py
"""
import math
import os
from pathlib import Path
from typing import List, Dict
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import torch
import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results


def merge_overlapping_boxes(boxes:torch.Tensor, iosa_threshold:float)-> List[torch.Tensor]:
    """
    Takes a list of bounding boxes and groups together the ones that overlap (more than given threshold)
    
    I chose to use intersection over smallest box area (which is the proportion of the smallest box contained
    in the bigger box) as a threshold instead of IoU because this way, if a tiny box is fully within a big box, the tiny
    one will for sure be merged (iosa will be 1).

    Args:
        boxes (torch.Tensor): tensor containing the bounding boxes
            format --> torch.tensor([[x1, y1, x2, y2],[x1, y1, x2, y2], ...])
        iosa_threshold (float): Intersection over smallest area threshold
            Boxes with a higher iosa than this value will be merged together

    Returns:
        merged_boxes (List[torch.Tensor]): List of merged bounding boxes 
    """

    # List that will contain the final merged boxes
    merged_boxes = []

    for box in boxes:
        i = 0
        while i < len(merged_boxes):
            merged_box = merged_boxes[i]

            iosa = intersection_over_smallest_area(box, merged_box)
            if iosa > iosa_threshold:
                # Expand the merged box with box
                box = expand_bbox(merged_box, box)

                del merged_boxes[i] # this box will be replaced by the newly merged box
                # Don't increment i since merged_boxes is staying the same length (one box is replaced)
            else:
                i += 1 

        merged_boxes.append(box.round().int())

    return merged_boxes


def intersection_over_smallest_area(boxA:torch.Tensor, boxB:torch.Tensor)-> float:
    """"
    Given two bounding boxes, calculates the intersection area over the smallest box's area
    
    Adapted from: https://gist.github.com/meyerjo/dd3533edc97c81258898f60d8978eddc

    Args:
        boxA (torch.Tensor): First bounding box
        boxB (torch.Tensor): Second bounding box
            format --> torch.tensor([x1, y1, x2, y2])

    Returns:
        Intersection over small area (float)
    """
     
    # determine the (x, y)-coordinates of the intersection rectangle
    x1 = max(boxA[0], boxB[0])
    y1 = max(boxA[1], boxB[1])
    x2 = min(boxA[2], boxB[2])
    y2 = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((x2 - x1, 0)) * max((y2 - y1), 0))

    if interArea == 0:
        return 0
    
    # compute the area of both the prediction and ground-truth rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    smallest_area = min(boxAArea, boxBArea)

    return interArea/smallest_area


def expand_bbox(box1:torch.Tensor, box2:torch.Tensor)-> torch.Tensor:
    """
    Returns a single bounding box that contains both input boxes

    Args:
        box1 (torch.Tensor): First bounding box
        box2 (torch.Tensor): Second bounding box
            format --> torch.tensor([x1, y1, x2, y2])

    Returns:
        expanded box (torch.Tensor): Bounding box containing both initial boxes
            format --> torch.tensor([x1, y1, x2, y2])
    """
    # Expand box1 to include box2
    b1_x1, b1_y1, b1_x2, b1_y2 = box1
    b2_x1, b2_y1, b2_x2, b2_y2 = box2
    x1 = min(b1_x1, b2_x1)
    x2 = max(b1_x2, b2_x2)
    y1 = min(b1_y1, b2_y1)
    y2 = max(b1_y2, b2_y2)

    return torch.Tensor([x1, y1, x2, y2]).int()


def _get_slice_results_dict(results:List[Results])-> Dict[str, torch.Tensor]:
    """
    Get a dictionnary of YOLO results with slice name as key

    Args:
        results (List[Results]): list of results from YOLO predict mode

    Returns:
        result_dict (Dict[str, torch.Tensor]): dictionary containing predictions for every slice
    """
    # Sort results into a dictionnary with slice names as keys
    result_dict = {}
    for result in results:
        slice = Path(result.path).name.replace(".png", "")
        result_dict[slice] = result.boxes.xyxy

    return result_dict


def _get_volume_results_dict(results:List[Results])-> Dict[str, torch.Tensor]:
    """
    Get a dictionnary of YOLO results with volume name as key

    Args:
        results (List[Results]): list of results from YOLO predict mode

    Returns:
        result_dict (Dict[str, torch.Tensor]): dictionary containing predictions for every volume

    """
    # Sort results into a dictionnary with volume names as keys
    result_dict = {}
    for result in results:
        parts = Path(result.path).name.split('_')
        volume = '_'.join(parts[:-1]) # volume name

        if volume in result_dict:
            result_dict[volume] = torch.cat((result_dict[volume], result.boxes.xyxy), dim=0)

        else:
            result_dict[volume] = result.boxes.xyxy

    return result_dict


def _main():
    parser = ArgumentParser(
    prog = 'yolo_inference',
    description = 'Detect MS lesions with a YOLOv8 model',
    formatter_class = ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--model',
                        required = True,
                        type = Path,
                        help = 'Path to trained YOLO model .pt file')
    parser.add_argument('-d', '--dataset-path',
                        required= True,
                        type = Path,
                        help = 'Path to dataset folder of png images. Check pre-process.py for format.') 
    parser.add_argument('-o', '--output-folder',
                        required= True,
                        type = Path,
                        help = 'Path to directory where results will be saved')
    parser.add_argument('-b', '--batch-size',
                        default= 64,
                        type = int,
                        help = 'Batch size to use for inference.')
    parser.add_argument('-c', '--conf-threshold',
                        default= 0.2,
                        type = float,
                        help = 'Confidence threshold to keep model predictions.')
    parser.add_argument('-s', '--slice',
                        default= False,
                        action='store_true',
                        help = 'If used, one set of bounding boxes will be saved for every slice. '
                               'By default, one set of bounding boxes is saved for every volume.')

    args = parser.parse_args()

    # Create output folder if it doesn't exist
    os.makedirs(args.output_folder, exist_ok=True)

    # Get list of image paths from dataset path
    img_names = os.listdir(args.dataset_path)
    img_paths = [os.path.join(args.dataset_path, file) for file in img_names if file.endswith(".png")] #only keep pngs

    # Load model
    model = YOLO(args.model)

    # Perform inference in batches
    # From https://github.com/ultralytics/ultralytics/issues/4835
    results=[]
    for i in range(0, len(img_paths), args.batch_size):
        print(f"\nPredicting batch {int(i/args.batch_size)+1}/{math.ceil(len(img_paths)/args.batch_size)}")
        preds = model.predict(img_paths[i:i+args.batch_size], conf=args.conf_threshold)
        for pred in preds:
            results.append(pred)

    # Put results in a dictionary
    # If volume is true, results will be sorted by volume
    if not args.slice:
        result_dict = _get_volume_results_dict(results)
    else:
        result_dict = _get_slice_results_dict(results)

    
    # Save results (and merge overlaping boxes if volume is true)
    print(f"\nSaving results to {str(args.output_folder)}")
    for name, boxes in result_dict.items():
        # name is either slice name (if volume is False)
        # or volume name (if volume is True)

        if boxes.numel() == 0:
            # If no boxes are predicted, skip volume/ slice (no txt file is saved)
            continue

        elif not args.slice:
            # Merge boxes
            boxes = merge_overlapping_boxes(boxes, 0.2)

        # Save to txt
        with open(args.output_folder/(name+".txt"), "w") as file:
            # Iterate over the tensors in the list
            for tensor in boxes:
                np.savetxt(file, [tensor.numpy()], fmt='%d')



if __name__ == "__main__":
    _main()