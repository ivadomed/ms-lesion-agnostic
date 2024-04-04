"""
Functions to predict slice-wise MS lesion positions using a trained YOLOv8 model.

Optionally performs post-processing by taking all slice predictions for a volume and 
merging boxes that overlap

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


def _get_slice_results_dict(results:List[Results])-> Dict[str, torch.Tensor]:
    """
    Get a dictionnary of YOLO results with slice name as key

    Args:
        results (List[Results]): list of results from YOLO predict mode

    Returns:
        result_dict (Dict[str, torch.Tensor]): dictionary containing predictions for every slice
            !! boxes are in x_center, y_center, width, height normalized format !!
    """
    # Sort results into a dictionnary with slice names as keys
    result_dict = {}
    for result in results:
        slice_name = Path(result.path).name.replace(".png", "")
        boxes = result.boxes.xywhn
        result_dict[slice_name] = boxes

    return result_dict


def _get_volume_results_dict(results:List[Results])-> Dict[str, torch.Tensor]:
    """
    Get a dictionnary of YOLO results with volume name as key

    Args:
        results (List[Results]): list of results from YOLO predict mode

    Returns:
        result_dict (Dict[str, torch.Tensor]): dictionary containing predictions for every volume
            !! boxes are in x1,y1,x2,y2 format !!

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
    if args.volume:
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

        # Save to txt
        with open(args.output_folder/(name+".txt"), "w") as file:
            # Iterate over the tensors in the list
            for tensor in boxes:
                line = ' '.join(['0'] + [str(val) for val in tensor.numpy()]) # add a zero to indicate the class (assuming there is only one class)
                file.write(line + '\n')



if __name__ == "__main__":
    _main()