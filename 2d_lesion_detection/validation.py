"""
Script for yolo model validation

Takes ground truth bounding box labels and predicted labels, and computes recall and precision.
Numbers of TPs, FPs and FNs for every image, as well as recall and precision for the whole batch are saved to a csv file.
Also saves both ground truth and predicted bounding boxes as nifti images (where the contour of the bboxes is 1)
"""

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from PIL import Image
import pandas as pd
import torch
import numpy as np
import nibabel as nib

IOSA = 0.2 # threshold for box merging, was determined by trying different values. With 0.2 boxes in a similar area
           # are merged together. Because of the slice thickness, this threshold can't be too high because lesions can 
           # shift quite a bit from one slice to the next.


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
    b1_s0, b1_sf, b1_x1, b1_y1, b1_x2, b1_y2 = box1
    b2_s0, b2_sf, b2_x1, b2_y1, b2_x2, b2_y2 = box2
    x1 = min(b1_x1, b2_x1)
    x2 = max(b1_x2, b2_x2)
    y1 = min(b1_y1, b2_y1)
    y2 = max(b1_y2, b2_y2)

    s0 = min(b1_s0, b2_s0)
    sf = max(b1_sf, b2_sf)

    return torch.Tensor([s0, sf, x1, y1, x2, y2]).int()


def intersection_over_smallest_area(boxA:torch.Tensor, boxB:torch.Tensor)-> float:
    """"
    Given two bounding boxes, calculates the intersection area over the smallest box's area
    
    Adapted from: https://gist.github.com/meyerjo/dd3533edc97c81258898f60d8978eddc

    Args:
        boxA (torch.Tensor): First bounding box
        boxB (torch.Tensor): Second bounding box
            format --> torch.tensor([s0, sf, x1, y1, x2, y2])

    Returns:
        Intersection over small area (float)
    """
     
    # determine the (x, y)-coordinates of the intersection rectangle
    x1 = max(boxA[2], boxB[2])
    y1 = max(boxA[3], boxB[3])
    x2 = min(boxA[4], boxB[4])
    y2 = min(boxA[5], boxB[5])

    # compute the area of intersection rectangle
    interArea = abs(max((x2 - x1, 0)) * max((y2 - y1), 0))

    if interArea == 0:
        return 0
    
    # compute the area of both the prediction and ground-truth rectangles
    boxAArea = abs((boxA[4] - boxA[2]) * (boxA[5] - boxA[3]))
    boxBArea = abs((boxB[4] - boxB[2]) * (boxB[5] - boxB[3]))

    smallest_area = min(boxAArea, boxBArea)

    return interArea/smallest_area


def boxes_overlap_or_consecutive(box1:torch.Tensor, box2:torch.Tensor)->bool:
    """
    Determines whether two boxes either overlap or are on consecutive slices.

    Args:
        box1 (torch.Tensor): First box
        box2 (torch.Tensor): Second box
            they should both be formatted as: torch.tensor([s0, sf, x1, y1, x2, y2])
    
    Returns:
        True or False
    """
    # Check if they overlap
    if box2[0] <= box1[1] and box2[0] >= box1[0]:
        return True
    
    elif box2[1] <= box1[1] and box2[1] >= box1[0]:
        return True
    
    # Check if they are consecutive
    elif box1[1] + 1 == box2[0] or box2[1] + 1 == box1[0]:
        return True
    
    else:
        return False

def merge_overlapping_boxes(boxes:torch.Tensor, iosa_threshold:float)-> List[torch.Tensor]:
    """
    Takes a tensor of bounding boxes and groups together the ones that overlap (more than given threshold)
    
    I chose to use intersection over smallest box area (which is the proportion of the smallest box contained
    in the bigger box) as a threshold instead of IoU because this way, if a tiny box is fully within a big box, the tiny
    one will for sure be merged (iosa will be 1).

    Args:
        boxes (torch.Tensor): tensor containing the bounding boxes
            format --> torch.tensor([[s0, sf, x1, y1, x2, y2],[s0, sf, x1, y1, x2, y2], ...])
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

            # Check if the slices are consecutive
            if boxes_overlap_or_consecutive(box, merged_box):
                iosa = intersection_over_smallest_area(box, merged_box)
                if iosa > iosa_threshold:
                    # Expand the merged box with box
                    box = expand_bbox(merged_box, box)

                    del merged_boxes[i] # this box will be replaced by the newly merged box
                    # Don't increment i since merged_boxes is staying the same length (one box is replaced)
                else:
                    i += 1
            else:
                i += 1
        merged_boxes.append(box.round().int())

    return merged_boxes

def xywhn_to_xyxy(bboxes: torch.Tensor, img_width: int, img_height: int) -> torch.Tensor:
    """
    Converts bounding box format from (x_center, y_center, width, height) normalized by image size 
    to (x1, y1, x2, y2) in pixels.

    Args:
        bboxes (torch.Tensor): Tensor of bounding boxes in (x_center, y_center, width, height) format
            format --> torch.tensor([[x_center, y_center, width, height],[x_center, y_center, width, height], ...])
        img_width (int): Width of the corresponding image
        img_height (int): Height of the corresponding image

    Returns:
        (torch.Tensor): Converted bounding box coordinates
            format --> torch.tensor([[x1, y1, x2, y2],[x1, y1, x2, y2], ...])
    """
    # Extract coordinates and sizes from the input tensor
    x_center, y_center, width, height = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]

    # Denormalize coordinates and sizes
    x_center = x_center * img_width
    width = width * img_width
    y_center = y_center * img_height
    height = height * img_height

    # Calculate (x1, y1, x2, y2) coordinates
    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2

    # Stack the converted coordinates into a single tensor
    return torch.stack((x1, y1, x2, y2), dim=-1)


def get_png_from_txt(txt_path:str)-> str:
    """
    Within a YOLO databsase, finds png image (or folder) path associated 
    with a given label file (or folder) path

    Args:
        txt_path (str): Path to txt file or labels folder

    Returns:
        images (str): Path to associated png file or images folder
    """
    images = txt_path.replace("labels", "images")

    # Check if path is a txt file
    if images.endswith(".txt"):
        return images.replace(".txt", ".png")
    
    # If not, assume it is a folder
    else:
        return images


def image_from_bboxes(bboxes: List[torch.Tensor], nii_data: np.ndarray)-> np.ndarray:
    """
    Creates a 3d image from bounding box coordinates. Sides of boxes will be 1 and background 0.
    Image will have the same dimensions as nii_data.

    Args:
        bboxes (List[torch.Tensor]): List of bounding boxes
            format --> [torch.tensor([[s0, sf, x1, y1, x2, y2]]), ...]
        nii_data (np.ndarray): Original nifti image

    Returns:
        nii_data (np.ndarray): Image of the bounding boxes
    """

    # Create an empty image volume
    nii_data.fill(0)

    # Set bounding box edges to 1
    for bbox in bboxes:
        (s0, sf, x1, y1, x2, y2) = bbox.tolist()

        y1 = nii_data.shape[1] - y1 -1
        y2 = nii_data.shape[1] - y2 -1

        if y1 >= nii_data.shape[1]-1:
            y1 = nii_data.shape[1] - 2
        if x2 >= nii_data.shape[0]-1:
            x2 = nii_data.shape[0] - 2

        nii_data[x1-1:x2+2, y1+1, s0:sf+1] = 1
        nii_data[x1-1:x2+2, y2-1, s0:sf+1] = 1
        nii_data[x1-1, y2-1:y1+1, s0:sf+1] = 1
        nii_data[x2+1, y2-1:y1+1, s0:sf+1] = 1

    return nii_data


def confusion_matrix(ground_truth:Optional[List[torch.Tensor]], 
                     predictions:Optional[List[torch.Tensor]], 
                     iou_threshold:float)-> Tuple[int, int, int]:
    """
    Computes True positive, False negative and False positive values from a list of
    ground truth bounding boxes and a list of predictions.
    
    A box from the ground truth list is considered a match with a prediction box if their iou is 
    above iou_threshold.

    Boxes should be formatted like this: s0, sf, x1, y1, x2, y2 (where s0 and sf are the first and last slices)
    
    Args:
        ground_truth (List[torch.Tensor]): list of ground truth bounding boxes
        predictions (List[torch.Tensor]): list of prediction bounding boxes
        iou_threshold (float): Intersection over union threshold 
            Boxes with an iou above or equal to this value are considered a match

    Returns:
        tp (int): number of true positives 
            gt boxes with an iou above or equal to iou_threshold with a prediction box
        fn (int): number of false negatives
            ground truth boxes that don't match with a prediction box
        fp (int): number of false positives
            prediction boxes that don't match with a ground truth box
    """
    # Start by checking if either of the lists is None
    if ground_truth is None:
        if predictions is None:
            return 0, 0, 0  # Both ground_truth and predictions are None, return 0s for tp, fn, fp
        else:
            return 0, 0, len(predictions)  # Ground_truth is None, so all predictions are false positives (fp)
    elif predictions is None:
        return 0, len(ground_truth), 0  # Predictions are None, so all ground truth are false negatives (fn)
    

    # Create matrix of ious between all ground truths (rows) and predictions (columns)
    ious = []
    for gt_box in ground_truth:
        iou = []
        for pred_box in predictions:
            # Calculate intersection
            xmin = max(gt_box[2], pred_box[2])
            ymin = max(gt_box[3], pred_box[3])
            zmin = max(gt_box[0], pred_box[0])

            xmax = min(gt_box[4], pred_box[4])
            ymax = min(gt_box[5], pred_box[5])
            zmax = min(gt_box[1], pred_box[1])

            # adding 1 to xmax, ymaz, zmax because if the box has a width of 1 px for example, xmax-xmin will be 0 (same for union)
            intersection = max(0, xmax+1 - xmin) * max(0, ymax+1 - ymin) * max(0, zmax+1 - zmin) 
            
            # Calculate union
            gt_volume = (gt_box[4]+1 - gt_box[2]) * (gt_box[5]+1 - gt_box[3]) * (gt_box[1]+1 - gt_box[0])
            pred_volume = (pred_box[4]+1 - pred_box[2]) * (pred_box[5]+1 - pred_box[3]) * (pred_box[1]+1 - pred_box[0])
            union = gt_volume + pred_volume - intersection
            
            # Calculate IoU
            iou.append(intersection / union if union > 0 else 0)
        ious.append(iou)
    

    # Count the number of tp, fn, fp
    tp = 0
    fn = 0
    fp = 0
    for _, gt_iou in enumerate(ious):
        if max(gt_iou) < iou_threshold:
            # For a given ground truth, if the max iou is below threshold -> FN
            fn+=1

    for _, pred_iou in enumerate(zip(*ious)):
        # If the max iou for a given prediction is over threshold -> TP
        if max(pred_iou) >= iou_threshold:
            tp+=1
        # If the max is below threshold -> FP
        else:
            fp+=1

    return tp, fn, fp


def get_volume_boxes(txt_paths:List[str], yolo_img_folder:Path, iosa:float)-> Dict[str, List[torch.Tensor]]:
    """
    From a list of txt file paths containing slice-wise bounding box coordinates, sorts 
    bbox coordinates by volume into a dictionary and merges overlapping boxes

    Converts format from x_center, y_center, width, height (normalized)
    to x1, y1, x2, y2 (in pixels)

    Example of expected filenames -> sub-cal080_ses-M0_STIR_2
        where 2 is the slice number

    Args:
        txt_paths (List(str)): List of txt file paths that contain the bounding box coordinates
        yolo_img_folder (Path): Path to the yolo dataset folder containing the images that correspond to txt_paths
        iosa (float): Intersection over smallest area threshold for two bboxes to be merged
    
    Returns:
        labels_dict (Dict[str, List[torch.Tensor]]): dictionary containing bounding boxes for every volume
            key is volume name (sub-cal080_ses-M0_STIR for example)
            value is a Tensor containg bounding box coordinates 
                format -> torch.tensor([s0, sf, x1, y1, x2, y2], [s0, sf, x1, y1, x2, y2], ...)
    """
    # Start by making a dictionary that groups slices of each volume together
    labels_dict_unmerged = {}
    for txt_path in txt_paths:
        parts = Path(txt_path).name.split('_')
        slice_no = parts[-1].replace(".txt","")
        volume = '_'.join(parts[:-1])

        # Get bbox coordinates as tensor
        data = []
        with open(txt_path, 'r') as file:
            for line in file:
                line = line.strip().split()
                line = [float(x) for x in line[1:]] # take line[1:] to ignore the class number
                data.append(line)
            boxes_tensor = torch.tensor(data)

            image_path = yolo_img_folder/f"{volume}_{slice_no}.png" # corresponding image in yolo dataset
            img = Image.open(image_path) # Img dimensions needed for bbox format conversion
            boxes_tensor = xywhn_to_xyxy(boxes_tensor, img.width, img.height).round().int()

            # Add slice number at the beginning of each row
            slice_indices = torch.tensor([int(slice_no), int(slice_no)]).repeat(boxes_tensor.shape[0], 1)
            boxes_tensor = torch.cat((slice_indices, boxes_tensor), dim=1)

        # Add to dict
        if volume in labels_dict_unmerged:
            labels_dict_unmerged[volume] = torch.cat((labels_dict_unmerged[volume], boxes_tensor), dim=0)

        else:
            labels_dict_unmerged[volume] = boxes_tensor


    # Merge overlapping boxes within a volume
    labels_dict={}
    for volume, boxes in labels_dict_unmerged.items():
        labels_dict[volume]= merge_overlapping_boxes(boxes, iosa)

    return labels_dict


def compute_metrics(volumes_list:List[str], 
                    labels_dict:Dict[str, List[torch.Tensor]],
                    preds_dict:Dict[str, List[torch.Tensor]],
                    canproco_path:Path,
                    output_folder:Path,
                    iou_threshold:float)-> pd.DataFrame:
    """
    Compute TP, FP and FN values between predictions and labels.
    Save nifti volumes of ground truth and prediction boxes to output folder

    Args:
        volumes_list (List[str]): list of volume names to process
        labels_dict (Dict[str, List[torch.Tensor]]): dictionary containing ground truth boxes per volume
        preds_dict (Dict[str, List[torch.Tensor]]): dictionary containing prediction boxes per volume
        canproco_path (Path): path to canproco database 
        output_folder (Path): path to output folder where nifti volumes of gt and prediction boxes are saved
        iou_threshold (float): Intersection over union threshold for a label and prediction box to be considered a match

    Returns:
        all_metrics_df (pd.DataFrame): Dataframe containing tp, fp, fn values for every volume 
    """
    df_columns = ['Volume', 'TP', 'FP', 'FN']
    all_metrics_df = pd.DataFrame(columns=df_columns)
    for volume in volumes_list:

        ## 1- Process ground truth
        if volume in labels_dict:
            # Get original nifti from canproco
            parts = volume.split('_')
            nii_path = canproco_path/ parts[0]/ parts[1]/ "anat"/ (volume+".nii.gz")
            nii_data = nib.load(str(nii_path))

            # Save boxes as nifti
            label_bboxes = labels_dict[volume]

            label_boxes_image = image_from_bboxes(label_bboxes, nii_data.get_fdata())
            boxes_nii = nib.Nifti1Image(label_boxes_image, nii_data.affine) # keep all metadata from original image
            boxes_nii.header.set_data_shape(nii_data.shape)

            nib.save(boxes_nii, str(output_folder/ (volume +"_label.nii.gz")))

        else:
            # If no ground truth boxes exist for this volume
            label_bboxes = None


        ## 2- Process prediction
        if volume in preds_dict:
            # Get original nifti from canproco
            parts = volume.split('_')
            nii_path = canproco_path/ parts[0]/ parts[1]/ "anat"/ (volume+".nii.gz")
            nii_data = nib.load(str(nii_path))

            # Save boxes as nifti
            pred_bboxes = preds_dict[volume]

            pred_boxes_image = image_from_bboxes(pred_bboxes, nii_data.get_fdata())
            boxes_nii = nib.Nifti1Image(pred_boxes_image, nii_data.affine) # keep all metadata from original image
            boxes_nii.header.set_data_shape(nii_data.shape)

            nib.save(boxes_nii, str(output_folder/ (volume +"_pred.nii.gz")))

        else:
            # If no prediction boxes exist for this volume
            pred_bboxes = None


        ## 3- Get metrics for given volume and add to all_metrics_df
        tp, fn, fp = confusion_matrix(label_bboxes, pred_bboxes, iou_threshold)
        all_metrics_df= pd.concat([all_metrics_df, pd.DataFrame([[volume, tp, fp, fn]], columns=df_columns)], ignore_index=True)

    return all_metrics_df


def _main():
    parser = ArgumentParser(
    prog = 'Validation',
    description = 'Validate a yolo model',
    formatter_class = ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--gt-path',
                        required= True,
                        type = str,
                        help = 'Path to YOLO dataset folder of ground truth txt files')
    parser.add_argument('-p', '--preds-path',
                        required= True,
                        type = str,
                        help = 'Path to prediction folder of txt files')
    parser.add_argument('-o', '--output-folder',
                        required= True,
                        type = Path,
                        help = 'Path to directory where results will be saved')
    parser.add_argument('-c', '--canproco',
                        required= True,
                        type = Path,
                        help = 'Path to canproco database')
    parser.add_argument('-i', '--iou',
                        default= 0.2,
                        type = float,
                        help = 'IoU threshold for a TP')

    args = parser.parse_args()

    # Create output folder if it doesn't exist
    os.makedirs(args.output_folder, exist_ok=True)

    ## 1-Get ground truth
    print("Loading ground truth...")
    # Get list of txt file paths from dataset path
    txt_names = os.listdir(args.gt_path)
    txt_paths = [os.path.join(args.gt_path, file) for file in txt_names if file.endswith(".txt")] # only keep txts

    # Get dictionary with volume names as keys 
    # And ground truth bounding box tensors as values
    labels_dict = get_volume_boxes(txt_paths, Path(get_png_from_txt(args.gt_path)), IOSA)


    ## 2-Get predictions
    print("Loading predictions...")
    # Get list of txt file paths from dataset path
    txt_names = os.listdir(args.preds_path)
    txt_paths = [os.path.join(args.preds_path, file) for file in txt_names if file.endswith(".txt")] #only keep txts

    # Get dictionary with volume names as keys 
    # And prediction bounding box tensors as values
    preds_dict = get_volume_boxes(txt_paths, Path(get_png_from_txt(args.gt_path)), IOSA)

        
    ## Save images with labels and predictions
    print("Saving images...")
    # Since not all images have a txt file (some don't contain lesions), get list of 
    # all images from images folder instead of labels folder
    img_names = os.listdir(get_png_from_txt(args.gt_path))
    img_paths = [os.path.join(get_png_from_txt(args.gt_path), file) for file in img_names if file.endswith(".png")] #only keep pngs

    # Get a list of all volumes from image paths
    volumes=[]
    for img_path in img_paths:
        parts = Path(img_path).name.split('_')
        volume = '_'.join(parts[:-1])
        if not volume in volumes:
            volumes.append(volume)

    # Compute metrics and save nifti images of label and pred boxes
    all_metrics_df = compute_metrics(volumes, labels_dict, preds_dict, args.canproco, args.output_folder, args.iou)

    all_tp = all_metrics_df['TP'].sum()
    all_fp = all_metrics_df['FP'].sum()
    all_fn = all_metrics_df['FN'].sum()

    precision = all_tp/(all_tp+all_fp)
    recall = all_tp/(all_tp+all_fn)

    # Add a row to dataframe with recall and precision
    new_row = pd.DataFrame({'Volume': ['Total'],
                            'TP': all_tp,
                            'FP': all_fp,
                            'FN': all_fn,
                            'Recall': recall,
                            'Precision': precision})

    all_metrics_df= pd.concat([all_metrics_df, new_row], ignore_index=True)

    # Save dataframe to csv file
    all_metrics_df.to_csv(str(args.output_folder/"metrics_report.csv"), index=False)

    # Print final metrics
    print('\nRecall: ', recall)
    print('Precision: ', precision)


if __name__ == "__main__":
    _main()