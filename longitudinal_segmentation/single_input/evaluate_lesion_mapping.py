"""
This code evaluates the lesion mapping results between two timepoints.
It computes segmentation metrics of the predicted segmentations against the ground truth.
It also computes lesion matching metrics based on the lesion mapping results and the ground truth mapping.

Input:
    -i : path to the input MSD dataset
    -p : path to the folder where predictions were stored
    -o : path to the output folder where evaluation results will be stored

Output:
    None
"""
from datetime import date
import os
import argparse
import json
from pathlib import Path
import nibabel as nib
import numpy as np
import sys
from loguru import logger
from tqdm import tqdm
# From the utils file in the evaluation folder
# Import the functions from utils in parent folder
file_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.abspath(os.path.join(file_path, "../.."))
sys.path.insert(0, root_path)
from evaluation.utils import dice_score, lesion_ppv, lesion_f1_score, lesion_sensitivity, lesion_wise_tp_fp_fn
from longitudinal_segmentation.single_input.map_lesions_registered_with_IoU import compute_lesion_mapping, compute_IoU_matrix


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-msd', type=str, required=True, help='Path to the input MSD dataset')
    parser.add_argument('-p', '--predictions-folder', type=str, required=True, help='Path to the folder where predictions were stored')
    parser.add_argument('-o', '--output-folder', type=str, required=True, help='Path to the output folder where evaluation results will be stored')
    return parser.parse_args()


def compute_segmentation_metrics(pred_seg_path, gt_seg_path):
    """
    Compute segmentation metrics between predicted and ground truth segmentations.

    Args:
        pred_seg_path (str): Path to the predicted segmentation NIfTI file.
        gt_seg_path (str): Path to the ground truth segmentation NIfTI file.
    Returns:
        dice (float): Dice similarity coefficient.
        ppv (float): Positive predictive value.
        f1 (float): Lesion F1 score.
        sensitivity (float): Lesion sensitivity.
        TP (int): True Positives.
        FP (int): False Positives.
        FN (int): False Negatives.
        count_lesion_GT (int): Number of lesions in ground truth.
        count_lesion_pred (int): Number of lesions in prediction.
    """
    # Load the segmentations
    pred_data = nib.load(pred_seg_path).get_fdata()
    gt_data = nib.load(gt_seg_path).get_fdata()

    # Binarize the segmentations
    pred_binary = (pred_data > 0).astype(int)
    gt_binary = (gt_data > 0).astype(int)

    # Compute metrics
    dice = dice_score(pred_binary, gt_binary)
    ppv = lesion_ppv(pred_binary, gt_binary)
    f1 = lesion_f1_score(pred_binary, gt_binary)
    sensitivity = lesion_sensitivity(pred_binary, gt_binary)
    TP, FP, FN = lesion_wise_tp_fp_fn(gt_binary, pred_binary)
    count_lesion_GT = np.unique(gt_data).size - 1  # Exclude background
    count_lesion_pred = np.unique(pred_data).size - 1  # Exclude background

    return dice, ppv, f1, sensitivity, TP, FP, FN, count_lesion_GT, count_lesion_pred


def convert_mapping(pred_mapping_file, convert_gt1_to_pred1, convert_gt2_to_pred2):
    """
    Convert the predicted lesion mapping using the two conversion mappings.

    Args:
        pred_mapping_file (str): Path to the predicted lesion mapping JSON file.
        convert_gt1_to_pred1 (dict): Conversion mapping from GT timepoint 1 to predicted timepoint 1.
        convert_gt2_to_pred2 (dict): Conversion mapping from GT timepoint 2 to predicted timepoint 2.
    Returns:
        converted_mapping (dict): Converted lesion mapping.
    """
    with open(pred_mapping_file, 'r') as f:
        pred_mapping = json.load(f)

    # We initialize the converted mapping
    converted_mapping = {}
    # For each lesion in GT timepoint 1
    for gt_lesion_id, pred_lesion_id_time1 in convert_gt1_to_pred1.items():
        converted_mapping[int(gt_lesion_id)] = []
        for pred_lesion_id in pred_lesion_id_time1:
            if str(pred_lesion_id) not in pred_mapping.keys():
                continue
            values_in_pred2 = pred_mapping[str(pred_lesion_id)]
            for val in values_in_pred2:
                values_in_gt2 = convert_gt2_to_pred2[int(val)]
                converted_mapping[int(gt_lesion_id)].extend(values_in_gt2)

    return converted_mapping


def evaluate_lesion_mappings(pred_mapping, gt_mapping):
    """
    Evaluate the predicted lesion mapping against the ground truth mapping.

    Args:
        pred_mapping (dict): Predicted lesion mapping.
        gt_mapping (dict): Ground truth lesion mapping.
    Returns:
        TP (list): list of True Positives for each lesion.
        FP (list): list of False Positives for each lesion.
        FN (list): list of False Negatives for each lesion.
    """
    TP = []
    FP = []
    FN = []

    # For each lesion in GT mapping, check if it is correctly mapped in predicted mapping
    for gt_lesion_id, gt_mapped_lesions in gt_mapping.items():
        tp, fp, fn = 0, 0, 0
        if int(gt_lesion_id) in pred_mapping:
            pred_mapped_lesions = pred_mapping[int(gt_lesion_id)]
            # For each gt_mapped_lesion we check if is mapped in the prediction
            for lesion in gt_mapped_lesions:
                if int(lesion) in pred_mapped_lesions:
                    tp += 1
                else:
                    fn += 1
            # For each predicted mapped lesion we check it they are potentially false positives
            for lesion in pred_mapped_lesions:
                if int(lesion) not in gt_mapped_lesions:
                    fp += 1
        else:
            # All lesions in gt_mapped_lesions are false negatives
            fn += len(gt_mapped_lesions)
        TP.append(tp)
        FP.append(fp)
        FN.append(fn)
    return TP, FP, FN


def lesion_volume(segmentation_path):
    """
    Compute the total lesion volume in mm^3 for a given segmentation.

    Args:
        segmentation_path (str): Path to the segmentation NIfTI file.
    Returns:
        volume_mm3 (float): Total lesion volume in mm^3.
    """
    seg_img = nib.load(segmentation_path)
    seg_data = seg_img.get_fdata()
    voxel_volume = np.prod(seg_img.header.get_zooms())  # in mm^3
    # Binarize the segmentation
    seg_data = (seg_data > 0).astype(np.uint8)
    lesion_voxels = np.sum(seg_data)
    volume_mm3 = lesion_voxels * voxel_volume
    return volume_mm3


def evaluate_lesion_mapping(input_msd_dataset, predictions_folder, output_folder):
    """
    This file evaluates the lesion mapping results between two timepoints.
    It computes segmentation metrics of the predicted segmentations against the ground truth.
    It also computes lesion matching metrics based on the lesion mapping results and the ground truth mapping.

    Args:
        input_msd_dataset (str): Path to the input MSD dataset.
        predictions_folder (str): Path to the folder where predictions were stored.
        output_folder (str): Path to the output folder where evaluation results will be stored.

    Returns:
        results_dict (dict): Dictionary containing evaluation results for each subject.
    """
    # Initialize logger
    logger.add(os.path.join(output_folder, f'logger_{str(date.today())}.log'))
    logger.info(f"Input MSD dataset: {input_msd_dataset}")
    logger.info(f"Predictions folder: {predictions_folder}")
    logger.info(f"Output folder: {output_folder}")

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Load the msd dataset
    with open(input_msd_dataset, 'r') as f:
        msd_data = json.load(f)
    data = msd_data['data']

    # Initilialize results dictionary
    results_dict = {}

    # We run evaluation now on all the data
    for subject in tqdm(data):
        subject_id = subject
        # Build output folder for the subject
        subject_output_folder = os.path.join(output_folder, subject_id)
        os.makedirs(subject_output_folder, exist_ok=True)
        # Initialize the timepoints and images
        timepoint1 = "ses-M0"
        timepoint2 = "ses-M12"
        input_image1 = data[subject][timepoint1][0]
        input_image2 = data[subject][timepoint2][0]
        # We build the path to the labeled predicted segmentations, lesion mapping file
        predicted_lesion_seg_1 = os.path.join(predictions_folder, subject_id, Path(input_image1).name.replace('.nii.gz', '_lesion-seg-labeled.nii.gz'))
        predicted_lesion_seg_2 = os.path.join(predictions_folder, subject_id, Path(input_image2).name.replace('.nii.gz', '_lesion-seg-labeled.nii.gz'))
        pred_lesion_mapping_file = os.path.join(predictions_folder, subject_id, 'lesion_mapping.json')
        # Build path to the GT segmentations and GT lesion mapping file
        gt_lesion_seg_1 = input_image1.replace('canproco', 'canproco/derivatives/labels-ms-spinal-cord-only').replace('.nii.gz', '_lesion-manual-labeled.nii.gz')
        gt_lesion_seg_2 = input_image2.replace('canproco', 'canproco/derivatives/labels-ms-spinal-cord-only').replace('.nii.gz', '_lesion-manual-labeled.nii.gz')
        gt_lesion_mapping_file = str(Path(gt_lesion_seg_1).parent).replace('ses-M0/anat', 'lesion-mapping_M0-M12.json')
        # Copy all files in the temporary output folder
        temp_folder = os.path.join(subject_output_folder, 'temp')
        os.makedirs(temp_folder, exist_ok=True)
        os.system(f"cp {predicted_lesion_seg_1} {temp_folder}/")
        os.system(f"cp {predicted_lesion_seg_2} {temp_folder}/")
        os.system(f"cp {gt_lesion_seg_1} {temp_folder}/")
        os.system(f"cp {gt_lesion_seg_2} {temp_folder}/")
        os.system(f"cp {pred_lesion_mapping_file} {temp_folder}/")
        os.system(f"cp {gt_lesion_mapping_file} {temp_folder}/")

        # We compute the dice scores and other segmentation metrics for both timepoints
        dice1, ppv1, f1_1, sensitivity1, TP1, FP1, FN1, count_lesion_GT1, count_lesion_pred1  = compute_segmentation_metrics(predicted_lesion_seg_1, gt_lesion_seg_1)
        dice2, ppv2, f1_2, sensitivity2, TP2, FP2, FN2, count_lesion_GT2, count_lesion_pred2 = compute_segmentation_metrics(predicted_lesion_seg_2, gt_lesion_seg_2)
        logger.info(f"Timepoint 1 - Dice: {dice1:.4f}, PPV: {ppv1:.4f}, F1: {f1_1:.4f}, Sensitivity: {sensitivity1:.4f}, TP: {TP1}, FP: {FP1}, FN: {FN1}, Count GT Lesions: {count_lesion_GT1}, Count Pred Lesions: {count_lesion_pred1}")
        logger.info(f"Timepoint 2 - Dice: {dice2:.4f}, PPV: {ppv2:.4f}, F1: {f1_2:.4f}, Sensitivity: {sensitivity2:.4f}, TP: {TP2}, FP: {FP2}, FN: {FN2}, Count GT Lesions: {count_lesion_GT2}, Count Pred Lesions: {count_lesion_pred2}")
        
        # Now we compute the lesion mapping from pred 1 to GT 1
        convert_gt1_to_pred1 = compute_lesion_mapping(compute_IoU_matrix(nib.load(gt_lesion_seg_1).get_fdata(), nib.load(predicted_lesion_seg_1).get_fdata()), 1e-4)
        logger.info(f"Lesion mapping from GT timepoint 1 to predicted timepoint 1 :\n{convert_gt1_to_pred1}")
        # Now we compute the lesion mapping from pred 2 to GT 2
        convert_pred2_to_gt2 = compute_lesion_mapping(compute_IoU_matrix(nib.load(predicted_lesion_seg_2).get_fdata(), nib.load(gt_lesion_seg_2).get_fdata()), 1e-4)
        logger.info(f"Lesion mapping from predicted timepoint 2 to GT timepoint 2:\n{convert_pred2_to_gt2}")

        # Now we convert the predicted lesion mapping using the two conversion mappings
        converted_predicted_lesion_mapping = convert_mapping(pred_lesion_mapping_file, convert_gt1_to_pred1, convert_pred2_to_gt2)
        logger.info(f"Converted predicted lesion mapping:\n{converted_predicted_lesion_mapping}")

        # Now we evaluate the converted predicted lesion mapping against the GT lesion mapping
        ## Load the GT lesion mapping
        with open(gt_lesion_mapping_file, 'r') as f:
            gt_lesion_mapping = json.load(f)
        TP, FP, FN = evaluate_lesion_mappings(converted_predicted_lesion_mapping, gt_lesion_mapping)
        logger.info(f"Lesion mapping evaluation - TP: {TP}, FP: {FP}, FN: {FN}")

        # Compute total lesion volume in mm^3 for both timepoints and both predicted and GT segmentations
        vol_gt1 = lesion_volume(gt_lesion_seg_1)
        vol_pred1 = lesion_volume(predicted_lesion_seg_1)
        vol_gt2 = lesion_volume(gt_lesion_seg_2)
        vol_pred2 = lesion_volume(predicted_lesion_seg_2)

        # Add all the results to a dictionnary
        results = {
            'dice1': dice1,
            'ppv1': ppv1,
            'f1_1': f1_1,
            'sensitivity1': sensitivity1,
            'TP1': TP1,
            'FP1': FP1,
            'FN1': FN1,
            'count_lesion_GT1': count_lesion_GT1,
            'count_lesion_pred1': count_lesion_pred1,
            'dice2': dice2,
            'ppv2': ppv2,
            'f1_2': f1_2,
            'sensitivity2': sensitivity2,
            'TP2': TP2,
            'FP2': FP2,
            'FN2': FN2,
            'count_lesion_GT2': count_lesion_GT2,
            'count_lesion_pred2': count_lesion_pred2,
            'lesion_mapping_TP': TP,
            'lesion_mapping_FP': FP,
            'lesion_mapping_FN': FN,
            'vol_gt1_mm3': vol_gt1,
            'vol_pred1_mm3': vol_pred1,
            'vol_gt2_mm3': vol_gt2,
            'vol_pred2_mm3': vol_pred2
        }
        
        # Save the results to a JSON file
        results_file = os.path.join(subject_output_folder, 'results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)

        # We add the results to the overall results dictionary
        results_dict[subject_id] = results

    # Save the overall results to a JSON file
    overall_results_file = os.path.join(output_folder, 'overall_results.json')
    with open(overall_results_file, 'w') as f:
        json.dump(results_dict, f, indent=4)

    # Remove the temp folder
    # os.system(f"rm -rf {temp_folder}")

    return results_dict


if __name__ == "__main__":
    # Parse arguments
    args = parse_args()
    input_msd_dataset = args.input_msd
    predictions_folder = args.predictions_folder
    output_folder = args.output_folder
    # Run evaluation
    results = evaluate_lesion_mapping(input_msd_dataset, predictions_folder, output_folder)