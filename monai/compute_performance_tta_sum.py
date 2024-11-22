"""
This script is used to sum all the image predictions of the same subject, then threshold to 0.5 and then compute the dice score.

Input:
    --path-pred: Path to the directory containing the predictions
    --path-json: Path to the json file containing the data split
    --split: Data split to use (train, validation, test)
    --output-dir: Output directory to save the dice scores

Output:
    None

Example:
    python compute_performance_tta_sum.py --path-pred /path/to/predictions --path-json /path/to/data.json --split test --output-dir /path/to/output

Author: Pierre-Louis Benveniste
"""

import os
import numpy as np
import argparse
from pathlib import Path
import json
import nibabel as nib
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-pred", type=str, required=True, help="Path to the directory containing the predictions")
    parser.add_argument("--path-json", type=str, required=True, help="Path to the json file containing the data split")
    parser.add_argument("--split", type=str, required=True, help="Data split to use (train, validation, test)")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory to save the dice scores")
    return parser.parse_args()


def dice_score(prediction, groundtruth, smooth=1.):
    numer = (prediction * groundtruth).sum()
    denor = (prediction + groundtruth).sum()
    # loss = (2 * numer + self.smooth) / (denor + self.smooth)
    dice = (2 * numer + smooth) / (denor + smooth)
    return dice


def main():

    # Parse arguments
    args = parse_args()
    path_pred = args.path_pred
    path_json = args.path_json
    split = args.split
    output_dir = args.output_dir

    # Create the output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get all the predictions (with rglob)
    predictions = list(Path(path_pred).rglob("*.nii.gz"))

    # List of subjects
    subjects = [pred.name for pred in predictions]

    n_tta = 10

    for i in range(n_tta):
        # Remove the _pred_0, _pred_1 ... _pred_9 at the end of the name
        subjects = [sub.replace(f"_pred_{i}", "") for sub in subjects]

    # Open the conversion dictionary (its a json file)
    with open(path_json, "r") as f:
        conversion_dict = json.load(f)
    conversion_dict = conversion_dict[split]

    # Dict of dice score
    dice_scores = {}

    # Iterate over the subjects in the predictions
    for subject in subjects:
        print(f"Processing subject {subject}")

        # Get all predictions corresponding to the subject
        subject_predictions = [str(pred) for pred in predictions if subject.replace(".nii.gz", "") in pred.name]
        # print(subject_predictions)

        # Find the corresponding label from the conversion dict
        
        image_dict = [data for data in conversion_dict if subject in data["image"]]
        label = image_dict[0]["label"]
        image = image_dict[0]["image"]

        # We now sum all the predictions
        summed_prediction = None
        for pred in subject_predictions:
            pred_data = nib.load(pred).get_fdata()
            if summed_prediction is None:
                summed_prediction = pred_data
            else:
                summed_prediction += pred_data
        
        # Threshold the summed prediction
        summed_prediction[summed_prediction >= 0.5] = 1
        summed_prediction[summed_prediction < 0.5] = 0

        # Load the label
        label_data = nib.load(label).get_fdata()

        # Compute dice score
        dice = dice_score(summed_prediction, label_data)
        # print(f"Dice score for summed prediction: {dice}")

        # Compare the dice score with the individual predictions
        for pred in subject_predictions:
            pred_data = nib.load(pred).get_fdata()
            dice_pred = dice_score(pred_data, label_data)
            # print(f"Dice score for {pred}: {dice_pred}")

        # Save the dice score
        dice_scores[image] = dice

    # Save the results  
    with open(os.path.join(output_dir, "dice_scores.txt"), "w") as f:
        for key, value in dice_scores.items():
            f.write(f"{key}: {value}\n")
    
    return None


if __name__ == "__main__":
    main()
