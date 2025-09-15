"""
This script is used to calibrate the model predictions by thresholding the lesion segmentations.
Here we binarize the lesions with a treshold from 0 to 1 by steps of 0.2.

Input:
    --pred-folder: Path to the folder containing the soft lesion segmentation predictions.
    --output: Path to the output folder where the calibrated lesion segmentations will be saved.

Output:
    None

Example:
    python 4_calibration.py --pred-folder /path/to/predictions --output /path/to/output

Author: Pierre-Louis Benveniste
"""
import argparse
import os
from pathlib import Path
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Calibrate lesion segmentations by thresholding.")
    parser.add_argument("--pred-folder", type=str, required=True, help="Path to the folder containing the soft lesion segmentation predictions.")
    parser.add_argument("--output", type=str, required=True, help="Path to the output folder where the calibrated lesion segmentations will be saved.")
    return parser.parse_args()


def main():
    # Parse command line arguments
    args = parse_args()
    pred_folder = args.pred_folder
    output_folder = args.output

    # thresholds
    threshold_list = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    # Build the output folder if it does not exist
    os.makedirs(output_folder, exist_ok=True)
    for thresh in threshold_list:
        # Build the folder corresponding to the threshold
        thresh_folder = os.path.join(output_folder, f"thresh_{thresh}")
        os.makedirs(thresh_folder, exist_ok=True)
    
    # List all predictions
    list_pred = sorted(list(Path(pred_folder).rglob("*.nii.gz")))

    # For each prediction, apply the threshold and save the result
    for pred in tqdm(list_pred):
        # We iterate over the thresholds
        for thresh in threshold_list:
            # Build the output file path
            output_file = os.path.join(output_folder, f"thresh_{thresh}", Path(pred).name)

            # Use SCT to threshold the lesion segmentation
            assert os.system(f"sct_maths -i {pred} -bin {thresh} -o {output_file}") == 0, "SCT command failed"

    print("Done with thresholding")

    return None


if __name__ == "__main__":
    main()