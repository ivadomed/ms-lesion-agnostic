"""
This script averages the predictions from the 5 folds of the models. 
It also binarizes the averaged predictions to create a final segmentation mask with threshold 0.5.

Input:
    -input-fold0: Path to the first fold's predictions
    -output-avg: Path to save the averaged predictions
    -output-bin: Path to save the binarized predictions
    -thresh: Threshold for binarization (default is 0.5)

Output:
    None

Example usage:
    python average_predictions.py -input-fold0 /path/to/fold0/pred/ -output-avg /path/to/avg/pred -output-bin /path/to/bin/pred

Author: Pierre-Louis Benveniste
"""
import argparse
import os
from pathlib import Path
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Average and binarize predictions from multiple folds")
    parser.add_argument("-input-fold0", type=str, required=True, help="Path to the first fold's predictions")
    parser.add_argument("-output-avg", type=str, required=True, help="Path to save the averaged predictions")
    parser.add_argument("-output-bin", type=str, required=True, help="Path to save the binarized predictions")
    parser.add_argument("-thresh", type=float, default=0.5, help="Threshold for binarization (default: 0.5)")

    return parser.parse_args()


def main():
    args = parse_args()
    
    input_fold0 = Path(args.input_fold0)
    output_avg = Path(args.output_avg)
    output_bin = Path(args.output_bin)
    thresh = args.thresh

    # Ensure output directories exist
    output_avg.mkdir(parents=True, exist_ok=True)
    output_bin.mkdir(parents=True, exist_ok=True)

    # Build the path to predictions from the other folds
    input_fold1 = str(input_fold0).replace("fold0", "fold1")
    input_fold2 = str(input_fold0).replace("fold0", "fold2")
    input_fold3 = str(input_fold0).replace("fold0", "fold3")
    input_fold4 = str(input_fold0).replace("fold0", "fold4")

    # List predictions
    pred_fold0 = sorted(list(Path(input_fold0).rglob("*.nii.gz")))
    pred_fold0 = [str(f) for f in pred_fold0]

    # For each prediction file, average the predictions from all folds:
    for file_fold0 in tqdm(pred_fold0):
        # Build path to corresponding files from other folds
        file_fold1 = str(file_fold0).replace("fold0", "fold1")
        file_fold2 = str(file_fold0).replace("fold0", "fold2")
        file_fold3 = str(file_fold0).replace("fold0", "fold3")
        file_fold4 = str(file_fold0).replace("fold0", "fold4")

        # Average the predictions
        avg_file = os.path.join(output_avg, Path(file_fold0).name)
        assert os.system(f"sct_maths -i {file_fold0} -add {file_fold1} {file_fold2} {file_fold3} {file_fold4} -o {avg_file} -type float64") == 0
        # Binarize the averaged predictions
        bin_file = os.path.join(output_bin, Path(file_fold0).name)
        assert os.system(f"sct_maths -i {avg_file} -bin {thresh} -o {bin_file}") == 0

    return None


if __name__ == "__main__":
    main()