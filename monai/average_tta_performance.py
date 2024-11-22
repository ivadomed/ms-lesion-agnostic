"""
This file is used to get all the dice_scores_X.txt files in a directory and average them.

Input:
    - Path to the directory containing the dice_scores_X.txt files

Output:
    None

Example:
    python average_tta_performance.py --pred-dir-path /path/to/dice_scores

Author: Pierre-Louis Benveniste
"""

import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path


def get_parser():
    """
    This function returns the parser for the command line arguments.
    """
    parser = argparse.ArgumentParser(description="Average the performance of the model")
    parser.add_argument("--pred-dir-path", help="Path to the directory containing the dice_scores_X.txt files", required=True)
    return parser


def main():
    """
    This function is used to average the performance of the model on the test set.

    Args:
        None

    Returns:
        None
    """
    # Get the parser
    parser = get_parser()
    args = parser.parse_args()
    
    # Path to the dice_scores
    path_to_outputs = args.pred_dir_path
    
    # Get all the dice_scores_X.txt files using rglob
    dice_score_files = [str(file) for file in Path(path_to_outputs).rglob("dice_scores_*.txt")]
    
    # Dict to store the dice scores
    dice_scores = {}
    
    # Loop over the dice_scores_X.txt files
    for dice_score_file in dice_score_files:
        # Open dice results (they are txt files)
        with open(os.path.join(path_to_outputs, dice_score_file), 'r') as file:
            for line in file:
                key, value = line.strip().split(':')
                if key in dice_scores:
                    dice_scores[key].append(float(value))
                else:
                    dice_scores[key] = [float(value)]
    
    # Average the dice scores ang get standard deviation
    std = {}
    for key in dice_scores:
        std[key] = np.std(dice_scores[key])
        dice_scores[key] = np.mean(dice_scores[key])
    
    # Save the averaged dice scores
    with open(os.path.join(path_to_outputs, "dice_scores.txt"), 'w') as file:
        for key in dice_scores:
            file.write(f"{key}: {dice_scores[key]}\n")
    
    # Save the standard deviation
    with open(os.path.join(path_to_outputs, "std.txt"), 'w') as file:
        for key in std:
            file.write(f"{key}: {std[key]}\n")


if __name__ == "__main__":
    main()