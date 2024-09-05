"""
This script plots the performance of the model based on the threshold applied to the predictions.

Input:
    --path-scores: Path to the directory containing the dice_scores_X.txt files

Output:
    None

Example:
    python plot_optThresh.py --path-scores /path/to/dice_scores

Author: Pierre-Louis Benveniste
"""

import os
import argparse
import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def get_parser():
    """
    This function returns the parser for the command line arguments.
    """
    parser = argparse.ArgumentParser(description="Plot the optimal threshold")
    parser.add_argument("--path-scores", help="Path to the directory containing the dice_scores_X.txt files", required=True)
    return parser


def main():

    # Get the parser
    parser = get_parser()
    args = parser.parse_args()

    # Path to the dice_scores
    path_to_outputs = args.path_scores

    # Get all the dice_scores_X.txt files using rglob
    dice_score_files = [str(file) for file in Path(path_to_outputs).rglob("dice_scores_*.txt")]

    # Create a list to store the dataframes
    test_dice_results_list = [None] * len(dice_score_files)

    # For each file, get the threshold and the dice score
    for i, dice_score_file in enumerate(dice_score_files):
        test_dice_results = {}
        with open(dice_score_file, 'r') as file:
            for line in file:
                key, value = line.strip().split(':')
                test_dice_results[key] = float(value)
        # convert to a df with name and dice score
        test_dice_results_list[i] = pd.DataFrame(list(test_dice_results.items()), columns=['name', 'dice_score'])
        # Create a column which stores the threshold
        test_dice_results_list[i]['threshold'] = str(Path(dice_score_file).name).replace('dice_scores_', '').replace('.txt', '').replace('_', '.')

    # Concatenate all the dataframes
    test_dice_results = pd.concat(test_dice_results_list)

    # Plot
    plt.figure(figsize=(20, 10))
    plt.grid(True)
    sns.violinplot(x='threshold', y='dice_score', data=test_dice_results)
    # y ranges from -0.2 to 1.2
    plt.ylim(-0.2, 1.2)
    plt.title('Dice scores per threshold')
    plt.show()

    # Save the plot
    plt.savefig(path_to_outputs + '/dice_scores_contrast.png')
    print(f"Saved the dice_scores plot in {path_to_outputs}")

    # Print the average dice score per threshold
    for thresh in test_dice_results['threshold'].unique():
        print(f"Threshold: {thresh} - Average dice score: {test_dice_results[test_dice_results['threshold'] == thresh]['dice_score'].mean()}")

    return None


if __name__ == "__main__":
    main()