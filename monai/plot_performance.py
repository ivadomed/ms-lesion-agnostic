""""
This script is used to plot the performance of the model on the test set, validation and train set.
It saves a plot of dice scores per contrat in the output folder 

"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import argparse


def get_parser():
    """
    This function returns the parser for the command line arguments.
    """
    parser = argparse.ArgumentParser(description="Plot the performance of the model")
    parser.add_argument("--pred-dir-path", help="Path to the directory containing the dice_score.txt file", required=True)
    return parser


def main():
    """
    This function is used to plot the performance of the model on the test set.

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
    dice_score_file = path_to_outputs + '/dice_scores.txt'

    # Open dice results (they are txt files)
    test_dice_results = {}
    with open(dice_score_file, 'r') as file:
        for line in file:
            key, value = line.strip().split(':')
            test_dice_results[key] = float(value)

    # convert to a df with name and dice score
    test_dice_results = pd.DataFrame(list(test_dice_results.items()), columns=['name', 'dice_score'])

    # Add the contrats column 
    test_dice_results['contrast'] = test_dice_results['name'].apply(lambda x: x.split('_')[-1])

    # plot a violin plot per contrast
    plt.figure(figsize=(20, 10))
    sns.violinplot(x='contrast', y='dice_score', data=test_dice_results)
    plt.title('Dice scores per contrast')
    plt.show()

    # Save the plot
    plt.savefig(path_to_outputs + '/dice_scores.png')
    print(f"Saved the dice_scores plot in {path_to_outputs}")


if __name__ == "__main__":
    main()