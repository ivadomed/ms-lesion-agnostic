""""
This script is used to plot the performance of the model on the test set, validation and train set.
It saves a plot of dice scores per contrat in the output folder 

"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import argparse
import json


def get_parser():
    """
    This function returns the parser for the command line arguments.
    """
    parser = argparse.ArgumentParser(description="Plot the performance of the model")
    parser.add_argument("--pred-dir-path", help="Path to the directory containing the dice_score.txt file", required=True)
    parser.add_argument("--data-json-path", help="Path to the json file containing the data split", required=True)
    parser.add_argument("--split", help="Data split to use (train, validation, test)", required=True, type=str)
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

    # Create an empty column for the contrast, the site and the resolution
    test_dice_results['contrast'] = None
    test_dice_results['site'] = None
    test_dice_results['resolution'] = None

    # Load the data json file
    data_json_path = args.data_json_path
    with open(data_json_path, 'r') as f:
        jsondata = json.load(f)
    
    # Iterate over the test files
    for file in test_dice_results['name']:
        # We find the corresponding file in the json file
        for data in jsondata[args.split]:
            if data["image"] == file:
                # Add the contrat, the site and the resolution to the df
                test_dice_results.loc[test_dice_results['name'] == file, 'contrast'] = data['contrast']
                test_dice_results.loc[test_dice_results['name'] == file, 'site'] = data['site']
                test_dice_results.loc[test_dice_results['name'] == file, 'orientation'] = data['orientation']

    # Count the number of samples per contrast
    contrast_counts = test_dice_results['contrast'].value_counts()
    
    # In the df replace the contrats by the number of samples of the contarsts( for example, T2 becomes T2 (n=10))
    test_dice_results['contrast_count'] = test_dice_results['contrast'].apply(lambda x: x + f' (n={contrast_counts[x]})')

    # Same for the site
    site_counts = test_dice_results['site'].value_counts()
    test_dice_results['site_count'] = test_dice_results['site'].apply(lambda x: x + f' (n={site_counts[x]})')

    # Same for the resolution
    resolution_counts = test_dice_results['orientation'].value_counts()
    test_dice_results['orientation_count'] = test_dice_results['orientation'].apply(lambda x: x + f' (n={resolution_counts[x]})')

    # plot a violin plot per contrast 
    plt.figure(figsize=(20, 10))
    plt.grid(True)
    sns.violinplot(x='contrast_count', y='dice_score', data=test_dice_results)
    # y ranges from -0.2 to 1.2
    plt.ylim(-0.2, 1.2)
    plt.title('Dice scores per contrast')
    plt.show()

    # Save the plot
    plt.savefig(path_to_outputs + '/dice_scores_contrast.png')
    print(f"Saved the dice_scores plot in {path_to_outputs}")

    # plot a violin plot per site
    plt.figure(figsize=(20, 10))
    plt.grid(True)
    sns.violinplot(x='site_count', y='dice_score', data=test_dice_results)
    # y ranges from -0.2 to 1.2
    plt.ylim(-0.2, 1.2)
    plt.title('Dice scores per site')
    plt.show()

    # Save the plot
    plt.savefig(path_to_outputs + '/dice_scores_site.png')
    print(f"Saved the dice_scores per site plot in {path_to_outputs}")

    # plot a violin plot per resolution
    plt.figure(figsize=(20, 10))
    plt.grid(True)
    sns.violinplot(x='orientation_count', y='dice_score', data=test_dice_results)
    # y ranges from -0.2 to 1.2
    plt.ylim(-0.2, 1.2)
    plt.title('Dice scores per orientation')
    plt.show()

    # Save the plot
    plt.savefig(path_to_outputs + '/dice_scores_orientation.png')
    print(f"Saved the dice_scores per orientation plot in {path_to_outputs}")

    # Save the test_dice_results dataframe
    test_dice_results.to_csv(path_to_outputs + '/dice_results.csv', index=False)
    
    return None


if __name__ == "__main__":
    main()