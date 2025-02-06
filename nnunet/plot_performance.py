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
    # parser.add_argument("--split", help="Data split to use (train, validation, test)", required=True, type=str)
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

    # Join the train, validation and test data
    jsondata = jsondata['train'] + jsondata['validation'] + jsondata['test'] + jsondata['externalValidation']
    
    # Iterate over the test files
    for file in test_dice_results['name']:
        # We find the corresponding file in the json file
        for data in jsondata:
            if data["image"] == file:
                # Add the contrat, the site and the resolution to the df
                test_dice_results.loc[test_dice_results['name'] == file, 'contrast'] = data['contrast']
                test_dice_results.loc[test_dice_results['name'] == file, 'site'] = data['site']
                # test_dice_results.loc[test_dice_results['name'] == file, 'acquisition'] = data['acquisition']
                test_dice_results.loc[test_dice_results['name'] == file, 'nb_lesions'] = data['nb_lesions']
                test_dice_results.loc[test_dice_results['name'] == file, 'total_lesion_volume'] = data['total_lesion_volume']
    
    # Replace all the MEGRE contrast by T2star
    test_dice_results['contrast'] = test_dice_results['contrast'].apply(lambda x: 'T2star' if x == 'MEGRE' else x)

    # Count the number of samples per contrast
    contrast_counts = test_dice_results['contrast'].value_counts()
    
    # In the df replace the contrats by the number of samples of the contarsts( for example, T2 becomes T2 (n=10))
    test_dice_results['contrast_count'] = test_dice_results['contrast'].apply(lambda x: x + f' (n={contrast_counts[x]})')

    # Same for the site
    site_counts = test_dice_results['site'].value_counts()
    test_dice_results['site_count'] = test_dice_results['site'].apply(lambda x: x + f' (n={site_counts[x]})')

    # Same for the resolution
    # resolution_counts = test_dice_results['acquisition'].value_counts()
    # test_dice_results['acquisition_count'] = test_dice_results['acquisition'].apply(lambda x: x + f' (n={resolution_counts[x]})')

    # then we add the ppv score to the df
    ppv_score_file = path_to_outputs + '/ppv_scores.txt'
    ppv_scores = {}
    with open(ppv_score_file, 'r') as file:
        for line in file:
            key, value = line.strip().split(':')
            ppv_scores[key] = float(value)
    test_dice_results['ppv_score'] = test_dice_results['name'].apply(lambda x: ppv_scores[x])

    # then we add the f1 score to the df
    f1_score_file = path_to_outputs + '/f1_scores.txt'
    f1_scores = {}
    with open(f1_score_file, 'r') as file:
        for line in file:
            key, value = line.strip().split(':')
            f1_scores[key] = float(value)
    test_dice_results['f1_score'] = test_dice_results['name'].apply(lambda x: f1_scores[x])

    # then we add the sensitivity score to the df
    sensitivity_score_file = path_to_outputs + '/sensitivity_scores.txt'
    sensitivity_scores = {}
    with open(sensitivity_score_file, 'r') as file:
        for line in file:
            key, value = line.strip().split(':')
            sensitivity_scores[key] = float(value)
    test_dice_results['sensitivity_score'] = test_dice_results['name'].apply(lambda x: sensitivity_scores[x])

    # We rename th df to metrics_results
    metrics_results = test_dice_results

    # Sort the order of the lines by contrast (alphabetical order)
    metrics_results = metrics_results.sort_values(by='contrast').reset_index(drop=True)

    # plot a violin plot per contrast for dice scores
    plt.figure(figsize=(20, 10))
    plt.grid(True)
    sns.violinplot(x='contrast_count', y='dice_score', data=metrics_results)
    # y ranges from -0.2 to 1.2
    plt.ylim(-0.2, 1.2)
    plt.title('Dice scores per contrast')
    plt.show()
    # # Save the plot
    plt.savefig(path_to_outputs + '/dice_scores_contrast.png')
    print(f"Saved the dice plot in {path_to_outputs}")

    # plot a violin plot per contrast for ppv scores
    plt.figure(figsize=(20, 10))
    plt.grid(True)
    sns.violinplot(x='contrast_count', y='ppv_score', data=metrics_results)
    # y ranges from -0.2 to 1.2
    plt.ylim(-0.2, 1.2)
    plt.title('PPV scores per contrast')
    plt.show()

    # # Save the plot
    plt.savefig(path_to_outputs + '/ppv_scores_contrast.png')
    print(f"Saved the ppv plot in {path_to_outputs}")

    # plot a violin plot per contrast for f1 scores
    plt.figure(figsize=(20, 10))
    plt.grid(True)
    sns.violinplot(x='contrast_count', y='f1_score', data=metrics_results)
    # y ranges from -0.2 to 1.2
    plt.ylim(-0.2, 1.2)
    plt.title('F1 scores per contrast')
    plt.show()

    # # Save the plot
    plt.savefig(path_to_outputs + '/f1_scores_contrast.png')
    print(f"Saved the F1 plot in {path_to_outputs}")

    # plot a violin plot per contrast for f1 scores
    plt.figure(figsize=(20, 10))
    plt.grid(True)
    sns.violinplot(x='contrast_count', y='sensitivity_score', data=metrics_results)
    # y ranges from -0.2 to 1.2
    plt.ylim(-0.2, 1.2)
    plt.title('Sensitivity scores per contrast')
    plt.show()

    # # Save the plot
    plt.savefig(path_to_outputs + '/sensitivity_scores_contrast.png')
    print(f"Saved the sensitivity plot in {path_to_outputs}")

    # Print the mean dice score per contrast and std
    print("\nDice score per contrast (mean ± std)")
    dice_stats = metrics_results.groupby('contrast_count')['dice_score'].agg(['mean', 'std'])
    # print global dice score and std
    print(f"Global dice score: {metrics_results['dice_score'].mean():.4f} ± {metrics_results['dice_score'].std():.4f}")
    for contrast, row in dice_stats.iterrows():
        print(f"{contrast}: {row['mean']:.4f} ± {row['std']:.4f}")

    # Print the mean ppv score per contrast and std
    print("\nPPV score per contrast (mean ± std)")
    ppv_stats = metrics_results.groupby('contrast_count')['ppv_score'].agg(['mean', 'std'])
    # print global ppv score and std
    print(f"Global ppv score: {metrics_results['ppv_score'].mean():.4f} ± {metrics_results['ppv_score'].std():.4f}")
    for contrast, row in ppv_stats.iterrows():
        print(f"{contrast}: {row['mean']:.4f} ± {row['std']:.4f}")
    
    # Print the mean f1 score per contrast and std
    print("\nF1 score per contrast (mean ± std)")
    f1_stats = metrics_results.groupby('contrast_count')['f1_score'].agg(['mean', 'std'])
    # print global f1 score and std
    print(f"Global f1 score: {metrics_results['f1_score'].mean():.4f} ± {metrics_results['f1_score'].std():.4f}")
    for contrast, row in f1_stats.iterrows():
        print(f"{contrast}: {row['mean']:.4f} ± {row['std']:.4f}")

    # Print the mean sensitivity score per contrast and std
    print("\nSensitivity score per contrast (mean ± std)")
    sensitivity_stats = metrics_results.groupby('contrast_count')['sensitivity_score'].agg(['mean', 'std'])
    # print global sensitivity score and std
    print(f"Global sensitivity score: {metrics_results['sensitivity_score'].mean():.4f} ± {metrics_results['sensitivity_score'].std():.4f}")
    for contrast, row in sensitivity_stats.iterrows():
        print(f"{contrast}: {row['mean']:.4f} ± {row['std']:.4f}")
    
    # Save the prints in a txt file
    with open(path_to_outputs + '/metrics_stats.txt', 'w') as f:
        f.write("Dice score per contrast (mean ± std)\n")
        f.write(f"Global dice score: {metrics_results['dice_score'].mean():.4f} ± {metrics_results['dice_score'].std():.4f}\n")
        for contrast, row in dice_stats.iterrows():
            f.write(f"{contrast}: {row['mean']:.4f} ± {row['std']:.4f}\n")
        f.write("\nPPV score per contrast (mean ± std)\n")
        f.write(f"Global ppv score: {metrics_results['ppv_score'].mean():.4f} ± {metrics_results['ppv_score'].std():.4f}\n")
        for contrast, row in ppv_stats.iterrows():
            f.write(f"{contrast}: {row['mean']:.4f} ± {row['std']:.4f}\n")
        f.write("\nF1 score per contrast (mean ± std)\n")
        f.write(f"Global f1 score: {metrics_results['f1_score'].mean():.4f} ± {metrics_results['f1_score'].std():.4f}\n")
        for contrast, row in f1_stats.iterrows():
            f.write(f"{contrast}: {row['mean']:.4f} ± {row['std']:.4f}\n")
        f.write("\nSensitivity score per contrast (mean ± std)\n")
        f.write(f"Global sensitivity score: {metrics_results['sensitivity_score'].mean():.4f} ± {metrics_results['sensitivity_score'].std():.4f}\n")
        for contrast, row in sensitivity_stats.iterrows():
            f.write(f"{contrast}: {row['mean']:.4f} ± {row['std']:.4f}\n")
    
    return None


if __name__ == "__main__":
    main()