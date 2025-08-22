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
from pathlib import Path
import os


def get_parser():
    """
    This function returns the parser for the command line arguments.
    """
    parser = argparse.ArgumentParser(description="Plot the performance of the model")
    parser.add_argument("--pred-dir-path", help="Path to the directory containing the dice_score.txt file. The structure should be /0.5/dice_score.txt for resampling by 0.5", required=True)
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

    # List of factors
    factors = [0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7]

    # List all the score files
    dice_score_files = []
    sens_score_files = []
    ppv_score_files = []
    f1_score_files = []
    for factor in factors:
        dice_score_files.append(Path(path_to_outputs) / f"{factor}" / "dice_scores.txt")
        sens_score_files.append(Path(path_to_outputs) / f"{factor}" / "sensitivity_scores.txt")
        ppv_score_files.append(Path(path_to_outputs) / f"{factor}" / "ppv_scores.txt")
        f1_score_files.append(Path(path_to_outputs) / f"{factor}" / "f1_scores.txt")

    # Now we store all the score in a dataframe
    metrics_results = pd.DataFrame(columns=['name', 'factor', 'dice'])
    for i, factor in enumerate(factors):
        # We open the dice score file corresponding to the current factor
        with open(dice_score_files[i], 'r') as file:
            for line in file:
                name, dice = line.strip().split(':')
                # We store the value as a line in the df with column: name, factor and score
                metrics_results = pd.concat([metrics_results, pd.DataFrame({'name': [name], 'factor': [factor], 'dice': [float(dice)]})], ignore_index=True)
        # We do the same with sensitivity
        with open(sens_score_files[i], 'r') as file:
            for line in file:
                name, sens = line.strip().split(':')
                # We store the value as a line in the df with column: name, factor and score
                metrics_results = pd.concat([metrics_results, pd.DataFrame({'name': [name], 'factor': [factor], 'sensitivity': [float(sens)]})], ignore_index=True)
        # We do the same with ppv
        with open(ppv_score_files[i], 'r') as file:
            for line in file:
                name, ppv = line.strip().split(':')
                # We store the value as a line in the df with column: name, factor and score
                metrics_results = pd.concat([metrics_results, pd.DataFrame({'name': [name], 'factor': [factor], 'ppv': [float(ppv)]})], ignore_index=True)
        # We do the same with f1
        with open(f1_score_files[i], 'r') as file:
            for line in file:
                name, f1 = line.strip().split(':')
                # We store the value as a line in the df with column: name, factor and score
                metrics_results = pd.concat([metrics_results, pd.DataFrame({'name': [name], 'factor': [factor], 'f1': [float(f1)]})], ignore_index=True)

    # Create an empty column for the contrast, the site and the resolution
    metrics_results['contrast'] = None
    metrics_results['site'] = None
    metrics_results['resolution_RL'] = None
    metrics_results['resolution_AP'] = None
    metrics_results['resolution_IS'] = None

    # Load the data json file
    data_json_path = args.data_json_path
    with open(data_json_path, 'r') as f:
        jsondata = json.load(f)

    # Join the train, validation and test data
    jsondata = jsondata['train'] + jsondata['validation'] + jsondata['test'] + jsondata['externalValidation']
    
    # Iterate over the test files
    for file in metrics_results['name']:
        # We find the corresponding file in the json file
        for data in jsondata:
            if data["image"] == file:
                # Add the contrat, the site and the resolution to the df
                metrics_results.loc[metrics_results['name'] == file, 'contrast'] = data['contrast']
                metrics_results.loc[metrics_results['name'] == file, 'site'] = data['site']
                metrics_results.loc[metrics_results['name'] == file, 'nb_lesions'] = data['nb_lesions']
                metrics_results.loc[metrics_results['name'] == file, 'total_lesion_volume'] = data['total_lesion_volume']
                metrics_results.loc[metrics_results['name'] == file, 'resolution_RL'] = data['resolution'][0]
                metrics_results.loc[metrics_results['name'] == file, 'resolution_AP'] = data['resolution'][1]
                metrics_results.loc[metrics_results['name'] == file, 'resolution_IS'] = data['resolution'][2]

    # Replace all the MEGRE contrast by T2star
    metrics_results['contrast'] = metrics_results['contrast'].apply(lambda x: 'T2star' if x == 'MEGRE' else x)

    # In the df, we compute the final resolution of the images
    metrics_results['final_resolution_RL'] = metrics_results['resolution_RL'] / metrics_results['factor']
    metrics_results['final_resolution_AP'] = metrics_results['resolution_AP'] / metrics_results['factor']
    metrics_results['final_resolution_IS'] = metrics_results['resolution_IS'] / metrics_results['factor']
    metrics_results['voxel_volume'] = metrics_results['final_resolution_RL'] * metrics_results['final_resolution_AP'] * metrics_results['final_resolution_IS']

    ###########################################
    # Dice
    ###########################################
    # We first plot the dice performance per resampling factor with bar plots
    plt.figure(figsize=(10, 6))
    sns.barplot(data=metrics_results, x='factor', y='dice')
    plt.title('Dice per resampling factor')
    plt.xlabel('Resampling factor')
    plt.ylabel('Dice score')
    plt.savefig(os.path.join(path_to_outputs, 'dice_per_resampling_factor.png'))

    # We make 3 subplots with the dice score per final resolution (RL, AP and IS)
    # Line plots show the average dice score by bins of 1
    metrics_results_copy = metrics_results.copy()
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    for ax, axis in zip(axs, ["RL", "AP", "IS"]):
        col = f"final_resolution_{axis}"
        
        # Bin the data with width 1
        metrics_results_copy[f"{col}_bin"] = pd.cut(metrics_results_copy[col], 
                                            bins=np.arange(metrics_results_copy[col].min(),
                                                            metrics_results_copy[col].max() + 1,
                                                            1))
        
        # Compute mean dice per bin
        grouped = metrics_results_copy.groupby(f"{col}_bin")["dice"].mean().reset_index()
        
        # For plotting, use bin midpoints instead of categorical intervals
        grouped["bin_center"] = grouped[f"{col}_bin"].apply(lambda x: x.mid)
        # Also compute the proportion of samples in each bin
        grouped["sample_proportion"] = metrics_results_copy.groupby(f"{col}_bin")["dice"].count().reset_index(drop=True) / len(metrics_results_copy)

        # We also plot on the right vert axis the proportion of samples in each bin
        ax2 = ax.twinx()
        sns.lineplot(data=grouped, x="bin_center", y="sample_proportion", ax=ax2, color="gray", marker="o")
        ax2.set_ylabel("Proportion of samples")

        sns.lineplot(data=grouped, x="bin_center", y="dice", ax=ax, marker="o")
        ax.set_title(f"Dice per inference resolution ({axis})")
        ax.set_xlabel(f"Inference resolution ({axis})")
        ax.set_ylabel("Dice score")
    plt.tight_layout()
    plt.savefig(os.path.join(path_to_outputs, 'dice_per_inference_resolution.png'))
    plt.close()

    # We now plot the performance per voxel volume by bins of 10
    fig, ax = plt.subplots(figsize=(7, 5))
    metrics_results_copy = metrics_results.copy()
    metrics_results_copy["voxel_volume_bin"] = pd.cut(metrics_results_copy["voxel_volume"], 
                                            bins=np.arange(metrics_results_copy["voxel_volume"].min(),
                                                           metrics_results_copy["voxel_volume"].max() + 10,
                                                           10))

    # Compute mean dice per bin
    grouped = metrics_results_copy.groupby("voxel_volume_bin")["dice"].mean().reset_index()

    # For plotting, use bin midpoints instead of categorical intervals
    grouped["bin_center"] = grouped["voxel_volume_bin"].apply(lambda x: x.mid)

    sns.lineplot(data=grouped, x="bin_center", y="dice", ax=ax, marker="o")
    ax.set_title(f"Dice per voxel volume")
    ax.set_xlabel(f"Voxel volume (mm³)")
    ax.set_ylabel("Dice score")
    plt.tight_layout()
    plt.savefig(os.path.join(path_to_outputs, 'dice_per_voxel_volume.png'))
    plt.close()

    ###########################################
    # PPV
    ###########################################
    # We first plot the dice performance per resampling factor with bar plots
    plt.figure(figsize=(10, 6))
    sns.barplot(data=metrics_results, x='factor', y='ppv')
    plt.title('PPV per resampling factor')
    plt.xlabel('Resampling factor')
    plt.ylabel('PPV score')
    plt.savefig(os.path.join(path_to_outputs, 'ppv_per_resampling_factor.png'))

    # We make 3 subplots with the PPV score per final resolution (RL, AP and IS)
    # Line plots show the average PPV score by bins of 1
    metrics_results_copy = metrics_results.copy()
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    for ax, axis in zip(axs, ["RL", "AP", "IS"]):
        col = f"final_resolution_{axis}"
        
        # Bin the data with width 1
        metrics_results_copy[f"{col}_bin"] = pd.cut(metrics_results_copy[col], 
                                            bins=np.arange(metrics_results_copy[col].min(),
                                                            metrics_results_copy[col].max() + 1,
                                                            1))

        # Compute mean PPV per bin
        grouped = metrics_results_copy.groupby(f"{col}_bin")["ppv"].mean().reset_index()

        # For plotting, use bin midpoints instead of categorical intervals
        grouped["bin_center"] = grouped[f"{col}_bin"].apply(lambda x: x.mid)
        # Also compute the proportion of samples in each bin
        grouped["sample_proportion"] = metrics_results_copy.groupby(f"{col}_bin")["ppv"].count().reset_index(drop=True) / len(metrics_results_copy)

        # We also plot on the right vert axis the proportion of samples in each bin
        ax2 = ax.twinx()
        sns.lineplot(data=grouped, x="bin_center", y="sample_proportion", ax=ax2, color="gray", marker="o")
        ax2.set_ylabel("Proportion of samples")

        sns.lineplot(data=grouped, x="bin_center", y="ppv", ax=ax, marker="o")
        ax.set_title(f"PPV per inference resolution ({axis})")
        ax.set_xlabel(f"Final resolution ({axis})")
        ax.set_ylabel("PPV score")
    plt.tight_layout()
    plt.savefig(os.path.join(path_to_outputs, 'ppv_per_inference_resolution.png'))
    plt.close()

    # We now plot the performance per voxel volume by bins of 10
    fig, ax = plt.subplots(figsize=(7, 5))
    metrics_results_copy = metrics_results.copy()
    metrics_results_copy["voxel_volume_bin"] = pd.cut(metrics_results_copy["voxel_volume"], 
                                            bins=np.arange(metrics_results_copy["voxel_volume"].min(),
                                                           metrics_results_copy["voxel_volume"].max() + 10,
                                                           10))

    # Compute mean PPV per bin
    grouped = metrics_results_copy.groupby("voxel_volume_bin")["ppv"].mean().reset_index()

    # For plotting, use bin midpoints instead of categorical intervals
    grouped["bin_center"] = grouped["voxel_volume_bin"].apply(lambda x: x.mid)

    sns.lineplot(data=grouped, x="bin_center", y="ppv", ax=ax, marker="o")
    ax.set_title(f"PPV per voxel volume")
    ax.set_xlabel(f"Voxel volume (mm³)")
    ax.set_ylabel("PPV score")
    plt.tight_layout()
    plt.savefig(os.path.join(path_to_outputs, 'ppv_per_voxel_volume.png'))
    plt.close()

    ###########################################
    # Sensitivity
    ###########################################
    # We first plot the dice performance per resampling factor with bar plots
    plt.figure(figsize=(10, 6))
    sns.barplot(data=metrics_results, x='factor', y='sensitivity')
    plt.title('Sensitivity per resampling factor')
    plt.xlabel('Resampling factor')
    plt.ylabel('Sensitivity score')
    plt.savefig(os.path.join(path_to_outputs, 'sensitivity_per_resampling_factor.png'))

    # We make 3 subplots with the sensitivity score per final resolution (RL, AP and IS)
    # Line plots show the average sensitivity score by bins of 1
    metrics_results_copy = metrics_results.copy()
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    for ax, axis in zip(axs, ["RL", "AP", "IS"]):
        col = f"final_resolution_{axis}"
        
        # Bin the data with width 1
        metrics_results_copy[f"{col}_bin"] = pd.cut(metrics_results_copy[col], 
                                            bins=np.arange(metrics_results_copy[col].min(),
                                                            metrics_results_copy[col].max() + 1,
                                                            1))

        # Compute mean sensitivity per bin
        grouped = metrics_results_copy.groupby(f"{col}_bin")["sensitivity"].mean().reset_index()

        # For plotting, use bin midpoints instead of categorical intervals
        grouped["bin_center"] = grouped[f"{col}_bin"].apply(lambda x: x.mid)
        # Also compute the proportion of samples in each bin
        grouped["sample_proportion"] = metrics_results_copy.groupby(f"{col}_bin")["sensitivity"].count().reset_index(drop=True) / len(metrics_results_copy)

        # We also plot on the right vert axis the proportion of samples in each bin
        ax2 = ax.twinx()
        sns.lineplot(data=grouped, x="bin_center", y="sample_proportion", ax=ax2, color="gray", marker="o")
        ax2.set_ylabel("Proportion of samples")

        sns.lineplot(data=grouped, x="bin_center", y="sensitivity", ax=ax, marker="o")
        ax.set_title(f"Sensitivity per inference resolution ({axis})")
        ax.set_xlabel(f"Inference resolution ({axis})")
        ax.set_ylabel("Sensitivity score")
    plt.tight_layout()
    plt.savefig(os.path.join(path_to_outputs, 'sensitivity_per_inference_resolution.png'))
    plt.close()

    # We now plot the performance per voxel volume by bins of 10
    fig, ax = plt.subplots(figsize=(7, 5))
    metrics_results_copy = metrics_results.copy()
    metrics_results_copy["voxel_volume_bin"] = pd.cut(metrics_results_copy["voxel_volume"], 
                                            bins=np.arange(metrics_results_copy["voxel_volume"].min(),
                                                           metrics_results_copy["voxel_volume"].max() + 10,
                                                           10))

    # Compute mean sensitivity per bin
    grouped = metrics_results_copy.groupby("voxel_volume_bin")["sensitivity"].mean().reset_index()

    # For plotting, use bin midpoints instead of categorical intervals
    grouped["bin_center"] = grouped["voxel_volume_bin"].apply(lambda x: x.mid)

    sns.lineplot(data=grouped, x="bin_center", y="sensitivity", ax=ax, marker="o")
    ax.set_title(f"Sensitivity per voxel volume")
    ax.set_xlabel(f"Voxel volume (mm³)")
    ax.set_ylabel("Sensitivity score")
    plt.tight_layout()
    plt.savefig(os.path.join(path_to_outputs, 'sensitivity_per_voxel_volume.png'))
    plt.close()

    ###########################################
    # F1 score
    ###########################################
    # We first plot the dice performance per resampling factor with bar plots
    plt.figure(figsize=(10, 6))
    sns.barplot(data=metrics_results, x='factor', y='f1')
    plt.title('F1 score per resampling factor')
    plt.xlabel('Resampling factor')
    plt.ylabel('F1 score')
    plt.savefig(os.path.join(path_to_outputs, 'f1_per_resampling_factor.png'))

    # We make 3 subplots with the F1 score per final resolution (RL, AP and IS)
    # Line plots show the average F1 score by bins of 1
    metrics_results_copy = metrics_results.copy()
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    for ax, axis in zip(axs, ["RL", "AP", "IS"]):
        col = f"final_resolution_{axis}"
        
        # Bin the data with width 1
        metrics_results_copy[f"{col}_bin"] = pd.cut(metrics_results_copy[col], 
                                            bins=np.arange(metrics_results_copy[col].min(),
                                                            metrics_results_copy[col].max() + 1,
                                                            1))

        # Compute mean F1 per bin
        grouped = metrics_results_copy.groupby(f"{col}_bin")["f1"].mean().reset_index()

        # For plotting, use bin midpoints instead of categorical intervals
        grouped["bin_center"] = grouped[f"{col}_bin"].apply(lambda x: x.mid)
        # Also compute the proportion of samples in each bin
        grouped["sample_proportion"] = metrics_results_copy.groupby(f"{col}_bin")["f1"].count().reset_index(drop=True) / len(metrics_results_copy)

        # We also plot on the right vert axis the proportion of samples in each bin
        ax2 = ax.twinx()
        sns.lineplot(data=grouped, x="bin_center", y="sample_proportion", ax=ax2, color="gray", marker="o")
        ax2.set_ylabel("Proportion of samples")

        sns.lineplot(data=grouped, x="bin_center", y="f1", ax=ax, marker="o")
        ax.set_title(f"F1 score per inference resolution ({axis})")
        ax.set_xlabel(f"Inference resolution ({axis})")
        ax.set_ylabel("F1 score")
    plt.tight_layout()
    plt.savefig(os.path.join(path_to_outputs, 'f1_per_inference_resolution.png'))
    plt.close()

    # We now plot the performance per voxel volume by bins of 10
    fig, ax = plt.subplots(figsize=(7, 5))
    metrics_results_copy = metrics_results.copy()
    metrics_results_copy["voxel_volume_bin"] = pd.cut(metrics_results_copy["voxel_volume"], 
                                            bins=np.arange(metrics_results_copy["voxel_volume"].min(),
                                                           metrics_results_copy["voxel_volume"].max() + 10,
                                                           10))

    # Compute mean F1 per bin
    grouped = metrics_results_copy.groupby("voxel_volume_bin")["f1"].mean().reset_index()

    # For plotting, use bin midpoints instead of categorical intervals
    grouped["bin_center"] = grouped["voxel_volume_bin"].apply(lambda x: x.mid)

    sns.lineplot(data=grouped, x="bin_center", y="f1", ax=ax, marker="o")
    ax.set_title(f"F1 per voxel volume")
    ax.set_xlabel(f"Voxel volume (mm³)")
    ax.set_ylabel("F1 score")
    plt.tight_layout()
    plt.savefig(os.path.join(path_to_outputs, 'f1_per_voxel_volume.png'))
    plt.close()

    # Now we save in a txt file each scores per factor per contrast
    # Save summary scores per contrast in a readable format
    metrics = [
        ("dice", "Dice score"),
        ("ppv", "PPV score"),
        ("f1", "F1 score"),
        ("sensitivity", "Sensitivity score"),
    ]
    # Save 1 file per factor
    for factor in factors:
        factor_df = metrics_results[metrics_results['factor'] == factor]
        out_path = os.path.join(path_to_outputs, f'scores_per_contrast_factor_{factor}.txt')
        with open(out_path, 'w') as f:
            for metric_col, metric_name in metrics:
                f.write(f"{metric_name} per contrast (mean ± std)\n")
                global_mean = factor_df[metric_col].mean()
                global_std = factor_df[metric_col].std()
                f.write(f"Global {metric_col} score: {global_mean:.4f} ± {global_std:.4f}\n")
                for contrast in factor_df['contrast'].dropna().unique():
                    subset = factor_df[factor_df['contrast'] == contrast]
                    mean = subset[metric_col].mean()
                    std = subset[metric_col].std()
                    n = len(subset)
                    f.write(f"{contrast} (n={n}): {mean:.4f} ± {std:.4f}\n")
                f.write("\n")

    return None


if __name__ == "__main__":
    main()