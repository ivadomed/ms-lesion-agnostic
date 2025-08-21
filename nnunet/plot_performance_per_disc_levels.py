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

    # List all dice scores files:
    dice_score_files = list(Path(path_to_outputs).rglob("dice_scores_*.txt"))
    dice_score_file = os.path.join(path_to_outputs, 'dice_scores.txt')

    # Open dice results (they are txt files)
    metrics_results = {}
    with open(dice_score_file, 'r') as file:
        for line in file:
            key, value = line.strip().split(':')
            metrics_results[key] = float(value)

    # convert to a df with name and dice score
    metrics_results = pd.DataFrame(list(metrics_results.items()), columns=['name', 'dice_score'])

    # Now for each file in dice_score_files, we add it as a column by matching file names
    for dice_file in dice_score_files:
        level = str(dice_file).split("/")[-1].replace('.txt', '')
        metrics_results[level] = np.nan  # Create a new column for each file
        with open(dice_file, 'r') as file:
            for line in file:
                key, value = line.strip().split(':')
                metrics_results.loc[metrics_results['name'] == key, level] = float(value)

    # Create an empty column for the contrast, the site and the resolution
    metrics_results['contrast'] = None
    metrics_results['site'] = None
    metrics_results['resolution'] = None

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
                # metrics_results.loc[metrics_results['name'] == file, 'acquisition'] = data['acquisition']
                metrics_results.loc[metrics_results['name'] == file, 'nb_lesions'] = data['nb_lesions']
                metrics_results.loc[metrics_results['name'] == file, 'total_lesion_volume'] = data['total_lesion_volume']
    
    # Replace all the MEGRE contrast by T2star
    metrics_results['contrast'] = metrics_results['contrast'].apply(lambda x: 'T2star' if x == 'MEGRE' else x)

    # Count the number of samples per contrast
    contrast_counts = metrics_results['contrast'].value_counts()
    
    # In the df replace the contrats by the number of samples of the contarsts( for example, T2 becomes T2 (n=10))
    metrics_results['contrast_count'] = metrics_results['contrast'].apply(lambda x: x + f' (n={contrast_counts[x]})')

    # Same for the site
    site_counts = metrics_results['site'].value_counts()
    metrics_results['site_count'] = metrics_results['site'].apply(lambda x: x + f' (n={site_counts[x]})')

    # Now we perform the same with ppv, sensitivity and f1 scores
    other_scores = ["ppv_scores", "f1_scores", "sensitivity_scores"]

    for score in other_scores:
        score_file = path_to_outputs + f'/{score}.txt'
        scores = {}
        with open(score_file, 'r') as file:
            for line in file:
                key, value = line.strip().split(':')
                scores[key] = float(value)
        metrics_results[score] = metrics_results['name'].apply(lambda x: scores[x])

        # Then we list the scores per vertebral levels same as above
        score_files = list(Path(path_to_outputs).rglob(f"{score}_*.txt"))
        for score_file in score_files:
            level = str(score_file).split("/")[-1].replace('.txt', '')
            level_scores = {}
            with open(score_file, 'r') as file:
                for line in file:
                    key, value = line.strip().split(':')
                    metrics_results.loc[metrics_results['name'] == key, level] = float(value)

    # Sort the order of the lines by contrast (alphabetical order)
    metrics_results = metrics_results.sort_values(by='contrast').reset_index(drop=True)

    # save to csv
    metrics_results.to_csv(path_to_outputs + '/metrics_results.csv', index=False)

    # SC disc dictionnary
    sc_disc_dict = {
        "1": "C1", "2": "C2", "3": "C3", "4": "C4", "5": "C5", "6": "C6", "7": "C7",
        "8": "T1", "9": "T2", "10": "T3", "11": "T4", "12": "T5", "13": "T6", "14": "T7",
        "15": "T8", "16": "T9", "17": "T10", "18": "T11", "19": "T12",
        "20": "L1", "21": "L2", "22": "L3", "23": "L4", "24": "L5",
        "25": "S1"
    }

    # Now we plot the performance per vertebral level
    ####################################################
    ## For dice first
    ####################################################
    data_dice = metrics_results.filter(like='dice').copy()
    data_dice['contrast'] = metrics_results['contrast']
    data_dice['site'] = metrics_results['site']
    data_dice['name'] = metrics_results['name']

    dice_cols = data_dice.columns.tolist()
    dice_cols = [col for col in dice_cols if col.startswith('dice_scores_')]

    # We reorganize the data for plotting
    df_long_dice = data_dice.melt(id_vars=['contrast', 'site', 'name'], value_vars=dice_cols,
        var_name='vertebral_level',value_name='dice'
    )

    # We only keep the first vert level as location:
    df_long_dice['vertebral_level'] = df_long_dice['vertebral_level'].str.replace('dice_scores_', '')
    df_long_dice['level_start'] = (df_long_dice['vertebral_level'].str.extract(r'(\d+)(?=_to_)')[0].astype(int))

    # Mean per level
    mean_dice = df_long_dice.groupby('level_start')['dice'].mean()

    # --- Proportion of present levels ---
    total_images = df_long_dice['name'].nunique()
    present_levels = df_long_dice[df_long_dice['dice'] > 0]
    proportions = present_levels['level_start'].value_counts() / total_images

    # Ensure both series share the same order
    levels = sorted(set(mean_dice.index) | set(proportions.index))
    mean_dice = mean_dice.reindex(levels, fill_value=0)
    proportions = proportions.reindex(levels, fill_value=0)

    # Map levels to disc names using sc_disc_dict, fallback to str(level) if not found
    x_labels = [sc_disc_dict.get(str(l), str(l)) for l in levels]

    # --- Combined plot with all labels on the x-axis ---
    fig, ax1 = plt.subplots(figsize=(12,6))

    # Barres = Dice score (axe gauche)
    bar_positions = [l + 0.5 for l in levels]
    bars = ax1.bar(bar_positions, mean_dice, width=0.9, color="skyblue", alpha=0.7, label="Mean Dice score")
    ax1.set_ylabel("Mean Dice score", color="blue")
    ax1.set_xlabel("Disc level")
    ax1.tick_params(axis='y', labelcolor="blue")

    # Axe secondaire pour proportions
    ax2 = ax1.twinx()
    ax2.plot(bar_positions, proportions, color="red", marker="o", label="Proportion present")
    ax2.set_ylabel("Proportion of present disc levels", color="red")
    ax2.tick_params(axis='y', labelcolor="red")

    # Afficher tous les labels de niveaux sur l'axe x
    ax1.set_xticks(levels)
    ax1.set_xticklabels(x_labels, rotation=45)

    plt.title("Dice score and proportion of present disc levels")
    fig.tight_layout()
    plt.savefig(path_to_outputs + '/dice_scores_per_disc_level.png', dpi=300)

    ####################################################
    ## For sensitivity
    ####################################################
    data_sensitivity = metrics_results.filter(like='sensitivity').copy()
    data_sensitivity['contrast'] = metrics_results['contrast']
    data_sensitivity['site'] = metrics_results['site']
    data_sensitivity['name'] = metrics_results['name']

    sensitivity_cols = data_sensitivity.columns.tolist()
    sensitivity_cols = [col for col in sensitivity_cols if col.startswith('sensitivity_scores_')]

    # We reorganize the data for plotting
    df_long_sens = data_sensitivity.melt(id_vars=['contrast', 'site', 'name'], value_vars=sensitivity_cols,
        var_name='vertebral_level',value_name='sensitivity'
    )

    # We only keep the first vert level as location:
    df_long_sens['vertebral_level'] = df_long_sens['vertebral_level'].str.replace('sensitivity_scores_', '')
    df_long_sens['level_start'] = (df_long_sens['vertebral_level'].str.extract(r'(\d+)(?=_to_)')[0].astype(int))

    # Mean per level
    mean_sens = df_long_sens.groupby('level_start')['sensitivity'].mean()

    # --- Proportion of present levels ---
    total_images = df_long_sens['name'].nunique()
    present_levels = df_long_sens[df_long_sens['sensitivity'] > 0]
    proportions = present_levels['level_start'].value_counts() / total_images

    # Ensure both series share the same order
    levels = sorted(set(mean_sens.index) | set(proportions.index))
    mean_sens = mean_sens.reindex(levels, fill_value=0)
    proportions = proportions.reindex(levels, fill_value=0)

    # Map levels to disc names using sc_disc_dict, fallback to str(level) if not found
    x_labels = [sc_disc_dict.get(str(l), str(l)) for l in levels]

    # --- Combined plot with all labels on the x-axis ---
    fig, ax1 = plt.subplots(figsize=(12,6))

    # Barres = Sensitivity score (axe gauche)
    bar_positions = [l + 0.5 for l in levels]
    bars = ax1.bar(bar_positions, mean_sens, width=0.9, color="skyblue", alpha=0.7, label="Mean Sensitivity score")
    ax1.set_ylabel("Mean Sensitivity score", color="blue")
    ax1.set_xlabel("Disc level")
    ax1.tick_params(axis='y', labelcolor="blue")

    # Axe secondaire pour proportions
    ax2 = ax1.twinx()
    ax2.plot(bar_positions, proportions, color="red", marker="o", label="Proportion present")
    ax2.set_ylabel("Proportion of present disc levels", color="red")
    ax2.tick_params(axis='y', labelcolor="red")

    # Afficher tous les labels de niveaux sur l'axe x
    ax1.set_xticks(levels)
    ax1.set_xticklabels(x_labels, rotation=45)

    plt.title("Sensitivity score and proportion of present disc levels")
    fig.tight_layout()
    plt.savefig(path_to_outputs + '/sensitivity_scores_per_disc_level.png', dpi=300)

    ####################################################
    ## For PPV
    ####################################################
    data_ppv = metrics_results.filter(like='ppv').copy()
    data_ppv['contrast'] = metrics_results['contrast']
    data_ppv['site'] = metrics_results['site']
    data_ppv['name'] = metrics_results['name']

    ppv_cols = data_ppv.columns.tolist()
    ppv_cols = [col for col in ppv_cols if col.startswith('ppv_scores_')]

    # We reorganize the data for plotting
    df_long_ppv = data_ppv.melt(id_vars=['contrast', 'site', 'name'], value_vars=ppv_cols,
        var_name='vertebral_level',value_name='ppv'
    )

    # We only keep the first vert level as location:
    df_long_ppv['vertebral_level'] = df_long_ppv['vertebral_level'].str.replace('ppv_scores_', '')
    df_long_ppv['level_start'] = (df_long_ppv['vertebral_level'].str.extract(r'(\d+)(?=_to_)')[0].astype(int))

    # Mean per level
    mean_ppv = df_long_ppv.groupby('level_start')['ppv'].mean()

    # --- Proportion of present levels ---
    total_images = df_long_ppv['name'].nunique()
    present_levels = df_long_ppv[df_long_ppv['ppv'] > 0]
    proportions = present_levels['level_start'].value_counts() / total_images

    # Ensure both series share the same order
    levels = sorted(set(mean_ppv.index) | set(proportions.index))
    mean_ppv = mean_ppv.reindex(levels, fill_value=0)
    proportions = proportions.reindex(levels, fill_value=0)

    # Map levels to disc names using sc_disc_dict, fallback to str(level) if not found
    x_labels = [sc_disc_dict.get(str(l), str(l)) for l in levels]

    # --- Combined plot with all labels on the x-axis ---
    fig, ax1 = plt.subplots(figsize=(12,6))

    # Barres = PPV score (axe gauche)
    bar_positions = [l + 0.5 for l in levels]
    bars = ax1.bar(bar_positions, mean_ppv, width=0.9, color="skyblue", alpha=0.7, label="Mean PPV score")
    ax1.set_ylabel("Mean PPV score", color="blue")
    ax1.set_xlabel("Disc level")
    ax1.tick_params(axis='y', labelcolor="blue")

    # Axe secondaire pour proportions
    ax2 = ax1.twinx()
    ax2.plot(bar_positions, proportions, color="red", marker="o", label="Proportion present")
    ax2.set_ylabel("Proportion of present disc levels", color="red")
    ax2.tick_params(axis='y', labelcolor="red")

    # Afficher tous les labels de niveaux sur l'axe x
    ax1.set_xticks(levels)
    ax1.set_xticklabels(x_labels, rotation=45)

    plt.title("PPV score and proportion of present disc levels")
    fig.tight_layout()
    plt.savefig(path_to_outputs + '/ppv_scores_per_disc_level.png', dpi=300)

    ####################################################
    ## For F1 score
    ####################################################
    data_f1 = metrics_results.filter(like='f1').copy()
    data_f1['contrast'] = metrics_results['contrast']
    data_f1['site'] = metrics_results['site']
    data_f1['name'] = metrics_results['name']

    f1_cols = data_f1.columns.tolist()
    f1_cols = [col for col in f1_cols if col.startswith('f1_scores_')]

    # We reorganize the data for plotting
    df_long_f1 = data_f1.melt(id_vars=['contrast', 'site', 'name'], value_vars=f1_cols,
        var_name='vertebral_level',value_name='f1'
    )

    # We only keep the first vert level as location:
    df_long_f1['vertebral_level'] = df_long_f1['vertebral_level'].str.replace('f1_scores_', '')
    df_long_f1['level_start'] = (df_long_f1['vertebral_level'].str.extract(r'(\d+)(?=_to_)')[0].astype(int))

    # Mean per level
    mean_f1 = df_long_f1.groupby('level_start')['f1'].mean()

    # --- Proportion of present levels ---
    total_images = df_long_f1['name'].nunique()
    present_levels = df_long_f1[df_long_f1['f1'] > 0]
    proportions = present_levels['level_start'].value_counts() / total_images

    # Ensure both series share the same order
    levels = sorted(set(mean_f1.index) | set(proportions.index))
    mean_f1 = mean_f1.reindex(levels, fill_value=0)
    proportions = proportions.reindex(levels, fill_value=0)

    # Map levels to disc names using sc_disc_dict, fallback to str(level) if not found
    x_labels = [sc_disc_dict.get(str(l), str(l)) for l in levels]

    # --- Combined plot with all labels on the x-axis ---
    fig, ax1 = plt.subplots(figsize=(12,6))

    # Barres = F1 score (axe gauche)
    bar_positions = [l + 0.5 for l in levels]
    bars = ax1.bar(bar_positions, mean_f1, width=0.9, color="skyblue", alpha=0.7, label="Mean F1 score")
    ax1.set_ylabel("Mean F1 score", color="blue")
    ax1.set_xlabel("Disc level")
    ax1.tick_params(axis='y', labelcolor="blue")

    # Axe secondaire pour proportions
    ax2 = ax1.twinx()
    ax2.plot(bar_positions, proportions, color="red", marker="o", label="Proportion present")
    ax2.set_ylabel("Proportion of present disc levels", color="red")
    ax2.tick_params(axis='y', labelcolor="red")

    # Afficher tous les labels de niveaux sur l'axe x
    ax1.set_xticks(levels)
    ax1.set_xticklabels(x_labels, rotation=45)

    plt.title("F1 score and proportion of present disc levels")
    fig.tight_layout()
    plt.savefig(path_to_outputs + '/f1_scores_per_disc_level.png', dpi=300)

    #####################################
    # Save the summary table
    #####################################
    # Now we want to build a table which displays mean and std for all 4 measures per disc levels
    summary_table = pd.DataFrame({
        "Dice score": round(mean_dice,4).astype(str) + " ± " + round(df_long_dice.groupby('level_start')['dice'].std(),4).astype(str),
        "Precision (PPV)": round(mean_ppv,4).astype(str) + " ± " + round(df_long_ppv.groupby('level_start')['ppv'].std(),4).astype(str),
        "Sensitivity": round(mean_sens,4).astype(str) + " ± " + round(df_long_sens.groupby('level_start')['sensitivity'].std(),4).astype(str),
        "F1 Score": round(mean_f1,4).astype(str) + " ± " + round(df_long_f1.groupby('level_start')['f1'].std(),4).astype(str),
        "Proportion of presence": round(proportions,4).astype(str) + " ± " + round(df_long_f1.groupby('level_start')['f1'].std(),4).astype(str)
    })
    # Save the table in csv file
    summary_table.to_csv(os.path.join(path_to_outputs, 'summary_table.csv'), sep='\t')

    return None


if __name__ == "__main__":
    main()