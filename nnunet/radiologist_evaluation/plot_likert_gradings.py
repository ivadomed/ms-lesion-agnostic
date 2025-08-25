"""
This script uploads the qc reports submitted by the radiologists, generates a dataframe and the visualizations for the Likert gradings.

Input:
    --qc-reports: Path to the folder containing the qc reports
    --conversion-dict: path to the conversion dictionary file to find which label correspond to which class (prediction or ground truth)
    --output-dir: Path to the output directory where the visualizations will be saved

Output:
    None

Example:
    python plot_likert_gradings.py --qc-reports /path/to/qc/reports --conversion-dict /path/to/conversion/dict --output-dir /path/to/output/dir

Author: Pierre-Louis Benveniste
"""
import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
from scipy.stats import wilcoxon
import os


def parse_args():
    parser = argparse.ArgumentParser(description='Plot radiologist evaluation')
    parser.add_argument('--qc-reports', type=str, required=True, help='Path to the folder containing the qc reports')
    parser.add_argument('--conversion-dict', type=str, required=True, help='Path to the conversion dictionary file to find which label correspond to which class (prediction or ground truth)')
    parser.add_argument('--output-dir', type=str, required=True, help='Path to the output directory where the visualizations will be saved')
    return parser.parse_args()


def main():
    args = parse_args()

    qc_reports = Path(args.qc_reports)
    conversion_dict = Path(args.conversion_dict)
    output_dir = Path(args.output_dir)

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # List all qc report (i.e. json files)
    qc_report_files = list(qc_reports.glob('*.json'))

    # Initialize the DataFrame to store all report data
    results_template = pd.DataFrame()
    results_template['image']=None

    # Load the conversion dictionary
    with open(conversion_dict, 'r') as f:
        conversion_dict = json.load(f)

    # For each image, initialize the results dataset
    for i, image in enumerate(conversion_dict):
        # Add the image to the df
        results_template.loc[i, 'image'] = image.split('/')[-1]

    # Initialize the colums corresponding to the score
    results_template['manual_score'] = np.nan
    results_template['predicted_score'] = np.nan
    results_template['rater'] = None

    # Combined results
    combined_results = pd.DataFrame()

    # Iterate through each QC report file
    for report_file in qc_report_files:
        # Create a copy of the results template
        results = results_template.copy()
        with open(report_file, 'r') as f:
            report_data = json.load(f)
        # Assign the rater
        rater = str(report_file).split('_')[-1].replace('.json', '')
        results['rater'] = rater
        for j, line in enumerate(report_data['datasets']):
            # Get the image used:
            image = line['cmdline'].split('sct_qc -i ')[1].split(' -s')[0].split('/')[-1]
            # Get the label as well:
            qc_label = line['cmdline'].split('-d ')[1].split(' -qc ')[0].split('/')[-1]
            # Get the score
            score = line['rank']
            # In the conversion dict we find if the label correspond to manual or predicted:
            for elem in conversion_dict:
                if conversion_dict[elem]['image'] == image:
                    if conversion_dict[elem]['label'] == qc_label:
                        # Assign the score to the correct column
                        results.loc[results['image'] == image, 'manual_score'] = score
                    elif conversion_dict[elem]['pred'] == qc_label:
                        results.loc[results['image'] == image, 'predicted_score'] = score
        # Combine the results
        combined_results = pd.concat([combined_results, results], ignore_index=True)

    # Save the combined results
    combined_results.to_csv(os.path.join(output_dir, 'combined_results.csv'), index=False)

    # Convert to long format for easier plotting
    plot_df = combined_results.melt(
        id_vars=['rater'],
        value_vars=['manual_score', 'predicted_score'],
        var_name='score_type',
        value_name='score'
    )

    # Plotting of the results side by side
    plt.figure(figsize=(10, 7))
    plt.ylim(-0.5, 6.5)
    plt.yticks(np.arange(1, 6, 1))
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    # Violin plot avec style similaire à ton exemple
    sns.violinplot(
        data=plot_df,
        x="rater",
        y="score",
        hue="score_type",
        split=True,      # pour fusionner en miroir
        gap=0.1,         # espace entre les deux distributions
        inner="quart"    # quartiles visibles
    )
    plt.title('Manual vs Predicted Scores by Rater')
    plt.xlabel('Rater')
    plt.ylabel('Score')
    plt.legend(title="Score Type", labels=["Manual", "Predicted"])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "violin_plot.png"))
    plt.close()


    # Now we want to print a table which the global average (and std) and then the average (and std) per rater
    print("\n--------Manual segmentation vs Predicted segmentation")
    print(f"Global Average Score: {combined_results['manual_score'].mean():.2f} ± {combined_results['manual_score'].std():.2f} vs {combined_results['predicted_score'].mean():.2f} ± {combined_results['predicted_score'].std():.2f}")
    # Then for the individual raters:
    raters = combined_results['rater'].unique()
    for rater in raters:
        rater_data = combined_results[combined_results['rater'] == rater]
        rater_avg = rater_data[['manual_score', 'predicted_score']].mean()
        rater_std = rater_data[['manual_score', 'predicted_score']].std()
        print(f"  {rater}. : {rater_avg['manual_score']:.2f} ± {rater_std['manual_score']:.2f}   vs   {rater_avg['predicted_score']:.2f} ± {rater_std['predicted_score']:.2f}")

    
    # Statistical tests: 
    # We perform a global test between manual and pred
    stat, p = wilcoxon(combined_results['manual_score'], combined_results['predicted_score'])
    print(f"Global Wilcoxon test statistic = {stat:.2f}, p-value = {p:.2f}")

    # Then for each rater, we compute wilcoxon test between manual and pred
    for rater in raters:
        rater_data = combined_results[combined_results['rater'] == rater]
        stat, p = wilcoxon(rater_data['manual_score'], rater_data['predicted_score'])
        print(f"  {rater}. : Wilcoxon test statistic = {stat:.2f}, p-value = {p:.2f}")

if __name__ == '__main__':
    main()