"""
This file runs the registration on the MS data using multiple methods.
It uses the register_multiple_methods function defined in register_multiple_methods.py
and saves the results in a CSV file.

Inputs:
    -i: path to the MSD dataset (json file)
    -o: path to the output folder where registration results will be stored

Outputs:
    - A CSV file containing the registration results for each method.

Usage:
    python run_registration_on_msd_data.py -i <path_to_msd_json> -o <output_folder>

Author: Pierre-Louis Benveniste
"""
import os
import argparse
from register_multiple_methods import register_multiple_methods
import json
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True, help='Path to the MSD dataset (json file)')
    parser.add_argument('-o', '--output', type=str, required=True, help='Path to the output folder where registration results will be stored')
    return parser.parse_args()


def run_on_all_data(input_json, output_folder):
    """
    This function runs the registration on all data in the MSD dataset.

    Inputs:
        input_json : path to the MSD dataset (json file)
        output_folder : path to the output folder where registration results will be stored
    Outputs:
        results : A dictionary containing the registration results for each subject.
    """
    # Load the MSD dataset
    with open(input_json, 'r') as f:
        msd_data = json.load(f)
    msd_data = msd_data['data']

    # Initialize output folder
    os.makedirs(output_folder, exist_ok=True)
    # Initialize QC folder
    qc_folder = os.path.join(output_folder, "qc")
    os.makedirs(qc_folder, exist_ok=True)
    # Initialize a results dictionary
    results = {}

    # Iterate over all subjects and timepoints
    for subject in tqdm(msd_data):
        if subject == "sub-van072":
            # Skip this subject as the registration fails
            continue
        # Create a subject-specific output folder
        subject_output_folder = os.path.join(output_folder, subject)
        os.makedirs(subject_output_folder, exist_ok=True)
        # For canproco, image1 is ses-M0, image2 is ses-M12
        image1_path = msd_data[subject]['ses-M0'][0]
        image2_path = msd_data[subject]['ses-M12'][0]
        # Run registration using multiple methods
        scores = register_multiple_methods(image1_path, image2_path, subject_output_folder, qc_folder)
        # Store results
        results[subject] = scores
        
    # Format thee results to a pandas DataFrame
    all_results = []
    for subject, scores in results.items():
        for score in scores:
            mse, mi, registered_file, method = score
            all_results.append({
                'subject': subject,
                'method': method,
                'mse': mse,
                'mi': mi,
                'registered_file': registered_file
            })
    results_df = pd.DataFrame(all_results)

    return results_df


def plot_results(results, output_folder):
    """
    This function plots the registration results.

    Inputs:
        results : A pandas DataFrame containing the registration results for each subject.
        output_folder : path to the output folder where plots will be stored

    Outputs:
        None
    """
    # Build output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Plot MSE results (using a violin plot) by grouping by method
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='method', y='mse', data=results)
    plt.title('Registration Mean Squared Error by Method')
    plt.xlabel('Registration Method')
    plt.ylabel('Mean Squared Error')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'registration_mse_by_method.png'))
    plt.close()

    # Plot MI results (using a violin plot) by grouping by method
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='method', y='mi', data=results)
    plt.title('Registration Mutual Information by Method')
    plt.xlabel('Registration Method')
    plt.ylabel('Mutual Information')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'registration_mi_by_method.png'))
    plt.close()

    return None


def main():
    args = parse_args()
    input_json = args.input
    output_folder = args.output
    # Run registration on all data
    results = run_on_all_data(input_json, output_folder)
    # Save the results to a CSV file
    results.to_csv(os.path.join(output_folder, 'registration_results.csv'), index=False)
    # Plot the results
    plot_results(results, output_folder)

    return None


if __name__ == "__main__":
    main()