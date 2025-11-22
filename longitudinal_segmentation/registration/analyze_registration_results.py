"""
This script analyzes the registration results of longitudinal MRI scans.
It loads the .csv file and plots mean and std.

Input:
    -i: path to the input csv file
    -o: path to the output folder where plots will be saved

Outputs:
    None

Author: Pierre-Louis Benveniste
"""
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze registration results from a CSV file.")
    parser.add_argument('-i', '--input-csv', type=str, required=True,
                        help='Path to the input CSV file containing registration results.')
    parser.add_argument('-o', '--output-folder', type=str, required=True,
                        help='Path to the output folder where plots will be saved.')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Create output folder if it doesn't exist
    os.makedirs(args.output_folder, exist_ok=True)

    # Build logger
    logger.add(os.path.join(args.output_folder, f'registration_results.log'))
    
    # Load the CSV file
    df = pd.read_csv(args.input_csv)
    logger.info(f"Reading file: {args.input_csv}")
    
    # For each "method" column, plot mean and std of MI and MSE
    methods = df['method'].unique()
    for method in methods:
        logger.info(f"Processing method: {method}")
        logger.info(f"Mean MI for method {method}: {df[df['method'] == method]['mi'].mean()}")
        logger.info(f"Mean MSE for method {method}: {df[df['method'] == method]['mse'].mean()}")
    
    return None


if __name__ == "__main__":
    main()