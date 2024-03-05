"""
This script is used to plot the performance of the model in the different subjects.

Args:
    -perfs : the csv file with the performance
    -output_folder : the folder where the plots will be saved
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_parser():
    """
    This function parses the command line arguments and returns an argparse object.

    Input:
        None

    Returns:
        parser : argparse object
    """
    parser = argparse.ArgumentParser(description='Plot performance of the model')
    parser.add_argument('--perfs', '-p', type=str, required=True, help='xml file with the performance')
    parser.add_argument('--output_folder', '-o', type=str, required=True, help='folder where the plots will be saved')
    args = parser.parse_args()
    return args


def main():
    args = get_parser()
    perfs = args.perfs
    output_folder = args.output_folder

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Read the csv file
    df = pd.read_csv(perfs)
    
    # Plot the performance : Dice
    plt.figure()
    plt.hist(df['Dice'], label='Dice')
    plt.savefig(os.path.join(output_folder, 'dice.png'))
    # print mean, std
    print('Dice mean:', np.mean(df['Dice']))
    print('Dice std:', np.std(df['Dice']))

    # Plot the performance : Sensitivity
    plt.figure()
    plt.hist(df['Sensitivity'], label='Sensitivity')
    plt.savefig(os.path.join(output_folder, 'sensitivity.png'))
    # print mean, std
    print('Sensitivity mean:', np.mean(df['Sensitivity']))
    print('Sensitivity std:', np.std(df['Sensitivity']))

    # Plot the performance : Specificity
    plt.figure()
    plt.hist(df['Specificity'], label='Specificity')
    plt.savefig(os.path.join(output_folder, 'specificity.png'))
    # print mean, std
    print('Specificity mean:', np.mean(df['Specificity']))
    print('Specificity std:', np.std(df['Specificity']))


if __name__ == '__main__':
    main()
