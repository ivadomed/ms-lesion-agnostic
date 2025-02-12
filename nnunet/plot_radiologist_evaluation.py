"""
This scripts load the tsv file of the evaluation of the neuro-radiologists and computes the scores and the plots. 

Input:
    --i: Path to the evaluation tsv file
    --path-out: Output directory

Returns:
    - None

Example:
    python plot_radiologist_evaluation.py -i /path/to/evaluation.tsv

Author: Pierre-Louis Benveniste
"""
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
from scipy.stats import wilcoxon


def parse_args():
    parser = argparse.ArgumentParser(description='Plot radiologist evaluation')
    parser.add_argument('-i', '--input', type=str, required=True, help='Path to the evaluation tsv file')
    parser.add_argument('--path-out', type=str, required=True, help='Output directory')
    return parser.parse_args()


def main():
    # Parse the arguments
    args = parse_args()
    input_file = args.input
    path_out = args.path_out

    # Build the path to the output directory
    path_out = Path(path_out)
    path_out.mkdir(parents=True, exist_ok=True)

    # Load the evaluation file
    df = pd.read_csv(input_file, sep=',')

    # Remove all lines below 21 for all columns
    df = df.iloc[:21]

    # remove columns 1, 2, 5, 6, 9 and 10
    df = df.drop(df.columns[[1, 2, 5, 6, 9, 10]], axis=1)

    # Rename the columns with the name in line 0
    df.columns = df.iloc[0]
    df = df.drop(df.index[0])
    
    # Split the dataframe in three for each radiologist
    # df1 takes columns images:, GT laurent and pred laurent
    df1 = df.iloc[:, [0, 1, 2]]
    df2 = df.iloc[:, [0, 3, 4]]
    df3 = df.iloc[:, [0, 5, 6]]

    # Now we concat the three dataframes so that all GT and pred are in the same column but we add a column rater with value Rater 1, Rater 2 or Rater 3
    df1['rater'] = 'Rater 1'
    df2['rater'] = 'Rater 2'
    df3['rater'] = 'Rater 3'

    # Rename the columns
    df1.columns = ['image', 'GT', 'pred', 'rater']
    df2.columns = ['image', 'GT', 'pred', 'rater']
    df3.columns = ['image', 'GT', 'pred', 'rater']

    # Concatenate the three dataframes
    df = pd.concat([df1, df2, df3])
    # reset the index
    df = df.reset_index(drop=True)

    # Change the dataset so that there is only one column for scores and the value GT or pred is stored in a column called type
    df = pd.melt(df, id_vars=['image', 'rater'], value_vars=['GT', 'pred'], var_name='type', value_name='score')
    df = df.reset_index(drop=True)

    # Rename column type to Segmentations and values GT to Manual and pred to Predicted
    df['type'] = df['type'].replace({'GT': 'Manual', 'pred': 'Auto'})
    df = df.rename(columns={'type': 'Segmentations'})
    # Convert score to int
    df['score'] = df['score'].astype(int)

    # color palette is green for manual and yellow for predicted
    sns.set_palette(sns.color_palette(['lime', 'yellow']))
    # Text size should be 15
    sns.set_context("talk")

    # Now we plot a violin plot of the scores
    plt.figure(figsize=(10, 7))
    # y axis goes from 1 to 5
    plt.ylim(-0.5, 6.5)
    # But don't display the 0 and the 6
    plt.yticks(np.arange(1, 6, 1))
    # Add y grid
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    sns.violinplot(data=df, x="rater", y="score", hue="Segmentations", split=True, gap=.1, inner="quart")
    # Don't display the x legend
    plt.xlabel('')
    # Display mean and median in the graph
    plt.ylabel('Likert scores')
    # For rater 1 auto, display the mean with a red line but only on the part of the graph which is regards
    plt.axhline(y=df[(df['rater'] == 'Rater 1') & (df['Segmentations'] == 'Auto')]['score'].mean(), color='red', linestyle='--', alpha=1, xmin=0.175, xmax=0.265)
    # For rater 1 manual, display the mean with a red line but only on the part of the graph which is regards
    plt.axhline(y=df[(df['rater'] == 'Rater 1') & (df['Segmentations'] == 'Manual')]['score'].mean(), color='red', linestyle='--', alpha=1, xmin=0.07, xmax=0.16)
    # For rater 2 auto, display the mean with a red line but only on the part of the graph which is regards
    plt.axhline(y=df[(df['rater'] == 'Rater 2') & (df['Segmentations'] == 'Auto')]['score'].mean(), color='red', linestyle='--', alpha=1, xmin=0.508, xmax=0.57)
    # For rater 2 manual, display the mean with a red line but only on the part of the graph which is regards
    plt.axhline(y=df[(df['rater'] == 'Rater 2') & (df['Segmentations'] == 'Manual')]['score'].mean(), color='red', linestyle='--', alpha=1, xmin=0.418, xmax=0.49)
    # For rater 3 auto, display the mean with a red line but only on the part of the graph which is regards
    plt.axhline(y=df[(df['rater'] == 'Rater 3') & (df['Segmentations'] == 'Auto')]['score'].mean(), color='red', linestyle='--', alpha=1, xmin=0.843, xmax=0.91)
    # For rater 3 manual, display the mean with a red line but only on the part of the graph which is regards
    plt.axhline(y=df[(df['rater'] == 'Rater 3') & (df['Segmentations'] == 'Manual')]['score'].mean(), color='red', linestyle='--', alpha=1, xmin=0.74, xmax=0.82)

    plt.savefig(path_out / 'GT_pred_scores.png')

    # Now for each rater we compute wilcoxon test between GT and pred
    ## For rater 1
    rater_1 = df[df['rater'] == 'Rater 1']
    rater_1['score'] = rater_1['score'].astype(int)
    diff1 = rater_1[rater_1['Segmentations'] == 'Manual']['score'].values - rater_1[rater_1['Segmentations'] == 'Auto']['score'].values
    stat1, p1 = wilcoxon(diff1)
    ## For rater 2
    rater_2 = df[df['rater'] == 'Rater 2']
    rater_2['score'] = rater_2['score'].astype(int)
    diff2 = rater_2[rater_2['Segmentations'] == 'Manual']['score'].values - rater_2[rater_2['Segmentations'] == 'Auto']['score'].values
    stat2, p2 = wilcoxon(diff2)
    ## For rater 3
    rater_3 = df[df['rater'] == 'Rater 3']
    rater_3['score'] = rater_3['score'].astype(int)
    diff3 = rater_3[rater_3['Segmentations'] == 'Manual']['score'].values - rater_3[rater_3['Segmentations'] == 'Auto']['score'].values
    stat3, p3 = wilcoxon(diff3)
    ## Print the results
    print(f'Rater 1: Statistics={stat1}, p={p1}')
    print(f'Rater 2: Statistics={stat2}, p={p2}')
    print(f'Rater 3: Statistics={stat3}, p={p3}')



    



    return None


if __name__ == '__main__':
    main()