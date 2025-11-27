"""
This file runs the lesion mapping evaluation script and plots the results.
It is designed to be run only on the Toronto test set.

Input:
    - input_msd_dataset: Path to the MSD dataset folder.
    - predictions_folder: Path to the folder containing model predictions.
    - output_folder: Path to the folder where evaluation results will be saved.
    - sub-edss: Path to a csv file containing subject EDSS scores.
Output:
    None

Author: Pierre-Louis Benveniste
"""
import json
from evaluate_lesion_mapping import evaluate_lesion_mapping
import os
import matplotlib.pyplot as plt
from loguru import logger
import json
import pandas as pd
import numpy as np
from prettytable import PrettyTable
from scipy import stats


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-msd', type=str, required=True, help='Path to the input MSD dataset')
    parser.add_argument('-p', '--predictions-folder', type=str, required=True, help='Path to the folder containing model predictions')
    parser.add_argument('-o', '--output-folder', type=str, required=True, help='Path to the output folder where evaluation results will be stored')
    parser.add_argument('-s', '--sub-edss', type=str, required=True, help='Path to a csv file containing subject EDSS scores')
    return parser.parse_args()


def plot_evaluation_results(results, output_folder, sub_edss):
    """
    This function plots the evaluation results.

    Input:
        - results: Dictionary containing evaluation results.
        - output_folder: Path to the folder where plots will be saved.
     Output:
        None
    """
    # Initialize a logger
    logger.add(os.path.join(output_folder, 'plot_evaluation.log'))
    # Build a pandas dataframe from the results
    df = pd.DataFrame(results).T

    # Load the subjects EDSS scores
    edss_df = pd.read_csv(sub_edss)
    # Remove some columns in edss_df
    edss_df = edss_df.drop(columns=['file', 'number', 'site', 'name', 'original_name' ])

    # Merge the two dataframes on subject ID
    df = df.merge(edss_df, left_index=True, right_on='sub')

    # We remove subjects that are not from Toronto site
    df = df[df['sub'].str.contains('tor')]

    # For values in EDSS columns, we convert them to float
    for col in edss_df.columns:
        if col != 'sub' and col != 'phenotye':
            # Replace , by . in the column
            df[col] = df[col].str.replace(',', '.')
            # Convert the column to float
            df[col] = df[col].astype(float)

    # Convert column lesion_mapping_TP, lesion_mapping_FP, lesion_mapping_FN to their average values and std and sum and count
    for columns in ['lesion_mapping_TP', 'lesion_mapping_FP', 'lesion_mapping_FN']:
        df[columns + '_mean'] = df[columns].apply(lambda x: np.mean(x))
        df[columns + '_std'] = df[columns].apply(lambda x: np.std(x))
        df[columns + '_sum'] = df[columns].apply(lambda x: np.sum(x))
        df[columns + '_count'] = df[columns].apply(lambda x: len(x))
    # For all columns, we print the mean and std and the median and range
    for column in df.columns:
        if column in ['lesion_mapping_TP', 'lesion_mapping_FP', 'lesion_mapping_FN', 'sub', 'phenotye']:
            continue
        logger.info(f" ----- Plotting column {column} ----- ")
        logger.info(f"Mean: {df[column].mean()}, Std: {df[column].std()}")
        logger.info(f"Median: {df[column].median()}, Range: {df[column].min()} - {df[column].max()}")
    
    # Now we create a table with all the results
    table = PrettyTable()
    table.field_names = ["Metric", "M0", "M12"]
    segmentation_columns = ['dice', 'ppv', 'f1', 'sensitivity', 'TP', 'FP', 'FN', 'count_lesion_GT', 'count_lesion_pred']
    for col in segmentation_columns:
        if "f1" in col:
            metric_M0 = f"{col}_1"
            metric_M12 = f"{col}_2"
        else:
            metric_M0 = f"{col}1"
            metric_M12 = f"{col}2"
        table.add_row([col.upper(), f"{df[metric_M0].mean():.4f} ± {df[metric_M0].std():.4f}", f"{df[metric_M12].mean():.4f} ± {df[metric_M12].std():.4f}"])
        table.add_row([f" ", f"{df[metric_M0].median():.4f} ({df[metric_M0].min():.4f} ; {df[metric_M0].max():.4f})", f"{df[metric_M12].median():.4f} ({df[metric_M12].min():.4f} ; {df[metric_M12].max():.4f})"])
    logger.info("\n" + table.get_string())

    # Now we compare lesion volumes between GT and predicted at both timepoints
    table_vol = PrettyTable()
    table_vol.field_names = ["Metric", "M0 (mm3)", "M12 (mm3)"]
    table_vol.add_row([f"GT Volume", f"{df['vol_gt1_mm3'].mean():.2f} ± {df['vol_gt1_mm3'].std():.2f}", f"{df['vol_gt2_mm3'].mean():.2f} ± {df['vol_gt2_mm3'].std():.2f}"])
    table_vol.add_row([f"Predicted Volume", f"{df['vol_pred1_mm3'].mean():.2f} ± {df['vol_pred1_mm3'].std():.2f}", f"{df['vol_pred2_mm3'].mean():.2f} ± {df['vol_pred2_mm3'].std():.2f}"])
    logger.info("\n" + table_vol.get_string())

    # Now we want to look at the count of lesions at timepoint 2 when there are different counts at timepoint 1 compared to GT 1
    ## Get subjects where count_lesion_GT1 != count_lesion_pred1
    diff_count_subjects = df[df['count_lesion_GT1'] != df['count_lesion_pred1']]
    logger.info(f"Number of subjects with different lesion counts at timepoint 1: {len(diff_count_subjects)}")
    table_diff = PrettyTable()
    table_diff.field_names = ["Metric", "M0", "M12"]
    table_diff.add_row([f"Count Lesion GT", f"{diff_count_subjects['count_lesion_GT1'].mean():.2f} ± {diff_count_subjects['count_lesion_GT1'].std():.2f}", f"{diff_count_subjects['count_lesion_GT2'].mean():.2f} ± {diff_count_subjects['count_lesion_GT2'].std():.2f}"])
    table_diff.add_row([f"Count Lesion Pred", f"{diff_count_subjects['count_lesion_pred1'].mean():.2f} ± {diff_count_subjects['count_lesion_pred1'].std():.2f}", f"{diff_count_subjects['count_lesion_pred2'].mean():.2f} ± {diff_count_subjects['count_lesion_pred2'].std():.2f}"])
    logger.info("\n" + table_diff.get_string())
    
    # Now we look at the lesion mapping metrics
    table_mapping = PrettyTable()
    table_mapping.field_names = ["Metric", "Sum over all subjects", "Mean per subject ± std", "Average per lesion ± std"]
    table_mapping.add_row([f"TP", f"{df['lesion_mapping_TP_sum'].sum()}", f"{df['lesion_mapping_TP_sum'].mean():.2f} ± {df['lesion_mapping_TP_sum'].std():.2f}", f"{df['lesion_mapping_TP_mean'].mean():.2f} ± {df['lesion_mapping_TP_mean'].std():.2f}"])
    table_mapping.add_row([f"FP", f"{df['lesion_mapping_FP_sum'].sum()}", f"{df['lesion_mapping_FP_sum'].mean():.2f} ± {df['lesion_mapping_FP_sum'].std():.2f}", f"{df['lesion_mapping_FP_mean'].mean():.2f} ± {df['lesion_mapping_FP_mean'].std():.2f}"])
    table_mapping.add_row([f"FN", f"{df['lesion_mapping_FN_sum'].sum()}", f"{df['lesion_mapping_FN_sum'].mean():.2f} ± {df['lesion_mapping_FN_sum'].std():.2f}", f"{df['lesion_mapping_FN_mean'].mean():.2f} ± {df['lesion_mapping_FN_mean'].std():.2f}"])
    table_mapping.add_row([f"Precision", f"{df['lesion_mapping_TP_sum'].sum() / (df['lesion_mapping_TP_sum'].sum() + df['lesion_mapping_FP_sum'].sum()):.4f}", "", ""])
    table_mapping.add_row([f"Recall", f"{df['lesion_mapping_TP_sum'].sum() / (df['lesion_mapping_TP_sum'].sum() + df['lesion_mapping_FN_sum'].sum()):.4f}", "", ""])
    logger.info("\n" + table_mapping.get_string())

    # Finally for each subject with non-0 EDSS_diff, we look at the correlation between EDSS diff and volume change
    df_nonzero_edss = df[df['EDSS diff'] != 0]
    # We compute the difference in lesion volume GT and predicted
    df_nonzero_edss['vol_gt_diff_mm3'] = df_nonzero_edss['vol_gt2_mm3'] - df_nonzero_edss['vol_gt1_mm3']
    df_nonzero_edss['vol_pred_diff_mm3'] = df_nonzero_edss['vol_pred2_mm3'] - df_nonzero_edss['vol_pred1_mm3']
    
    # We want to compute the correlation coefficient on ground truth volumes
    # Plot the scatter plot between vol_gt_diff_mm3 and EDSS diff
    plt.figure()
    plt.scatter(df_nonzero_edss['vol_gt_diff_mm3'], df_nonzero_edss['EDSS diff'])
    plt.xlabel('Lesion Volume Change (mm3)')
    plt.ylabel('EDSS Change')
    plt.title('Lesion Volume Change vs EDSS Change')
    plt.grid()
    plt.savefig(os.path.join(output_folder, 'gt_lesion_volume_change_vs_EDSS_change.png'))
    
    res = stats.spearmanr(list(df_nonzero_edss['vol_gt_diff_mm3']), list(df_nonzero_edss['EDSS diff']))
    logger.info(f"Spearman correlation coefficient (GT volume): {res}")
    res = stats.pearsonr(list(df_nonzero_edss['vol_gt_diff_mm3']), list(df_nonzero_edss['EDSS diff']))
    logger.info(f"Pearson correlation coefficient (GT volume): {res}")
    # We also want to compute the correlation coefficient on predicted volumes
    ## Plot the scatter plot between vol_pred_diff_mm3 and EDSS diff
    plt.figure()
    plt.scatter(df_nonzero_edss['vol_pred_diff_mm3'], df_nonzero_edss['EDSS diff'])
    plt.xlabel('Predicted Lesion Volume Change (mm3)')
    plt.ylabel('EDSS Change')
    plt.title('Predicted Lesion Volume Change vs EDSS Change')
    plt.grid()
    plt.savefig(os.path.join(output_folder, 'predicted_lesion_volume_change_vs_EDSS_change.png'))
    res = stats.spearmanr(list(df_nonzero_edss['vol_pred_diff_mm3']), list(df_nonzero_edss['EDSS diff']))
    logger.info(f"Spearman correlation coefficient (predicted volume): {res}")
    res = stats.pearsonr(list(df_nonzero_edss['vol_pred_diff_mm3']), list(df_nonzero_edss['EDSS diff']))
    logger.info(f"Pearson correlation coefficient (predicted volume): {res}")

    return None


if __name__ == "__main__":
    args = parse_args()
    input_msd_dataset = args.input_msd
    predictions_folder = args.predictions_folder
    output_folder = args.output_folder
    sub_edss = args.sub_edss

    # Run the evaluation
    results = evaluate_lesion_mapping(input_msd_dataset, predictions_folder, output_folder)

    # results_path = "~/net/longitudinal_ms/20251121_lesion_matching_reg_com_results_debug/overall_results.json"
    # with open(os.path.expanduser(results_path), 'r') as f:
    #     results = json.load(f)

    plot_evaluation_results(results, output_folder, sub_edss)
