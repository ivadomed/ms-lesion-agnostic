"""
This file performs statistical tests for the SCT methods to see if the model performances are statistically different.

Input: 
    --path-perfs: Path to the directory containing the performance files for each model.

Output:
    None

Example:
    python stat_test_sct_methods.py --path-perfs /path/to/perfs --path-msd /path/to/msd

Author: Pierre-Louis Benveniste
"""

import argparse
from pathlib import Path
import pandas as pd
import itertools
from scipy import stats
from statsmodels.sandbox.stats.multicomp import multipletests


def parse_args():
    parser = argparse.ArgumentParser(description="Statistical tests for SCT methods")
    parser.add_argument("--path-perfs", type=str, required=True, help="Path to the directory containing the performance files for each model.")
    return parser.parse_args()


def main():

    # Parse command line arguments
    args = parse_args()
    path_perfs = args.path_perfs

    # Metrics dict:
    metrics = ["dice", "f1", "ppv", "sensitivity"]

    for metric in metrics:

        print("---------------")
        print(f" METRIC = {metric}")
        print("---------------")

        # Initialize the results dataframe
        result_df = pd.DataFrame()

        # we load all dice score in a file
        dice_score_files = list(Path(path_perfs).rglob(f"{metric}_scores.txt"))
        dice_score_files = [str(f) for f in dice_score_files]

        # We iterate over each result file and add the results to the DF
        for result_file in dice_score_files:
            model = result_file.split('/')[-2]
            with open(result_file, 'r') as f:
                for line in f:
                    key, value = line.strip().split(':')
                    result_df.loc[key, model] = float(value)

        # Now we perform statistical tests
        columns = list(result_df.columns)
        combinations = list(itertools.combinations(columns, 2))
        # Run Wilcoxon (groups are dependent, same subjects)
        pvals = []
        for i in range(len(combinations)):
            stat, pval = stats.wilcoxon(x=result_df[combinations[i][0]], y=result_df[combinations[i][1]])
            pvals.append(pval)
        p_adjusted = multipletests(pvals, method='bonferroni', alpha=0.01)
        print("\n")
        
        # Now we plot the statistical results
        for i, combination in enumerate(combinations):
            if 'our_model' in combination[0] or 'our_model' in combination[1]:
                model1 = combination[0]
                model2 = combination[1]
                pval = p_adjusted[1][i]
                print(f"Comparison between {model1} and {model2}: p-value = {pval}")
                if p_adjusted[0][i]:
                    print(f" -> Statistically significant difference \n")


if __name__ == "__main__":
    main()