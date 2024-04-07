"""
Script for generating Precision-Recall curve and PR-AUC

First, run yolo inference with a low confidence threshold (LOWER_CONF), 
then give those predictions as --preds
"""

import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path
import subprocess
import tempfile
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LOWER_CONF = 0.01
UPPER_CONF = 0.5

def _main():
    parser = ArgumentParser(
    prog = 'PR_curve',
    description = 'Generate PR curve and AUC-PR for yolo model',
    formatter_class = ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--gt-path',
                        required= True,
                        type = str,
                        help = 'Path to YOLO dataset folder of ground truth txt files')
    parser.add_argument('-p', '--preds',
                        required = True,
                        type = Path,
                        help = 'Path to prediction folder containing txt files with confidence values.')
    parser.add_argument('-c', '--canproco',
                        required= True,
                        type = str,
                        help = 'Path to canproco database')
    parser.add_argument('-o', '--output',
                        required = True,
                        type = Path,
                        help = 'Output directory to save the PR curve to.')
    parser.add_argument('-i', '--iou',
                        default= 0.4,
                        type = str,
                        help = 'IoU threshold for a TP')
   
    args = parser.parse_args()

    # Create output folder if it doesn't exist
    os.makedirs(args.output, exist_ok=True)

    recalls = []
    precisions = []
    for conf in np.arange(LOWER_CONF, UPPER_CONF, 0.01):
        print(f"\n\nComputing metrics for {conf} conf")
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir)/"preds").mkdir(parents=True, exist_ok=True)
            (Path(tmpdir)/"val").mkdir(parents=True, exist_ok=True)

            # 1. Create new txt files with only boxes that have confidence higher than conf
                # load predictions
            txt_names = os.listdir(args.preds)
            txt_paths = [os.path.join(args.preds, file) for file in txt_names if file.endswith(".txt")] # only keep txts

            print("Copying over txt files")
            for txt_path in txt_paths:
                # For every file create copy but only keeping boxes with confidence higher than conf
                with open(txt_path, "r") as infile:
                    # Read lines from the input file
                    lines = infile.readlines()

                filtered_lines = [line for line in lines if float(line.split()[-1]) > conf]

                if filtered_lines:
                    # only create file if there are boxes
                    filename = Path(txt_path).name
                    with open(Path(tmpdir)/"preds"/filename, "w") as outfile:
                        outfile.writelines(filtered_lines)

            # 2. Call validation and get recall and precision
            print("Calling validation")
            command = ["python",
                        "validation.py",
                        "-g", args.gt_path,
                        "-p", str(Path(tmpdir)/"preds"),
                        "-o", str(Path(tmpdir)/"val"),
                        "-c", args.canproco,
                        "-i", args.iou]
            subprocess.run(command, check=True)

            # 3. Get recall and precision and add to dict
            print("Getting recall and precision")
            df = pd.read_csv(Path(tmpdir)/"val"/"metrics_report.csv")

                # Extract Recall and Precision from the last row
            recalls.append(df.iloc[-1]['Recall'])
            precisions.append(df.iloc[-1]['Precision'])

    # Plot 
    plt.plot(recalls, precisions, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve with {args.iou} iou threshold')
    plt.savefig(args.output/f'precision_recall_curve_{args.iou}iou.png')

    # Calculate PR-AUC
    auc_pr = np.trapz(precisions[::-1], recalls[::-1])
    print('Area under Precision-Recall curve (AUC-PR):', auc_pr)


if __name__ == "__main__":
    _main()