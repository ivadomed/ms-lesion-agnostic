"""
Evaluates the lesion segmentations predicted by the SCT model against the manual lesion segmentations,
for each scan (run-01 and run-02) of study_files.json (output of generate_study_files.py) which has a
manual lesion segmentation. Predictions are computed (if they do not exist yet) with:
    SCT_USE_GPU=1 sct_deepseg lesion_ms -i <image> -o <prediction> [flags of the mode]

Modes:
    default         no flag
    tta             -test-time-aug
    single-fold     -single-fold
    soft-bin        -soft-seg, binarized at 0.5

Metrics (same as evaluation/evaluate_predictions.py): Dice, lesion-wise PPV, F1 score and sensitivity.

Outputs (in the output folder):
    - model_vs_manual.csv: one line per scan (including the has_lesion field)
    - model_vs_manual.png: boxplots of the metrics, labeled by focal lesion visible or uncertain

Arguments:
    -i / --input        Path to the output folder of generate_study_files.py
    -o / --output       Path to the output folder
    -m / --mode         Inference mode (default: default)

Author: Pierre-Louis Benveniste
"""

import argparse
import json
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import nibabel as nib
import pandas as pd
import seaborn as sns
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent / "evaluation"))
from utils import dice_score, lesion_ppv, lesion_f1_score, lesion_sensitivity

MODES = {"default": "", "tta": "-test-time-aug", "single-fold": "-single-fold", "soft-bin": "-soft-seg"}
METRICS = ["dice", "ppv", "f1", "sensitivity"]
PALETTE = {"visible": "tab:orange", "uncertain": "tab:blue"}


def load_seg(path):
    """Loads a segmentation binarized at 0.5."""
    return (nib.load(path).get_fdata() > 0.5).astype(float)


def process_scan(pair, run, dataset, study_dir, mode):
    image = dataset / pair["scan_run-01" if run == "01" else "rescan_run-02"]
    flags = MODES[mode]

    # Inference (shared with evaluate_model_lesion_seg_variability.py)
    pred = (study_dir / pair[f"sc_seg_run-{run}"]).parent / image.name.replace(".nii.gz", f"_label-lesion_desc-{flags.lstrip('-') or 'default'}_seg.nii.gz")
    if not pred.exists():
        assert os.system(f"SCT_USE_GPU=1 sct_deepseg lesion_ms -i {image} -o {pred} {flags}") == 0, f"Inference failed for {image}"

    pred_data, label_data = load_seg(pred), load_seg(dataset / pair[f"lesion_seg_run-{run}"])
    return {
        "subject": pair["subject"],
        "session": pair["session"],
        "acquisition": pair["acquisition"],
        "run": run,
        "has_lesion": pair["has_lesion"],
        "dice": dice_score(pred_data, label_data),
        "ppv": lesion_ppv(label_data, pred_data),
        "f1": lesion_f1_score(label_data, pred_data),
        "sensitivity": lesion_sensitivity(label_data, pred_data),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate the model lesion segmentations against the manual lesion segmentations.")
    parser.add_argument("-i", "--input", required=True, type=Path, help="Path to the output folder of generate_study_files.py")
    parser.add_argument("-o", "--output", required=True, type=Path, help="Path to the output folder")
    parser.add_argument("-m", "--mode", default="default", choices=MODES, help="Inference mode")
    args = parser.parse_args()

    study_dir = args.input.resolve()
    with open(study_dir / "study_files.json") as f:
        study = json.load(f)

    # All the scans with a manual lesion segmentation
    scans = [(p, r) for p in study["pairs"] if p["status"] == "success" for r in ("01", "02") if p[f"lesion_seg_run-{r}"]]
    results = []
    for pair, run in tqdm(scans):
        try:
            results.append(process_scan(pair, run, Path(study["dataset"]), study_dir, args.mode))
        except Exception as e:
            print(f"Failed for {pair['subject']}/{pair['session']}/acq-{pair['acquisition']}/run-{run}: {e}")

    args.output.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    df["Focal lesion"] = df["has_lesion"].astype(str).str.strip().str.lower().map({"true": "visible", "false": "uncertain"})
    df.to_csv(args.output / "model_vs_manual.csv", index=False)
    print(df.groupby("Focal lesion")[METRICS].agg(["mean", "std"]).round(3))

    # Boxplots of the metrics, labeled by focal lesion
    hue = "Focal lesion"
    fig, axes = plt.subplots(1, len(METRICS), figsize=(5 * len(METRICS), 5))
    for ax, metric in zip(axes, METRICS):
        sns.boxplot(data=df, x=hue, y=metric, hue=hue, palette=PALETTE, legend=False, ax=ax, showfliers=False)
        sns.stripplot(data=df, x=hue, y=metric, ax=ax, color="k")
        ax.set(title=metric, ylim=(-0.05, 1.05))
    fig.tight_layout()
    fig.savefig(args.output / "model_vs_manual.png", dpi=200)
    print(f"Results of {len(results)} scans saved to {args.output}")


if __name__ == "__main__":
    main()
