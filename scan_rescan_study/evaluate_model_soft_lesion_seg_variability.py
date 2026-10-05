"""
Evaluates the scan-rescan variability of the soft lesion segmentations predicted by the SCT model, for
each pair of study_files.json (output of generate_study_files.py), with:
    SCT_USE_GPU=1 sct_deepseg lesion_ms -i <image> -o <prediction> -average-per-fold

Modes:
    soft            one inference per image
    soft-tta        test-time augmentation: the inference is run on 5 versions of the image (copy, affine,
                    bias field, contrast, noise) and the 5 soft predictions are averaged. The prediction
                    of the affine version is brought back to the native space before averaging.

Metrics (computed on the soft segmentations):
    - soft lesion volume of each run
    - lesion volume difference (mm3 and %) between run-01 and run-02, and between run-01 and run-02
      registered to run-01 (using the warping field of generate_study_files.py)
    - soft Dice between the run-01 prediction and the run-02 prediction registered to run-01
    - soft Dice of each prediction relative to the manual segmentation when available

Outputs (in the output folder), with pairs labeled by focal lesion visible or uncertain (has_lesion field):
    - lesion_seg_variability.csv: one line per pair
    - lesion_seg_variability_native.png: lesion volumes and Bland-Altman plots (mm3 and %), run-01 vs run-02
    - lesion_seg_variability_registered.png: same, run-01 vs run-02 registered to run-01, and Dice
    - lesion_seg_variability_<native/registered>_visible.png: same plots, only for the pairs with
      visible focal lesions
    - lesion_seg_variability_thresholds.csv and .png: same metrics on the soft segmentations without
      threshold and binarized at each threshold (0.1, 0.2, ..., 1.0): one line per pair and threshold,
      and boxplots per threshold

Arguments:
    -i / --input        Path to the output folder of generate_study_files.py
    -o / --output       Path to the output folder
    -m / --mode         Inference mode: soft or soft-tta (default: soft)
    --exclude-stitch    Exclude the stitch acquisitions (included by default)

Author: Pierre-Louis Benveniste
"""

import argparse
import json
import os
import tempfile
from pathlib import Path
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

VOL_1 = "lesion_volume_run-01_mm3"
DICE = "dice_run-02_registered_to_run-01"
PALETTE = {"visible": "tab:orange", "uncertain": "tab:blue"}
LABELS = {"true": "visible", "false": "uncertain"}
THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def run(cmd):
    assert os.system(cmd) == 0, f"Command failed: {cmd}"


def predict(image, pred):
    """Predicts the soft lesion segmentation of an image."""
    run(f"SCT_USE_GPU=1 sct_deepseg lesion_ms -i {image} -o {pred} -average-per-fold")


def predict_tta(image, pred, seed=0):
    """Averages the soft predictions of 5 augmented versions of an image."""
    import torch
    from monai.data import MetaTensor
    from monai.transforms import Identityd, RandAdjustContrastd, RandAffined, RandBiasFieldd, RandGaussianNoised

    nii = nib.load(image)
    header = nii.header.copy()
    header.set_data_dtype(np.float32)
    data = {"image": MetaTensor(torch.from_numpy(nii.get_fdata(dtype=np.float32)[None]))}
    zooms = np.array(nii.header.get_zooms()[:3])
    thick = zooms > 1.5 * zooms.min()  # thick-slice axes: rotation only around them, translation and scaling only along the others
    versions = {
        "copy": Identityd(keys="image"),
        "affine": RandAffined(keys="image", prob=1, rotate_range=0.05 * thick, translate_range=2 * ~thick, scale_range=0.05 * ~thick, mode="bilinear"),
        "bias": RandBiasFieldd(keys="image", prob=1, coeff_range=(-0.02, 0.02)),
        "contrast": RandAdjustContrastd(keys="image", prob=1, gamma=(0.9, 1.1)),
        "noise": RandGaussianNoised(keys="image", prob=1, std=0.05 * float(data["image"].std())),
    }

    preds = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        for name, transform in versions.items():
            if name != "copy":
                transform.set_random_state(seed=seed)
            augmented = transform({"image": data["image"].clone()})["image"]
            augmented_path, augmented_pred = Path(tmp_dir) / f"{name}.nii.gz", Path(tmp_dir) / f"{name}_pred.nii.gz"
            nib.save(nib.Nifti1Image(augmented[0].numpy(), nii.affine, header), augmented_path)
            predict(augmented_path, augmented_pred)
            pred_data = nib.load(augmented_pred).get_fdata(dtype=np.float32)
            if name == "affine":
                # Bring the prediction back to the native space with the inverse of the affine transform
                pred_data = MetaTensor(torch.from_numpy(pred_data[None]), applied_operations=augmented.applied_operations)
                pred_data = transform.inverse({"image": pred_data})["image"][0].numpy()
            preds.append(pred_data)
    nib.save(nib.Nifti1Image(np.mean(preds, axis=0), nii.affine, header), pred)


def load_seg(path):
    """Loads a (soft) segmentation and its voxel volume."""
    img = nib.load(path)
    return np.asarray(img.dataobj, dtype=float), np.prod(img.header.get_zooms()[:3])


def dice(seg_1, seg_2):
    """Soft Dice score, 1 if both segmentations are empty."""
    denominator = seg_1.sum() + seg_2.sum()
    return 2 * (seg_1 * seg_2).sum() / denominator if denominator > 0 else 1.0


def volume_diff(vol_1, vol_2, prefix):
    """Absolute volume difference |vol_2 - vol_1| in mm3 and in % of the mean volume (0 % if both volumes are 0)."""
    return {
        f"{prefix}_abs_diff_mm3": abs(vol_2 - vol_1),
        f"{prefix}_abs_diff_percent": abs(vol_2 - vol_1) / ((vol_1 + vol_2) / 2) * 100 if vol_1 + vol_2 > 0 else 0.0,
    }


def process_pair(pair, dataset, study_dir, mode):
    out_dir = (study_dir / pair["sc_seg_run-01"]).parent

    # Inference on both runs and registration of run-02 to run-01
    preds = []
    for key in ("scan_run-01", "rescan_run-02"):
        pred = out_dir / Path(pair[key]).name.replace(".nii.gz", f"_label-lesion_desc-{mode}_seg.nii.gz")
        if not pred.exists():
            (predict_tta if mode == "soft-tta" else predict)(dataset / pair[key], pred)
        preds.append(pred)
    pred_2_reg = Path(str(preds[1]).replace(".nii.gz", "_space-run01.nii.gz"))
    if not pred_2_reg.exists():
        run(f"sct_apply_transfo -i {preds[1]} -d {dataset / pair['scan_run-01']} -w {study_dir / pair['warp_run-02_to_run-01']} -x linear -o {pred_2_reg}")

    (seg_1, voxel_vol_1), (seg_2, voxel_vol_2), (seg_2_reg, _) = (load_seg(p) for p in (*preds, pred_2_reg))

    # Manual segmentations (if available)
    manual_1, manual_2 = (load_seg(dataset / pair[f"lesion_seg_run-{r}"])[0] > 0.5 if pair[f"lesion_seg_run-{r}"] else None for r in ("01", "02"))

    def metrics(seg_1, seg_2, seg_2_reg):
        vol_1, vol_2, vol_2_reg = seg_1.sum() * voxel_vol_1, seg_2.sum() * voxel_vol_2, seg_2_reg.sum() * voxel_vol_1
        return {
            "subject": pair["subject"],
            "session": pair["session"],
            "acquisition": pair["acquisition"],
            "has_lesion": pair["has_lesion"],
            VOL_1: vol_1,
            "lesion_volume_run-02_mm3": vol_2,
            **volume_diff(vol_1, vol_2, "lesion_volume"),
            "lesion_volume_run-02_registered_mm3": vol_2_reg,
            **volume_diff(vol_1, vol_2_reg, "lesion_volume_registered"),
            DICE: dice(seg_1, seg_2_reg),
            "dice_vs_manual_run-01": dice(seg_1, manual_1) if manual_1 is not None else np.nan,
            "dice_vs_manual_run-02": dice(seg_2, manual_2) if manual_2 is not None else np.nan,
        }

    # Metrics on the soft segmentations (no threshold), and on the segmentations binarized at each threshold
    soft_metrics = metrics(seg_1, seg_2, seg_2_reg)
    thresholds_metrics = [{"threshold": str(thr), **metrics(seg_1 >= thr, seg_2 >= thr, seg_2_reg >= thr)} for thr in THRESHOLDS]
    return soft_metrics, [{"threshold": "no threshold", **soft_metrics}] + thresholds_metrics


def plot_variability(df, prefix, vol_2, run_2_name, output, dice=False):
    """Plots the lesion volumes, their Bland-Altman plots (mm3 and %) and optionally the Dice, labeled by focal lesion."""
    hue = "Focal lesion"
    mean_volume = (df[VOL_1] + df[vol_2]) / 2
    fig, axes = plt.subplots(1, 4 if dice else 3, figsize=(24 if dice else 18, 5))

    sns.scatterplot(data=df, x=VOL_1, y=vol_2, hue=hue, palette=PALETTE, ax=axes[0])
    max_vol = df[[VOL_1, vol_2]].max().max()
    axes[0].plot([0, max_vol], [0, max_vol], "k--", linewidth=1)
    axes[0].set(title="Soft lesion volume", xlabel="Lesion volume run-01 (mm³)", ylabel=f"Lesion volume {run_2_name} (mm³)")

    # Bland-Altman plots of the volume difference in mm3 and in %
    for ax, diff, unit in ((axes[1], df[f"{prefix}_abs_diff_mm3"], "mm³"), (axes[2], df[f"{prefix}_abs_diff_percent"], "%")):
        sns.scatterplot(x=mean_volume, y=diff, hue=df[hue], palette=PALETTE, ax=ax)
        mean_diff, std_diff = diff.mean(), diff.std()
        for y, name in ((mean_diff, "mean"), (mean_diff + 1.96 * std_diff, "+1.96 SD")):
            ax.axhline(y, color="k", linestyle="--", linewidth=1)
            ax.annotate(f"{name}: {y:.1f}", xy=(1, y), xycoords=("axes fraction", "data"), ha="right", va="bottom", fontsize=8)
        ax.set(title=f"Bland-Altman of lesion volume ({unit})", xlabel="Mean lesion volume (mm³)", ylabel=f"Absolute lesion volume difference |{run_2_name} - run-01| ({unit})")
        sns.move_legend(ax, "upper right")

    if dice:
        sns.boxplot(data=df, x=hue, y=DICE, hue=hue, palette=PALETTE, legend=False, ax=axes[3], showfliers=False)
        sns.stripplot(data=df, x=hue, y=DICE, ax=axes[3], color="k")
        axes[3].set(title="Soft Dice (run-02 registered to run-01)", ylabel="Dice")

    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def plot_thresholds(df, output):
    """Plots the variability metrics of the segmentations binarized at each threshold, labeled by focal lesion."""
    df = df.assign(dice_vs_manual=df[["dice_vs_manual_run-01", "dice_vs_manual_run-02"]].mean(axis=1))
    panels = {
        "lesion_volume_abs_diff_percent": "Absolute lesion volume difference |run-02 - run-01| (%)",
        "lesion_volume_registered_abs_diff_percent": "Absolute lesion volume difference |run-02 registered - run-01| (%)",
        DICE: "Dice (run-02 registered to run-01)",
        "dice_vs_manual": "Dice relative to the manual segmentation (mean of both runs)",
    }
    fig, axes = plt.subplots(2, 2, figsize=(20, 10))
    for ax, (metric, title) in zip(axes.flat, panels.items()):
        sns.boxplot(data=df, x="threshold", y=metric, hue="Focal lesion", palette=PALETTE, ax=ax, showfliers=False,
                    order=["no threshold"] + [str(thr) for thr in THRESHOLDS])
        ax.set(title=title, xlabel="Binarization threshold", ylabel="")
        if "volume" in metric:
            sns.move_legend(ax, "upper right")
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Evaluate the scan-rescan variability of the model soft lesion segmentations.")
    parser.add_argument("-i", "--input", required=True, type=Path, help="Path to the output folder of generate_study_files.py")
    parser.add_argument("-o", "--output", required=True, type=Path, help="Path to the output folder")
    parser.add_argument("-m", "--mode", default="soft", choices=["soft", "soft-tta"], help="Inference mode")
    parser.add_argument("--exclude-stitch", action="store_true", help="Exclude the stitch acquisitions")
    args = parser.parse_args()

    study_dir = args.input.resolve()
    with open(study_dir / "study_files.json") as f:
        study = json.load(f)

    results, results_thresholds = [], []
    for pair in tqdm([p for p in study["pairs"] if p["status"] == "success" and not (args.exclude_stitch and p["acquisition"] == "stitch")]):
        try:
            soft_metrics, thresholds_metrics = process_pair(pair, Path(study["dataset"]), study_dir, args.mode)
            results.append(soft_metrics)
            results_thresholds += thresholds_metrics
        except Exception as e:
            print(f"Failed for {pair['subject']}/{pair['session']}/acq-{pair['acquisition']}: {e}")

    args.output.mkdir(parents=True, exist_ok=True)
    df, df_thresholds = pd.DataFrame(results), pd.DataFrame(results_thresholds)
    for d in (df, df_thresholds):
        d["Focal lesion"] = d["has_lesion"].astype(str).str.strip().str.lower().map(LABELS)
    df.to_csv(args.output / "lesion_seg_variability.csv", index=False)
    df_thresholds.to_csv(args.output / "lesion_seg_variability_thresholds.csv", index=False)
    plot_thresholds(df_thresholds.dropna(subset=["Focal lesion"]), args.output / "lesion_seg_variability_thresholds.png")

    # Only plot the labeled pairs (all of them, and only the pairs with visible focal lesions)
    df = df.dropna(subset=["Focal lesion"])
    for suffix, df_plot in (("", df), ("_visible", df[df["Focal lesion"] == "visible"])):
        plot_variability(df_plot, "lesion_volume", "lesion_volume_run-02_mm3", "run-02", args.output / f"lesion_seg_variability_native{suffix}.png")
        plot_variability(df_plot, "lesion_volume_registered", "lesion_volume_run-02_registered_mm3", "run-02 registered",
                         args.output / f"lesion_seg_variability_registered{suffix}.png", dice=True)
    print(f"Results of {len(results)} pairs saved to {args.output}")


if __name__ == "__main__":
    main()
