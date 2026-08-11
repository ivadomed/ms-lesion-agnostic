"""
Aggregates the scan-rescan stability CSVs (volume_stability.csv and
lesionwise_stability.csv) produced by evaluate_lesion_stability.py across models and
acquisitions, merges them, and writes a comparative statistical analysis across
models to a log file.

At least one of --upper-lower-root / --stitch-root must be given.
    --upper-lower-root  Output root of `evaluate_lesion_stability.py -a upper-lower`
                         (expects <model>/upper/volume_stability.csv and <model>/lower/...)
    --stitch-root        Output root of `evaluate_lesion_stability.py -a stitch` (default)
                         (expects <model>/volume_stability.csv)
    -o / --output        Output folder for the merged CSVs and the comparison log

Statistical comparison across models, computed separately per acquisition:
    - Whole-scan volume CV is paired by subject/session across models (every model
      is run on the same scans), so models are compared with a Friedman test
      (omnibus) and pairwise Wilcoxon signed-rank tests.
    - Matched lesion-wise CV is not paired across models (per-model lesion ids are
      not the same anatomical lesion), so models are compared with a Kruskal-Wallis
      test (omnibus) and pairwise Mann-Whitney U tests.

Output:
    <output>/volume_stability_merged.csv
    <output>/lesionwise_stability_merged.csv   (only if any lesionwise CSV is found)
    <output>/comparison.log

Author: Pierre-Louis Benveniste
"""

import argparse
from pathlib import Path

import pandas as pd
from scipy import stats


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate scan-rescan stability CSVs and compare models statistically.")
    parser.add_argument("--upper-lower-root", type=str, default=None, help="Output root of evaluate_lesion_stability.py -a upper-lower")
    parser.add_argument("--stitch-root", type=str, default=None, help="Output root of evaluate_lesion_stability.py -a stitch")
    parser.add_argument("-o", "--output", required=True, help="Output folder for the merged CSVs and comparison log")
    args = parser.parse_args()
    if not args.upper_lower_root and not args.stitch_root:
        parser.error("At least one of --upper-lower-root or --stitch-root must be provided")
    return args


def collect_csv(root: Path, filename: str, is_upper_lower: bool):
    """Collect and concatenate every `filename` CSV found under `root`, tagging rows with acquisition."""
    frames = []
    for csv_path in sorted(root.rglob(filename)):
        df = pd.read_csv(csv_path)
        if df.empty:
            continue
        df["acquisition"] = csv_path.parent.name if is_upper_lower else "stitch"
        frames.append(df)
    return frames


def friedman_test(df: pd.DataFrame, value_col: str, group_cols):
    """Friedman test of `value_col` across all models, matched on `group_cols`."""
    models = sorted(df["model"].unique())
    if len(models) < 3:
        return None
    pivot = df.pivot_table(index=group_cols, columns="model", values=value_col).dropna()
    if len(pivot) < 2:
        return None
    stat, p = stats.friedmanchisquare(*[pivot[m] for m in models])
    return len(pivot), stat, p


def pairwise_wilcoxon(df: pd.DataFrame, value_col: str, group_cols):
    """Pairwise Wilcoxon signed-rank tests of `value_col` between every pair of models."""
    models = sorted(df["model"].unique())
    results = []
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            m1, m2 = models[i], models[j]
            merged = pd.merge(
                df[df["model"] == m1][group_cols + [value_col]],
                df[df["model"] == m2][group_cols + [value_col]],
                on=group_cols, suffixes=(f"_{m1}", f"_{m2}"),
            )
            if len(merged) < 2:
                results.append((m1, m2, len(merged), None, None))
                continue
            stat, p = stats.wilcoxon(merged[f"{value_col}_{m1}"], merged[f"{value_col}_{m2}"])
            results.append((m1, m2, len(merged), stat, p))
    return results


def kruskal_test(df: pd.DataFrame, value_col: str):
    """Kruskal-Wallis test of `value_col` across all models (independent samples)."""
    models = sorted(df["model"].unique())
    groups = [df[df["model"] == m][value_col].dropna() for m in models]
    if len(models) < 3 or any(len(g) == 0 for g in groups):
        return None
    stat, p = stats.kruskal(*groups)
    return stat, p


def pairwise_mannwhitney(df: pd.DataFrame, value_col: str):
    """Pairwise Mann-Whitney U tests of `value_col` between every pair of models (independent samples)."""
    models = sorted(df["model"].unique())
    results = []
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            m1, m2 = models[i], models[j]
            g1 = df[df["model"] == m1][value_col].dropna()
            g2 = df[df["model"] == m2][value_col].dropna()
            if len(g1) == 0 or len(g2) == 0:
                results.append((m1, m2, len(g1), len(g2), None, None))
                continue
            stat, p = stats.mannwhitneyu(g1, g2, alternative="two-sided")
            results.append((m1, m2, len(g1), len(g2), stat, p))
    return results


def write_comparison(log_lines, df: pd.DataFrame, value_col: str, title: str, paired: bool, group_cols=None):
    log_lines.append(f"\n=== {title} ===")
    for acquisition in sorted(df["acquisition"].unique()):
        sub_df = df[df["acquisition"] == acquisition]
        log_lines.append(f"\n-- acquisition: {acquisition} --")

        log_lines.append("Model summary (mean +/- std, n):")
        for model, group in sub_df.groupby("model"):
            values = group[value_col].dropna()
            log_lines.append(f"  {model}: {values.mean():.2f} +/- {values.std():.2f}  (n={len(values)})")

        if paired:
            friedman = friedman_test(sub_df, value_col, group_cols)
            if friedman is not None:
                n, stat, p = friedman
                log_lines.append(f"Friedman test across models (n={n} paired {'/'.join(group_cols)}): statistic={stat:.4f}, p={p:.4g}")
            else:
                log_lines.append("Friedman test: not enough models/paired data")

            log_lines.append("Pairwise Wilcoxon signed-rank tests:")
            for m1, m2, n, stat, p in pairwise_wilcoxon(sub_df, value_col, group_cols):
                if stat is None:
                    log_lines.append(f"  {m1} vs {m2}: not enough paired data (n={n})")
                else:
                    log_lines.append(f"  {m1} vs {m2} (n={n}): statistic={stat:.4f}, p={p:.4g}")
        else:
            kruskal = kruskal_test(sub_df, value_col)
            if kruskal is not None:
                stat, p = kruskal
                log_lines.append(f"Kruskal-Wallis test across models: statistic={stat:.4f}, p={p:.4g}")
            else:
                log_lines.append("Kruskal-Wallis test: not enough models/data")

            log_lines.append("Pairwise Mann-Whitney U tests:")
            for m1, m2, n1, n2, stat, p in pairwise_mannwhitney(sub_df, value_col):
                if stat is None:
                    log_lines.append(f"  {m1} vs {m2}: not enough data (n={n1}, {n2})")
                else:
                    log_lines.append(f"  {m1} vs {m2} (n={n1}, {n2}): statistic={stat:.4f}, p={p:.4g}")


def main():
    args = parse_args()
    output_root = Path(args.output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    volume_frames = []
    lesion_frames = []

    if args.upper_lower_root:
        upper_lower_root = Path(args.upper_lower_root).resolve()
        volume_frames += collect_csv(upper_lower_root, "volume_stability.csv", is_upper_lower=True)
        lesion_frames += collect_csv(upper_lower_root, "lesionwise_stability.csv", is_upper_lower=True)

    if args.stitch_root:
        stitch_root = Path(args.stitch_root).resolve()
        volume_frames += collect_csv(stitch_root, "volume_stability.csv", is_upper_lower=False)
        lesion_frames += collect_csv(stitch_root, "lesionwise_stability.csv", is_upper_lower=False)

    if not volume_frames:
        raise SystemExit("No volume_stability.csv files found")

    volume_df = pd.concat(volume_frames, ignore_index=True)
    volume_df.to_csv(output_root / "volume_stability_merged.csv", index=False)
    print(f"Merged {len(volume_frames)} volume_stability.csv file(s) -> {len(volume_df)} rows")

    lesion_df = None
    if lesion_frames:
        lesion_df = pd.concat(lesion_frames, ignore_index=True)
        lesion_df.to_csv(output_root / "lesionwise_stability_merged.csv", index=False)
        print(f"Merged {len(lesion_frames)} lesionwise_stability.csv file(s) -> {len(lesion_df)} rows")

    log_lines = ["Scan-rescan lesion stability - comparative analysis across models"]

    write_comparison(log_lines, volume_df, "cv_percent", "Whole-scan lesion volume CV", paired=True, group_cols=["subject", "session"])

    if lesion_df is not None:
        matched_lesion_df = lesion_df[lesion_df["matched"] == True].copy()
        if not matched_lesion_df.empty:
            write_comparison(log_lines, matched_lesion_df, "cv_percent", "Matched lesion-wise CV", paired=False)

    log_path = output_root / "comparison.log"
    with open(log_path, "w") as f:
        f.write("\n".join(log_lines) + "\n")
    print(f"Comparison log written to {log_path}")


if __name__ == "__main__":
    main()
