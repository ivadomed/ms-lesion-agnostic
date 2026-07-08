"""
Measures sct_deepseg lesion_ms inference speed on a single image, running each of
the three lesion_ms models 20 times.

Each model is installed once with:
    sct_deepseg lesion_ms -install -custom-url <url>

then timed by repeatedly running:
    sct_deepseg lesion_ms -i <image> -o <output>

Only the inference calls are timed; model installation/download is excluded.
A per-model summary (mean/std/min/max over the runs) is saved as JSON in the
output folder.

Arguments:
    -i / --input       Path to a single UNIT1 image (.nii.gz)
    -o / --output      Path to the output folder
    -n / --n-runs       Number of inference repetitions per model (default: 20)
    --cpu               Run on CPU instead of GPU (GPU is used by default)

Author: Pierre-Louis Benveniste
"""

import argparse
import json
import os
import statistics
import time
from pathlib import Path
from tqdm import tqdm


MODELS = {
    "model_v1": {
        "release": "r20250909",
        "url": {
            "model_fold0": ["https://github.com/ivadomed/seg-sc-ms-lesion-multicontrast/releases/download/r20250909/model_fold0.zip"],
            "model_fold1": ["https://github.com/ivadomed/seg-sc-ms-lesion-multicontrast/releases/download/r20250909/model_fold1.zip"],
            "model_fold2": ["https://github.com/ivadomed/seg-sc-ms-lesion-multicontrast/releases/download/r20250909/model_fold2.zip"],
            "model_fold3": ["https://github.com/ivadomed/seg-sc-ms-lesion-multicontrast/releases/download/r20250909/model_fold3.zip"],
            "model_fold4": ["https://github.com/ivadomed/seg-sc-ms-lesion-multicontrast/releases/download/r20250909/model_fold4.zip"]
        },
        "crop": False
    },
    "model_v2": {
        "release": "r20260629",
        "url": {
            "model_fold0": ["https://github.com/ivadomed/seg-sc-ms-lesion-multicontrast/releases/download/r20260629/model_fold0.zip"],
            "model_fold1": ["https://github.com/ivadomed/seg-sc-ms-lesion-multicontrast/releases/download/r20260629/model_fold1.zip"],
            "model_fold2": ["https://github.com/ivadomed/seg-sc-ms-lesion-multicontrast/releases/download/r20260629/model_fold2.zip"],
            "model_fold3": ["https://github.com/ivadomed/seg-sc-ms-lesion-multicontrast/releases/download/r20260629/model_fold3.zip"],
            "model_fold4": ["https://github.com/ivadomed/seg-sc-ms-lesion-multicontrast/releases/download/r20260629/model_fold4.zip"]
        },
        "crop": True
    },
    "model_v3": {
        "release": "r20260703",
        "url": {
            "model_fold0": ["https://github.com/ivadomed/seg-sc-ms-lesion-multicontrast/releases/download/r20260703/model_fold0.zip"],
            "model_fold1": ["https://github.com/ivadomed/seg-sc-ms-lesion-multicontrast/releases/download/r20260703/model_fold1.zip"],
            "model_fold2": ["https://github.com/ivadomed/seg-sc-ms-lesion-multicontrast/releases/download/r20260703/model_fold2.zip"],
            "model_fold3": ["https://github.com/ivadomed/seg-sc-ms-lesion-multicontrast/releases/download/r20260703/model_fold3.zip"],
            "model_fold4": ["https://github.com/ivadomed/seg-sc-ms-lesion-multicontrast/releases/download/r20260703/model_fold4.zip"]
        },
        "crop": True
    }
}


def parse_args():
    parser = argparse.ArgumentParser(description="Measure sct_deepseg lesion_ms inference speed on a single image with each of the 3 models over N runs.")
    parser.add_argument("-i", "--input", required=True, help="Path to a single UNIT1 image")
    parser.add_argument("-o", "--output", required=True, help="Path to the output folder")
    parser.add_argument("-n", "--n-runs", type=int, default=20, help="Number of inference repetitions per model (default: 20)")
    parser.add_argument("--cpu", action="store_true", help="Run on CPU instead of GPU (GPU is used by default)")
    return parser.parse_args()


def main():
    args = parse_args()
    image = Path(args.input).resolve()
    output_root = Path(args.output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    env_prefix = "" if args.cpu else "SCT_USE_GPU=1 "

    results = {}

    for model_name, model_info in MODELS.items():
        print(f"Installing model '{model_name}' ...")
        # Model url is the path to each fold seperated by a space, so we need to join the list of urls into a single string
        model_url = " ".join([url for fold_urls in model_info['url'].values() for url in fold_urls])
        assert os.system(f"sct_deepseg lesion_ms -install -custom-url {model_url}") == 0, f"Installation failed for {model_name}"

        model_output_path = output_root / f"{model_name}_pred.nii.gz"
        crop_flag = "" if model_info["crop"] else "-no-crop"

        times = []
        for run_idx in tqdm(range(1, args.n_runs + 1), desc=model_name):
            if model_output_path.exists():
                model_output_path.unlink()
            cmd = f"{env_prefix}sct_deepseg lesion_ms -i {image} -o {model_output_path} {crop_flag}"
            start = time.perf_counter()
            assert os.system(cmd) == 0, f"Inference run {run_idx} failed for {model_name}"
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        results[model_name] = {
            "release": model_info["release"],
            "n_runs": args.n_runs,
            "times_sec": times,
            "mean_sec": statistics.mean(times),
            "std_sec": statistics.stdev(times) if len(times) > 1 else 0.0,
            "min_sec": min(times),
            "max_sec": max(times),
        }

        print(f"{model_name}: mean {results[model_name]['mean_sec']:.2f}s +/- {results[model_name]['std_sec']:.2f}s over {args.n_runs} runs")

    summary_path = output_root / "speed_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nSaved speed summary to {summary_path}")


if __name__ == "__main__":
    main()
