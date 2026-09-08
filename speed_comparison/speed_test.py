"""
Measures sct_deepseg lesion_ms inference speed by running inference once on each of
N images (default: 20) of a dataset, for each of the three lesion_ms models.

Each model is installed once with:
    sct_deepseg lesion_ms -install -custom-url <url>

then timed by running, for each of the N images:
    sct_deepseg lesion_ms -i <image> -o <output>

Only the inference calls are timed; model installation/download is excluded.
A per-model summary (mean/std/min/max over the N images) is saved as JSON in the
output folder.

Arguments:
    -i / --input        Path to the BIDS dataset root
    -o / --output        Path to the output folder
    -n / --n-images       Number of images to run inference on, per model (default: 20)
    --cpu                 Run on CPU instead of GPU (GPU is used by default)
    --seed                Random seed used to pick the images (default: 42)

Author: Pierre-Louis Benveniste
"""

import argparse
import json
import os
import random
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
    parser = argparse.ArgumentParser(description="Measure sct_deepseg lesion_ms inference speed by running inference once on each of N images, for each of the 3 models.")
    parser.add_argument("-i", "--input", required=True, help="Path to the dataset root")
    parser.add_argument("-o", "--output", required=True, help="Path to the output folder")
    parser.add_argument("-n", "--n-images", type=int, default=20, help="Number of images to run inference on, per model (default: 20)")
    parser.add_argument("--cpu", action="store_true", help="Run on CPU instead of GPU (GPU is used by default)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used to pick the images (default: 42)")
    return parser.parse_args()


def main():
    args = parse_args()
    bids_root = Path(args.input).resolve()
    output_root = Path(args.output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    all_images = sorted(bids_root.rglob("*.nii.gz"))
    assert len(all_images) >= args.n_images, f"Requested {args.n_images} images but only found {len(all_images)} image(s) under {bids_root}"
    images = random.Random(args.seed).sample(all_images, args.n_images)
    print(f"Using {len(images)} randomly selected image(s) (seed={args.seed})")

    env_prefix = "" if args.cpu else "SCT_USE_GPU=1 "

    results = {}

    for model_name, model_info in MODELS.items():
        print(f"Installing model '{model_name}' ...")
        # Model url is the path to each fold seperated by a space, so we need to join the list of urls into a single string
        model_url = " ".join([url for fold_urls in model_info['url'].values() for url in fold_urls])
        assert os.system(f"sct_deepseg lesion_ms -install -custom-url {model_url}") == 0, f"Installation failed for {model_name}"

        model_output_folder = output_root / model_name
        model_output_folder.mkdir(parents=True, exist_ok=True)
        crop_flag = "" if model_info["crop"] else "-no-crop"

        times = []
        for image in tqdm(images, desc=model_name):
            output_path = model_output_folder / image.name.replace(".nii.gz", "_label-lesion_seg.nii.gz")
            if output_path.exists():
                output_path.unlink()
            cmd = f"{env_prefix} sct_deepseg lesion_ms -i {image} -o {output_path} {crop_flag}"
            start = time.perf_counter()
            assert os.system(cmd) == 0, f"Inference failed for {image} with {model_name}"
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        results[model_name] = {
            "release": model_info["release"],
            "n_images": len(images),
            "times_sec": times,
            "mean_sec": statistics.mean(times),
            "std_sec": statistics.stdev(times) if len(times) > 1 else 0.0,
            "min_sec": min(times),
            "max_sec": max(times),
        }

        print(f"{model_name}: mean {results[model_name]['mean_sec']:.2f}s +/- {results[model_name]['std_sec']:.2f}s over {len(images)} images")

    summary_path = output_root / "speed_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nSaved speed summary to {summary_path}")


if __name__ == "__main__":
    main()
