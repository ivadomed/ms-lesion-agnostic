"""
Runs lesion segmentation inference with three sct_deepseg lesion_ms models on every
UNIT1 image of a BIDS dataset.

Each model is installed with:
    sct_deepseg lesion_ms -install -custom-url <url>

and applied to every image with:
    sct_deepseg lesion_ms -i <image> -o <output>

Predictions are stored in one BIDS-convention subfolder of the output folder per model:
    <output>/<model_release>/sub-XXX/ses-XXX/anat/sub-XXX_..._UNIT1_label-lesion_seg.nii.gz

Arguments:
    -i / --input       Path to the BIDS dataset root
    -o / --output      Path to the output folder

Author: Pierre-Louis Benveniste
"""

import argparse
import os
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
    parser = argparse.ArgumentParser(description="Run sct_deepseg lesion_ms inference on all UNIT1 images with 3 models.")
    parser.add_argument("-i", "--input", required=True, help="Path to the BIDS dataset root")
    parser.add_argument("-o", "--output", required=True, help="Path to the output folder")
    return parser.parse_args()


def main():
    args = parse_args()
    bids_root = Path(args.input).resolve()
    output_root = Path(args.output).resolve()

    images = sorted(bids_root.rglob("*_UNIT1.nii.gz"))
    print(f"Found {len(images)} UNIT1 image(s)")

    # Build SC seg folder
    sc_seg_folder = output_root / "sc_seg"
    sc_seg_folder.mkdir(parents=True, exist_ok=True)

    # Build the QC folder
    qc_folder = output_root / "qc"
    qc_folder.mkdir(parents=True, exist_ok=True)

    for model_name, model_info in MODELS.items():
        print(f"Installing model '{model_name}' ...")
        # Model url is the path to each fold seperated by a space, so we need to join the list of urls into a single string
        model_url = " ".join([url for fold_urls in model_info['url'].values() for url in fold_urls])
        print(model_url)
        assert os.system(f"sct_deepseg lesion_ms -install -custom-url {model_url}") == 0, f"Installation failed for {model_name}"

        model_output_root = output_root / model_info['release']

        for image in tqdm(images, desc=model_name):
            relative_path = image.relative_to(bids_root)

            # Segment sc for QC
            sc_output_path = sc_seg_folder / relative_path.parent / relative_path.name.replace("_UNIT1.nii.gz", "_UNIT1_label-sc_seg.nii.gz")
            if not sc_output_path.exists():
                sc_output_path.parent.mkdir(parents=True, exist_ok=True)
                assert os.system(f"SCT_USE_GPU=1 sct_deepseg sc -i {image} -o {sc_output_path}") == 0, f"Prediction failed for {image} with sct_deepseg sc"
        
            # Segment lesion
            lesion_output_path = model_output_root / relative_path.parent / relative_path.name.replace("_UNIT1.nii.gz", "_UNIT1_label-lesion_seg.nii.gz")
            lesion_output_path.parent.mkdir(parents=True, exist_ok=True)
            assert os.system(f"SCT_USE_GPU=1 sct_deepseg lesion_ms -i {image} -o {lesion_output_path} {'-no-crop' if not model_info['crop'] else ''} -qc {qc_folder} -qc-seg {sc_output_path}") == 0, f"Prediction failed for {image} with {model_name}"

    print("\nDone.")


if __name__ == "__main__":
    main()
