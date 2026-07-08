"""
Stitches the upper and lower FOV UNIT1 acquisitions of a BIDS dataset together
using `sct_image -stitch` with the Graf et al. method, producing a single acq-stitch image per run.
This is only done for the UNIT1.nii.gz images.
For each sub-*/ses-*/anat directory, every acq-upper_run-XX_UNIT1.nii.gz is
paired with the matching acq-lower_run-XX_UNIT1.nii.gz (same subject, session
and run) and stitched with:

    sct_image -i <upper> <lower> -stitch -o <sub>_<ses>_acq-stitch_run-XX_UNIT1.nii.gz

Arguments:
    -i / --input       Path to the BIDS dataset root

Author: Pierre-Louis Benveniste
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path
import tqdm
import os


def parse_args():
    p = argparse.ArgumentParser(description="Stitch upper/lower FOV images with sct_image -stitch.")
    p.add_argument("-i", "--input", required=True, help="Path to the BIDS dataset root")
    return p.parse_args()


def find_pairs(bids_root: Path):
    """Yield (upper_path, lower_path, output_path) for every matching upper/lower run."""
    upper_scans = list(bids_root.rglob("*_acq-upper_run-*_UNIT1.nii.gz"))

    for upper_path in upper_scans:
        # Get the corresponding lower path
        lower_path = Path(str(upper_path).replace("_acq-upper_", "_acq-lower_"))
        if not lower_path.exists():
            print(f"WARNING: No matching lower FOV image for {upper_path}")
            continue

        # Get the output path
        output_path = Path(str(upper_path).replace("_acq-upper_", "_acq-stitch_"))
        yield upper_path, lower_path, output_path


def main():
    args = parse_args()
    bids_root = Path(args.input).resolve()

    pairs = list(find_pairs(bids_root))
    print(f"Found {len(pairs)} upper/lower pair(s) to stitch")

    errors = []
    for upper_path, lower_path, output_path in tqdm.tqdm(pairs, desc="Stitching"):

        assert os.system(f"sct_image -i {upper_path} {lower_path} -stitch -o {output_path}")==0, f"Stitching failed for {upper_path} and {lower_path}"

    print("\nDone.")


if __name__ == "__main__":
    main()
