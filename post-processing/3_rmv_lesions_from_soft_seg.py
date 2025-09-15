"""
This script uses the binary segmentations where lesions were removed because outside of the spinal cord to remove those lesions from the soft segmentations.
It uses SCT to do so through multiplication

Input:
    --binary-seg-folder: Path to the folder containing the binary segmentations where lesions were removed.
    --soft-seg-folder: Path to the folder containing the soft segmentations.
    --output: Path to the output folder where the masked soft segmentations will be saved.

Output:
    None

Example:
    python 3_rmv_lesions_from_soft_seg.py --binary-seg-folder /path/to/binary/segmentation --soft-seg-folder /path/to/soft/segmentation --output /path/to/output/folder

Author: Pierre-Louis Benveniste
"""
import argparse
import os
from pathlib import Path
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Remove lesions from soft segmentations using binary segmentations.")
    parser.add_argument("--binary-seg-folder", type=str, required=True, help="Path to the folder containing the binary segmentations where lesions were removed.")
    parser.add_argument("--soft-seg-folder", type=str, required=True, help="Path to the folder containing the soft segmentations.")
    parser.add_argument("--output", type=str, required=True, help="Path to the output folder where the masked soft segmentations will be saved.")
    return parser.parse_args()


def main():
    args = parse_args()
    binary_seg_folder = args.binary_seg_folder
    soft_seg_folder = args.soft_seg_folder
    output_folder = args.output

    # List all binary segmentations
    list_binary_segs = sorted(list(Path(binary_seg_folder).rglob("*.nii.gz")))

    # Create the output folder if it does not exist
    os.makedirs(output_folder, exist_ok=True)

    for binary_seg in tqdm(list_binary_segs):
        # Get the corresponding soft segmentation file
        soft_seg = os.path.join(soft_seg_folder, Path(binary_seg).name)

        # Check if the soft segmentation exists
        if not os.path.exists(soft_seg):
            print(f"Soft segmentation {soft_seg} does not exist. Skipping.")
            break

        # Build output file path
        output_file = os.path.join(output_folder, Path(binary_seg).name)

        # Use SCT to multiply the soft segmentation by the binary segmentation
        assert os.system(f"sct_maths -i {soft_seg} -mul {binary_seg} -o {output_file}") == 0, "SCT command failed"

    print("Soft segmentations have been masked with binary segmentations where lesions were removed.")

    return None


if __name__ == "__main__":
    main()