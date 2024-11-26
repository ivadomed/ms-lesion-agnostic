"""
This script runs QC on the datasets to verify the quality of the manual segmentations. 
It uses sct_qc to generate a report for each dataset.

Args:
    -d: path to the dataset
    -qc: path to the qc folder
    -suffix: suffix of the derivative files in the dataset

Output:
    - QC reports for each dataset

Usage:
    python qc_datasets.py -d dataset -qc qc_folder -suffix suffix

Author: 
    Pierre-Louis Benveniste
"""
import argparse
import os
from tqdm import tqdm
from pathlib import Path
from utils.image import Image


def parse_arguments():
    parser = argparse.ArgumentParser(description='Run QC on the datasets to verify the quality of the manual segmentations.')
    parser.add_argument("-d", "--dataset", help="Path to the dataset.")
    parser.add_argument("-qc", "--qc_folder", help="Path to the qc folder.")
    parser.add_argument("-suffix", help="Suffix of the derivative files in the dataset. (Example: _lesion-manual)")
    args = parser.parse_args()
    return args


def main():
    # Parse arguments
    args = parse_arguments()
    dataset = args.dataset
    qc_folder = args.qc_folder
    suffix = args.suffix

    # Get all the derivatives in the dataset which have the suffix with rglob
    derivatives = list(Path(dataset).rglob(f"*{suffix}.nii.gz"))

    # Problematic files:
    problematic_files = []

    # Perform QC on each derivative
    for derivative in tqdm(derivatives):
        # Get the corresponding image
        image = str(derivative).replace("derivatives/labels/", "").replace(suffix, "")
        # Segment the spinal cord
        sc_seg_file = os.path.join(qc_folder, f"{Path(image).stem.replace('.nii','')}_sc_seg.nii.gz")
        os.system(f"sct_deepseg -i {image} -task seg_sc_contrast_agnostic -o {sc_seg_file}")
        # Dilate the spinal cord segmentation by 4 voxels
        os.system(f"sct_maths -i {sc_seg_file} -dilate 4 -o {sc_seg_file}") 
        # If ax is in image name than run axial QC
        if 'ax' in image:
            # Run QC axial on the spinal cord segmentation to check the quality of the segmentation
            os.system(f"sct_qc -i {image} -s {sc_seg_file} -d {derivative} -plane axial -p sct_deepseg_lesion -qc {qc_folder}") 
        else:
            # Run QC sagittal on the spinal cord segmentation to check the quality of the segmentation
            os.system(f"sct_qc -i {image} -s {sc_seg_file} -d {derivative} -plane sagittal -p sct_deepseg_lesion -qc {qc_folder}")
        # Check if the lesions are all in the spinal cord
        # Load both 
        sc_seg_data = Image(str(sc_seg_file)).data
        derivative_data = Image(str(derivative)).data
        # Check if the lesions are all in the spinal cord by doing the difference
        diff = derivative_data - (derivative_data * sc_seg_data)
        # If lesions are outside than some values will be positive
        if diff.max() > 0:
            print(f"Lesions are outside the spinal cord for {derivative}")
            print(f"Check {sc_seg_file} and {derivative}")
            print(f"Check the QC report in {qc_folder}")
            problematic_files.append(derivative)
    # In the QC folder we print the problematic files
    with open(os.path.join(qc_folder, "problematic_files.txt"), "w") as f:
        for file in problematic_files:
            f.write(file + "\n")


if __name__ == "__main__":
    main()