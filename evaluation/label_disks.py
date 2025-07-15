"""
This script is used to label the disks on the nnunet format dataset. 
The idea is to use the labeled disks to evaluate the performance of the model on each chunk of the spinal cord. 

Input:
    --input-path: path to the input folder containing the nnunet format dataset
    --output-path: path to the output folder where the labeled disks will be saved

Output: 
    None

Example:
    python label_disks.py --input-path /path/to/nnunet-dataset --output-path /path/to/output-folder

Author: Pierre-Louis Benveniste
"""
import os
import argparse
from pathlib import Path
from tqdm import tqdm
import shutil


def parse_args():
    parser = argparse.ArgumentParser(description='Label disks on the nnunet format dataset')
    parser.add_argument('--input-path', type=str, help='Path to the input folder containing the nnunet format dataset')
    parser.add_argument('--output-path', type=str, help='Path to the output folder where the labeled disks will be saved')
    args = parser.parse_args()
    return args


def main():

    # Parse arguments
    args = parse_args()
    input_folder = Path(args.input_path)
    output_folder = Path(args.output_path)

    # Create output directory if it doesn't exist
    output_folder.mkdir(parents=True, exist_ok=True)

    # Get the files to segment
    files_to_segment = sorted(list(input_folder.rglob("*.nii.gz")))

    # Iterate over the files and label disks
    for file in tqdm(files_to_segment):
        # Build a temp folder for the current file
        temp_folder = os.path.join(output_folder, f"{Path(file).name}_temp")
        os.makedirs(temp_folder, exist_ok=True)
        # Run totalspineseg
        output_file = os.path.join(temp_folder, "temp.nii.gz")
        assert os.system(f"sct_deepseg totalspineseg -i {str(file)} -o {output_file}") == 0, "SCT segmentation failed"
        # Then we copy only the file that we want which is called ..._levels.nii.gz
        sct_output = str(output_file).replace(".nii.gz", "_step1_levels.nii.gz")
        final_output = os.path.join(output_folder, Path(file).name)
        shutil.copy(sct_output, final_output)

    print("Done with the segmentations")

    return None


if __name__ == "__main__":
    main()