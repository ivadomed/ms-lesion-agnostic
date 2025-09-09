"""
This script is used to label the disks on the nnunet format dataset. 
The idea is to use the labeled disks to evaluate the performance of the model on each chunk of the spinal cord. 

Input:
    --input-path: path to the input folder containing the nnunet format dataset
    --output-path: path to the output folder where the labeled disks will be saved
    --min-idx: the minimum value of the index of the file
    --max-idx: the max value of the index of the file

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
import time


def parse_args():
    parser = argparse.ArgumentParser(description='Label disks on the nnunet format dataset')
    parser.add_argument('--input-path', type=str, help='Path to the input folder containing the nnunet format dataset')
    parser.add_argument('--output-path', type=str, help='Path to the output folder where the labeled disks will be saved')
    parser.add_argument('--min-idx', type=int, help='the minimum value of the index of the file')
    parser.add_argument('--max-idx', type=int, help='the maximum value of the index of the file')
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

    # Only keep the files between the two index values
    files_to_segment = [f for f in files_to_segment if int(f.stem.split("_")[-2]) >= args.min_idx and int(f.stem.split("_")[-2]) <= args.max_idx]

    # Initialize list of failed inferences
    failed_inferences = []

    # Iterate over the files and label disks
    for file in tqdm(files_to_segment):
        # Build a temp folder for the current file
        temp_folder = os.path.join(output_folder, f"{Path(file).name.replace('.nii.gz','')}_temp")
        os.makedirs(temp_folder, exist_ok=True)
        # Run totalspineseg
        output_file = os.path.join(temp_folder, "temp.nii.gz")
        # If the segmentation fails, we add the subject to the list of failed segmentations
        output_code = os.system(f"sct_deepseg totalspineseg -i {str(file)} -o {output_file}")
        # If fails
        if output_code!=0:
            failed_inferences.append(Path(file))
        # If doesn't fail then we copy the results
        else:
            # Then we copy only the file that we want which is called ..._levels.nii.gz
            sct_output = str(output_file).replace(".nii.gz", "_step1_levels.nii.gz")
            final_output = os.path.join(output_folder, Path(file).name)
            shutil.copy(sct_output, final_output)
        # Remove temp folder
        assert os.system(f"rm -rf {temp_folder}")==0
    # Save a txt file in the output folder with the failed inferences 
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_path_failed_inf = os.path.join(output_folder, f"failed_inferences_{timestamp}.txt")
    with open(output_path_failed_inf, "w") as f:
        for item in failed_inferences:
            f.write(str(item) + "\n")

    print("Done with the segmentations")

    return None


if __name__ == "__main__":
    main()