"""
This file was created to resample the dataset. The resampling will be performed isotropically with factors: 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7

Input:
    --image-folder: Path to the folder containing input images (NIfTI format)
    --label-folder: Path to the folder containing the labels (NIfTI format)
    --output-folder: Path to the folder where resampled images will be saved

Usage:
    python dataset_resampling.py --image-folder /path/to/images --label-folder /path/to/labels --output-folder /path/to/output/folder

Author: Pierre-Louis Benveniste
"""
import os
import argparse
from pathlib import Path
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Remove small lesions from lesion segmentations.")
    parser.add_argument('--image-folder', required=True, help='Folder with input images')
    parser.add_argument('--label-folder', required=True, help='Folder with label images')
    parser.add_argument('--output-folder', required=True, help='Folder to save resampled images')
    return parser.parse_args()


def main():
    # Parse arguments
    args = parse_args()
    image_folder = args.image_folder
    label_folder = args.label_folder
    output_folder = args.output_folder

    # Build the output folder if it does not exist
    os.makedirs(output_folder, exist_ok=True)

    # List all predictions
    list_images = sorted(list(Path(image_folder).rglob("*.nii.gz")))

    # List of resampling factors:
    resampling_factors = [0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7]

    # Iterate throught the files
    for file in tqdm(list_images):
        # Iterate through the resampling factors
        for factor in resampling_factors:
            # For each factor, we build an output folder
            output_folder_factor = os.path.join(output_folder, f"images_{factor}")
            output_folder_label_factor = os.path.join(output_folder, f"labels_{factor}")
            os.makedirs(output_folder_factor, exist_ok=True)
            os.makedirs(output_folder_label_factor, exist_ok=True)
            # We resample the image with SCT
            output_file_path = os.path.join(output_folder_factor, Path(file).name)
            assert os.system(f"sct_resample -i {str(file)} -o {output_file_path} -f {factor}")==0, f"Error resampling {file} with factor {factor}"
            # We resample the label with SCT
            label_file = os.path.join(label_folder, Path(file).name.replace("_0000.nii.gz", ".nii.gz"))
            output_label_file_path = os.path.join(output_folder_label_factor, Path(label_file).name)
            assert os.system(f"sct_resample -i {label_file} -o {output_label_file_path} -f {factor}")==0, f"Error resampling {label_file} with factor {factor}"

    print("Done")

    return None


if __name__ == "__main__":
    main()