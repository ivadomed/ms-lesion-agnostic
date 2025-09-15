"""
This file was created to resample the dataset. The resampling will be performed isotropically with factors: 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7
We only resample the images, not the labels. After inference, the predictions will be resampled back to the original resolution for evaluation.

Input:
    --image-folder: Path to the folder containing input images (NIfTI format)
    --output-folder: Path to the folder where resampled images will be saved

Usage:
    python dataset_resampling.py --image-folder /path/to/images --output-folder /path/to/output/folder

Author: Pierre-Louis Benveniste
"""
import os
import argparse
from pathlib import Path
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Remove small lesions from lesion segmentations.")
    parser.add_argument('--image-folder', required=True, help='Folder with input images')
    parser.add_argument('--output-folder', required=True, help='Folder to save resampled images')
    return parser.parse_args()


def main():
    # Parse arguments
    args = parse_args()
    image_folder = args.image_folder
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
            os.makedirs(output_folder_factor, exist_ok=True)
            # We resample the image with SCT
            output_file_path = os.path.join(output_folder_factor, Path(file).name)
            # If output file already exists, skip
            if os.path.exists(output_file_path):
                print(f"Output file {output_file_path} already exists. Skipping resampling for {file} with factor {factor}.")
                continue
            assert os.system(f"sct_resample -i {str(file)} -o {output_file_path} -f {factor}")==0, f"Error resampling {file} with factor {factor}"
            # Then we export the resampled image to float32 to limit the size
            assert os.system(f"sct_image -i {output_file_path} -o {output_file_path} -type float32")==0, f"Error converting {output_file_path} to float32"

    print("Done")

    return None


if __name__ == "__main__":
    main()