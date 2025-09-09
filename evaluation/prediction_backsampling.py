"""
This script is used to sample the predictions back to the original resolution of the image after inference.

Input:
    --prediction-folder: Path to the folder containing prediction images (NIfTI format)
    --image-folder: Path to the folder containing original images (NIfTI format)
    --output-folder: Path to the folder where resampled predictions will be saved

Output: 
    None

Example:
    python prediction_backsampling.py --prediction-folder /path/to/predictions --image-folder /path/to/images --output-folder /path/to/output/folder

Author: Pierre-Louis Benveniste
"""
import os
import argparse
from pathlib import Path
from tqdm import tqdm
import shutil


def parse_args():
    parser = argparse.ArgumentParser(description="Backsample predictions to original image resolution.")
    parser.add_argument('--prediction-folder', required=True, help='Folder with prediction images')
    parser.add_argument('--image-folder', required=True, help='Folder with original images')
    parser.add_argument('--output-folder', required=True, help='Folder to save resampled predictions')
    return parser.parse_args()


def main():
    # Parse arguments
    args = parse_args()
    prediction_folder = args.prediction_folder
    image_folder = args.image_folder
    output_folder = args.output_folder

    # Build the output folder if it does not exist
    os.makedirs(output_folder, exist_ok=True)

    # List all predictions
    list_predictions = sorted(list(Path(prediction_folder).rglob("*.nii.gz")))

    # List all original images
    list_images = sorted(list(Path(image_folder).rglob("*.nii.gz")))

    # Iterate through the predictions
    for pred_file in tqdm(list_predictions):
        # Find the corresponding original image (which has the same name but with _0000.nii.gz instead of .nii.gz)
        original_image_file = os.path.join(image_folder, Path(pred_file).name.replace(".nii.gz", "_0000.nii.gz"))
        if not os.path.exists(original_image_file):
            print(f"Warning: Original image {original_image_file} does not exist. Skipping {pred_file}.")
            break
        # Create a temp folder for this prediction
        temp_folder = os.path.join(output_folder, Path(pred_file).name.replace(".nii.gz", ""))
        os.makedirs(temp_folder, exist_ok=True)

        # Resample the prediction to the original image resolution
        output_file_path = os.path.join(temp_folder, Path(pred_file).name)

        # Run command line
        assert os.system(f"sct_register_multimodal -i {pred_file} -d {original_image_file} -identity 1 -o {output_file_path} -x nn") == 0

        # Copy the output file to the output folder
        output_final_path = os.path.join(output_folder, Path(pred_file).name)
        shutil.copy(output_file_path, output_final_path)

        # Clean up the temporary folder
        shutil.rmtree(temp_folder)

    print("Done")


if __name__ == "__main__":
    main()