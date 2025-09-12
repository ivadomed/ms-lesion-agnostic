"""
This script registers the labels from the nnunet folder to the super-resolved images.
It uses sct_register_multimodal -identity 1  from the spinal cord toolbox.

Input:
    -labels: The directory containing the labels to register.
    -images: The directory containing the super-resolved images.
    -output: The directory to save the registered labels.

Output:
    None

Example:
    python register_labels_to_superres.py --labels /path/to/labels --images /path/to/superres_images --output /path/to/output

Author: Pierre-Louis Benveniste
"""
import argparse
import os
from pathlib import Path
from tqdm import tqdm
import shutil


def parse_args():
    parser = argparse.ArgumentParser(description="Register labels to super-resolved images.")
    parser.add_argument("--labels", type=str, required=True, help="Directory containing the labels to register.")
    parser.add_argument("--images", type=str, required=True, help="Directory containing the super-resolved images.")
    parser.add_argument("--output", type=str, required=True, help="Directory to save the registered labels.")
    return parser.parse_args()


def main():
    # Parse arguments
    args = parse_args()
    labels_dir = args.labels
    images_dir = args.images
    output_dir = args.output

    # Build the output_dir if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    # List all .nii.gz files in the labels directory
    label_files = list(Path(labels_dir).rglob("*.nii.gz"))
    label_files = sorted(label_files)
    label_files = [str(f) for f in label_files]

    # For each label file, find the corresponding super-resolved image and register
    for label_file in tqdm(label_files):
        # Extract the base name to find the corresponding image
        image_name = label_file.split("/")[-1].replace(".nii.gz", "_0000.nii.gz")
        image_file = Path(images_dir) / image_name

        if not image_file.exists():
            print(f"Warning: Corresponding image for {label_file} not found. Skipping.")
            break

        # Build a temporary folder to store intermediate results
        temp_folder = Path(output_dir) / "temp"
        os.makedirs(temp_folder, exist_ok=True)
        
        output_temp_file = temp_folder / Path(label_file).name
        output_label_file = Path(output_dir) / Path(label_file).name


        # Run the registration command
        command = f"sct_register_multimodal -i {label_file} -d {image_file} -o {output_temp_file} -identity 1 -x nn"
        assert os.system(command) == 0, f"Error in registration for {label_file}"

        # Move the registered label to the output directory
        assert os.system(f"mv {output_temp_file} {output_label_file}") == 0, f"Error moving file {output_temp_file} to {output_label_file}"
        
        # Clean up the temporary folder
        shutil.rmtree(temp_folder)

    return None


if __name__ == "__main__":
    main()