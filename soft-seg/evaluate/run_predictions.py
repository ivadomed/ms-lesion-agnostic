"""
This code runs the predictions for the lesion segmentation model on the cropped images.
It outputs binary lesion masks in one folder and soft segmentation maps in another folder.

Input:
    -msd: Path to the msd dataset
    -i: input folder containing the cropped images
    -o_bin: output folder to save the binary lesion masks
    -o_soft: output folder to save the soft segmentation maps

Output:
    None

Author: Pierre-Louis Benveniste
"""
import os
import argparse
from tqdm import tqdm
import json


def parse_args():
    parser = argparse.ArgumentParser(description="Run lesion segmentation predictions on cropped images.")
    parser.add_argument("-msd", type=str, required=True, help="Path to the msd dataset")
    parser.add_argument("-i", type=str, required=True, help="Input folder containing the cropped images")
    parser.add_argument("-o-bin", type=str, required=True, help="Output folder to save the binary lesion masks")
    parser.add_argument("-o-soft", type=str, required=True, help="Output folder to save the soft segmentation maps")
    return parser.parse_args()


def main():
    # Parsing arguments
    args = parse_args()
    msd_path = args.msd
    input_folder = args.i
    output_folder_bin = args.o_bin
    output_folder_soft = args.o_soft

    # Build output folders if they do not exist
    os.makedirs(output_folder_bin, exist_ok=True)
    os.makedirs(output_folder_soft, exist_ok=True)

    # Open the msd dataset to get the list of images
    # Load the msd json file
    with open(msd_path, 'r') as f:
        msd_json = json.load(f)
    images = msd_json['data']

    for sub in tqdm(images):
        # iterate over sessions
        for session in images[sub]:
            # Iterate over the images
            for img in images[sub][session]['images']:
                # Build path to cropped image
                cropped_image_path = os.path.join(input_folder, sub, session, 'anat', img.split('/')[-1])
                # Build output paths
                output_bin_path = os.path.join(output_folder_bin, sub, session, 'anat', img.split('/')[-1].replace('.nii.gz', '_lesion-seg-bin.nii.gz'))
                output_soft_path = os.path.join(output_folder_soft, sub, session, 'anat', img.split('/')[-1].replace('.nii.gz', '_lesion-seg-soft.nii.gz'))
                os.makedirs(os.path.dirname(output_bin_path), exist_ok=True)
                os.makedirs(os.path.dirname(output_soft_path), exist_ok=True)
                # Run the prediction command
                ## For binary segmentation
                assert os.system(f"SCT_USE_GPU=1 sct_deepseg lesion_ms -i {cropped_image_path} -o {output_bin_path} -test-time-aug") == 0, f"Failed to run lesion segmentation on {cropped_image_path}"
                ## For soft segmentation
                assert os.system(f"SCT_USE_GPU=1 sct_deepseg lesion_ms -i {cropped_image_path} -o {output_soft_path} -test-time-aug -soft-ms-lesion") == 0, f"Failed to run lesion segmentation on {cropped_image_path}"
    return None


if __name__ == "__main__":
    main()