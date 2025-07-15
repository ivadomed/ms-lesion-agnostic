"""
This script is used to remove small lesions from a set of images.
Images are removed based on their size from 1mm^3 to 21 mm^3 by steps of 2 mm^3

Input:
    --pred-folder: Path to the folder containing the binarized lesion segmentation predictions.
    --output: Path to the output folder where the lesion segmentations will be saved after removing small lesions.

Output:
    None

Example:
    python 5_rmv_small_lesions.py --pred-folder /path/to/predictions --output /path/to/output

Author: Pierre-Louis Benveniste
"""
import argparse
import os
from pathlib import Path
from tqdm import tqdm
from image import Image, get_dimension
from scipy import ndimage
import numpy as np
import nibabel as nib


def parse_args():
    parser = argparse.ArgumentParser(description="Remove small lesions from lesion segmentations.")
    parser.add_argument("--pred-folder", type=str, required=True, help="Path to the folder containing the binarized lesion segmentation predictions.")
    parser.add_argument("--output", type=str, required=True, help="Path to the output folder where the lesion segmentations will be saved after removing small lesions.")
    return parser.parse_args()


def main():

    # Parse command line arguments
    args = parse_args()
    pred_folder = args.pred_folder
    output_folder = args.output

    # Build the output folder if it does not exist
    os.makedirs(output_folder, exist_ok=True)

    # List all predictions
    list_pred = sorted(list(Path(pred_folder).rglob("*.nii.gz")))

    # Define the size thresholds in mm^3
    volume_thresholds = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]

    # Initialize a list to store lesion sizes
    lesion_sizes = []

    # For each prediction, remove small lesions and save the result
    for pred in tqdm(list_pred):
        for min_volume in volume_thresholds:

            # For each volume, we build an output folder
            output_folder_volume = os.path.join(output_folder, f"rmv_lesions_{min_volume}mm3")
            os.makedirs(output_folder_volume, exist_ok=True)

            pred_image = Image(str(pred))
            pred_image_data = pred_image.data
            # Get voxel size: get dimension output is nx, ny, nz, nt, px, py, pz, pt
            voxel_size = get_dimension(pred_image)[4]*get_dimension(pred_image)[5]*get_dimension(pred_image)[6]

            # In the lesion mask we enumerate the lesions by using connected components
            instances, nb_labels = ndimage.label(pred_image_data)
            ### Now we want to split instances into individual masks for each lesion
            individual_instances = np.zeros((nb_labels, *pred_image_data.shape), dtype=np.float32)
            for i in range(1, nb_labels+1):
                instance_i = np.zeros_like(pred_image_data)
                instance_i[instances == i] = 1
                individual_instances[i-1] = instance_i
            ### For each individual instance, we check the number of voxels in the lesion
            for i in range(1, nb_labels+1):
                if min_volume ==0:
                    lesion_sizes.append(np.sum(individual_instances[i-1])*voxel_size)
                # If the lesion is smaller than the minimum volume, we remove it
                if np.sum(individual_instances[i-1]*voxel_size) < min_volume:
                    mask_data = mask_data * (1 - individual_instances[i-1])
            ### Save the modified T2w lesion mask
            modified_mask_path = os.path.join(output_folder_volume, Path(pred).name)
            nib.save(nib.Nifti1Image(mask_data, nib.load(str(pred)).affine), modified_mask_path)

    # Save the lesion sizes in the main output_folder
    with open(os.path.join(output_folder, "lesion_sizes.txt"), "w") as f:
        for size in lesion_sizes:
            f.write(f"{size}\n")

    return None


if __name__ == "__main__":
    main()