"""
This script removes lesions outside of the spinal cord.

Input:
    --pred-folder: Path to the folder containing the lesion segmentation predictions.
    --sc-seg-folder: Path to the folder containing the spinal cord segmentation masks.
    --conversion-dict: Path to the JSON file containing the conversion dictionary for raw image to nnUNet images.
    --training: to indicate that we working on the training set
    --output: Path to the output folder where the masked lesion predictions will be saved.
Output:
    None

Author: Pierre-Louis Benveniste
"""
import json
import os
from image import Image, get_orientation
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Remove lesions outside of the spinal cord")
    parser.add_argument("--pred-folder", type=str, required=True, help="Path to the folder containing the lesion segmentation predictions.")
    parser.add_argument("--sc-seg-folder", type=str, required=True, help="Path to the folder containing the spinal cord segmentation masks.")
    parser.add_argument("--conversion-dict", type=str, required=True, help="Path to the JSON file containing the conversion dictionary for raw image to nnUNet images.")
    parser.add_argument("--training", action="store_true", help="Indicate that we are working on the training set.")
    parser.add_argument("--output", type=str, required=True, help="Path to the output folder where the masked lesion predictions will be saved.")
    return parser.parse_args()


def main():

    args = parse_args()
    pred_folder = args.pred_folder
    sc_seg_folder = args.sc_seg_folder
    conversion_dict = args.conversion_dict
    output_folder = args.output

    # Build the output folder if it does not exist
    os.makedirs(output_folder, exist_ok=True)

    # List all predictions
    list_pred = sorted(list(Path(pred_folder).rglob("*.nii.gz")))

    # Load the conversion dictionary
    with open(conversion_dict, "r") as f:
        conversion_dict = json.load(f)
    list_images = list(conversion_dict.keys())
    list_images = [img for img in list_images if "/derivatives/" not in str(img)]
    # Keep training or testing files
    if args.training:
        list_images = [img for img in list_images if "imagesTr" in conversion_dict[str(img)]]
    else:
        list_images = [img for img in list_images if "imagesTs" in conversion_dict[str(img)]]
    list_images_nnunet = [conversion_dict[str(img)] for img in list_images]
    list_images_nnunet = [Path(img).name for img in list_images_nnunet]
    list_images_nnunet = [str(file).replace("_0000","") for file in list_images_nnunet]

    # List the spinal cord segmentation masks
    list_sc_seg = sorted(list(Path(sc_seg_folder).rglob("*.nii.gz")))
    list_sc_seg = [str(sc_seg) for sc_seg in list_sc_seg if "_dilated" in str(sc_seg)]

    # Initialize a list to store the files where some lesions were removed
    list_files_with_removed_lesions = []

    # iterate over the images
    for pred in tqdm(list_pred):
        # Load the prediction
        pred_image = Image(str(pred))
        # We get the corresponding sc_seg_mask
        pred_index = list_images_nnunet.index(Path(pred).name) # In practice it corresponds to the index of the iteration
        # Get the corresponding spinal cord segmentation mask
        sc_seg_mask_dilated = Path(list_images[pred_index]).name.replace(".nii.gz", "_seg-manual_dilated.nii.gz")
        found_corresponding_sc_seg = [sc_seg for sc_seg in list_sc_seg if sc_seg_mask_dilated in sc_seg]
        assert len(found_corresponding_sc_seg) == 1, f"Found {len(found_corresponding_sc_seg)} spinal cord segmentation masks for {sc_seg_mask_dilated}. Expected 1."
        sc_seg_mask_dilated_file = found_corresponding_sc_seg[0]
        # Build a temp folder in the output folder
        temp_folder = Path(output_folder) / "temp"
        temp_folder.mkdir(parents=True, exist_ok=True)
        # Change the orientation of the sc_seg_mask_dilated to match the pred
        orientation_pred = get_orientation(pred_image)
        assert os.system(f"sct_image -i {sc_seg_mask_dilated_file} -setorient {orientation_pred} -o {temp_folder}/sc_seg_dilated.nii.gz") == 0
        # We remove the lesions outside of the spinal cord
        masked_lesion_pred = os.path.join(output_folder, Path(pred).name)
        assert os.system(f"sct_maths -i {pred} -mul {temp_folder}/sc_seg_dilated.nii.gz -o {masked_lesion_pred}")==0     

        # Now we want to store the files where some lesions were removed (meaning where data are different)
        data_pred = pred_image.data
        data_pred_masked = Image(masked_lesion_pred).data
        if not np.array_equal(data_pred, data_pred_masked):
            list_files_with_removed_lesions.append(masked_lesion_pred)

    # Save the list of files where some lesions were removed to a txt file
    with open(os.path.join(output_folder, "files_with_removed_lesions.txt"), "w") as f:
        for file in list_files_with_removed_lesions:
            f.write(f"{file}\n")
    
    return None


if __name__ == "__main__":
    main()