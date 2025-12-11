"""
This script is used to crop images to have exactly the same visible portion of the spinal cord.

Input:
    -msd: Path to the msd dataset

Output:
    -discs: directory containing the labeling of the discs
    -o: output folder to save the cropped images

Author: Pierre-Louis Benveniste
"""
import os
import argparse
import json
import nibabel as nib
import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Crop images to have exactly the same visible portion of the spinal cord.")
    parser.add_argument("-msd", type=str, required=True, help="Path to the msd dataset")
    parser.add_argument("-discs", type=str, required=True, help="Directory containing the labeling of the discs")
    parser.add_argument("-o", type=str, required=True, help="Output folder to save the cropped images")
    return parser.parse_args()


def label_discs(image_path, output_file):
    """
    Dummy function to label discs in the image.
    In a real scenario, this function would implement the logic to label the discs.
    """
    temp_folder = os.path.join(os.path.dirname(output_file), 'temp')
    os.makedirs(temp_folder, exist_ok=True)
    temp_output = os.path.join(temp_folder, 'output.nii.gz')
    assert os.system(f"SCT_USE_GPU=1 sct_deepseg spine -i {image_path} -o {temp_output}") == 0, "Failed to copy image for disc labeling."
    # Move the predicted disc level file to the output file
    predicted_disc_file = temp_output.replace('.nii.gz', '_totalspineseg_discs.nii.gz')
    os.rename(predicted_disc_file, output_file)
    # Same for the json files
    predicted_json_file = temp_output.replace('.nii.gz', '_totalspineseg_discs.json')
    output_json_file = output_file.replace('.nii.gz', '.json')
    os.rename(predicted_json_file, output_json_file)
    # Remove the temp folder
    assert os.system(f"rm -r {temp_folder}") == 0, "Failed to remove temporary folder."
    return None


def get_common_discs(list_discs_paths):
    """
    Dummy function to get common discs from a list of disc paths.
    In a real scenario, this function would implement the logic to find common discs.
    """
    list_uniques = []
    # For each disc path, we load the data and get the unique labels
    for disc_path in list_discs_paths:
        data  = nib.load(disc_path).get_fdata()
        uniques = np.unique(data)
        uniques = uniques[uniques != 0]  # Remove background
        list_uniques.append(uniques)
    # Build intersection of all unique labels
    intersection = set(list_uniques[0])
    for uniques in list_uniques[1:]:
        intersection = intersection.intersection(set(uniques))
    intersection = sorted(list(intersection))
    # Now we find the common min and max discs across all images
    common_min =min(intersection)
    common_max = max(intersection)
    return common_min, common_max


def crop_image(image_path, disc_path, min_disc, max_disc, output_path):
    """
    Function to crop an image based on disc labels.

    Arguments:
        image_path: Path to the input image
        disc_path: Path to the disc labeling image
        min_disc: Minimum disc label to keep
        max_disc: Maximum disc label to keep
        output_path: Path to save the cropped image
    Output:
        None
    """
    # Load the image and disc data
    img_nib = nib.load(image_path)
    img_data = img_nib.get_fdata()
    disc_nib = nib.load(disc_path)
    disc_data = disc_nib.get_fdata()
    # Get the index of the IS axis
    orientation = nib.aff2axcodes(img_nib.affine)
    index_IS = orientation.index('S') if 'S' in orientation else orientation.index('I')
    # Get the coordinates of the min and max discs
    min_mask_coord = np.where(disc_data == min_disc)[index_IS][0]
    max_mask_coord = np.where(disc_data == max_disc)[index_IS][0]
    min_mask_coord, max_mask_coord = np.min([min_mask_coord, max_mask_coord]), np.max([min_mask_coord, max_mask_coord])
    # Crop the image
    coord = ['x', 'y', 'z']
    assert os.system(f"sct_crop_image -i {image_path} -o {output_path} -{coord[index_IS]}min {min_mask_coord} -{coord[index_IS]}max {max_mask_coord}") == 0, "Failed to crop image."

    return None


def main():
    args = parse_args()

    # Extract arguments
    msd_path = args.msd
    discs_path = args.discs
    output_path = args.o

    # Build output folders
    os.makedirs(discs_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)

    # Load the msd json file
    with open(msd_path, 'r') as f:
        msd_json = json.load(f)
    images = msd_json['data']

    # Skipped subjects/sessions with no FoV overlap
    skipped = []

    # Iterate over the subjects
    for subject in tqdm(images):
        # Iterate over sessions:
        for session in images[subject]:
            # Build output path in both discs and output folders
            output_disc_folder = os.path.join(discs_path, subject, session,'anat')
            output_cropped_folder = os.path.join(output_path, subject, session, 'anat')
            os.makedirs(output_disc_folder, exist_ok=True)
            os.makedirs(output_cropped_folder, exist_ok=True)
            list_discs_paths = []
            # Iterate over the images:
            for img in images[subject][session]['images']:
                # For that image we label the discs
                output_disc_path = os.path.join(output_disc_folder, img.split('/')[-1].replace('.nii.gz', '_discs.nii.gz'))
                label_discs(img, output_disc_path)
                list_discs_paths.append(output_disc_path)

            # We now identify the common min and max discs present in all images
            min, max = get_common_discs(list_discs_paths)
            ## If min == max, we skip the cropping
            if min == max:
                skipped.append(f"{subject}_{session}")
                print(f"Warning: For subject {subject} session {session}, min and max discs are the same ({min}). Skipping cropping.")
                continue

            # For each image we crop it to the common min and max discs
            for i, img in enumerate(images[subject][session]['images']):
                # Crop the image 
                output_cropped_path = os.path.join(output_cropped_folder, img.split('/')[-1])
                crop_image(img, list_discs_paths[i], min, max, output_cropped_path)
    print("Cropping completed successfully.")
    print(f"Skipped {len(skipped)} subject-sessions due to no FoV overlap: {skipped}")


if __name__ == "__main__":
    main()