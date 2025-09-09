"""
This script generates the QC for the 20 images which will be reviewed by the experts.

Input: 
    --path-img: path to the images
    --path-seg: path to the segmentations
    --path-out: path to the output folder

Output:
    None

Example:
    python generate_qc_for_review.py --path-img /path/to/images --path-seg /path/to/segmentations --path-out /path/to/output

Author: Pierre-Louis Benveniste
"""
import os
import argparse
from pathlib import Path
import json
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Generate QC for review')
    parser.add_argument('--path-img', type=str, required=True, help='Path to the images')
    parser.add_argument('--path-seg', type=str, required=True, help='Path to the segmentations')
    parser.add_argument('--path-out', type=str, required=True, help='Path to the output folder')
    parser.add_argument('--path-sc-seg', type=str, required=True, help='Path to the spinal cord segmentations (optional)')
    parser.add_argument('--path-msd-data', type=str, required=True, help='Path to the MSD data (optional)')
    return parser.parse_args()


def main():
    # Parse arguments
    args = parse_args()
    path_img = args.path_img
    path_seg = args.path_seg
    path_out = args.path_out
    path_sc_seg = args.path_sc_seg
    path_msd_data = args.path_msd_data

    # Create output folder
    if not os.path.exists(path_out):
        os.makedirs(path_out)
    
    # Open the msd dataset:
    # Load the data json file
    with open(path_msd_data, 'r') as f:
        jsondata = json.load(f)
    msdData = jsondata['train'] + jsondata['validation'] + jsondata['test'] + jsondata['externalValidation']
    
    # Iterate over the segmentations rglob
    list_segmentations = list(Path(path_seg).rglob('*.nii.gz'))
    # sort the list by name
    list_segmentations.sort()
    for seg in tqdm(list_segmentations):
        # Get the corresponding image
        img = seg.name.replace('_labelA', '').replace('_labelB', '')
        img = os.path.join(path_img, img)

        # For each image we find the corresponding segmentation
        ## We need to find the site of the image
        sub = [data for data in msdData if Path(data["image"]).name == Path(img).name][0]
        sc_seg = os.path.join(path_sc_seg, sub['site'], Path(img).name.replace('.nii.gz', '_seg-manual.nii.gz'))
        # Check if the image exists
        if not os.path.exists(img):
            print(f'Image {img} does not exist, skipping...')
            break

        # Determine the plane
        if 'sag' in sub["acquisition"]:
            plane = 'sagittal'
        else :
            plane = 'axial'

        # # Generate the QC
        assert os.system(f'sct_qc -i {img} -s {sc_seg} -d {seg} -qc {path_out} -p sct_deepseg_lesion -plane {plane}') == 0

    print('QC generated')

    return None

if __name__ == '__main__':
    main()