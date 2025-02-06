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


def parse_args():
    parser = argparse.ArgumentParser(description='Generate QC for review')
    parser.add_argument('--path-img', type=str, required=True, help='Path to the images')
    parser.add_argument('--path-seg', type=str, required=True, help='Path to the segmentations')
    parser.add_argument('--path-out', type=str, required=True, help='Path to the output folder')
    return parser.parse_args()


def main():
    # Parse arguments
    args = parse_args()
    path_img = args.path_img
    path_seg = args.path_seg
    path_out = args.path_out

    path_sc_seg = os.path.join(path_out, 'sc_segs')

    # Create output folder
    if not os.path.exists(path_out):
        os.makedirs(path_out)
    if not os.path.exists(path_sc_seg):
        os.makedirs(path_sc_seg)
    
    # Iterate over the segmentations rglob
    list_segmentations = list(Path(path_seg).rglob('*.nii.gz'))
    # sort the list by name
    list_segmentations.sort()
    for seg in list_segmentations:
        # Get the corresponding image
        img = seg.name.replace('_labelA', '').replace('_labelB', '')
        img = os.path.join(path_img, img)

        # For each image we segment the spinal cord
        sc_seg = os.path.join(path_sc_seg, Path(img).name)
        if not os.path.exists(sc_seg):
            os.system(f'sct_deepseg -i {img} -task seg_sc_contrast_agnostic -o {sc_seg}')

        # Determine the plane
        if 'sag' in img or 'PSIR' in img or 'STIR' in img:
            plane = 'sagittal'
        else :
            plane = 'axial'

        # Generate the QC
        os.system(f'sct_qc -i {img} -s {sc_seg} -d {seg} -qc {path_out} -p sct_deepseg_lesion -plane {plane}')

    print('QC generated')

    return None

if __name__ == '__main__':
    main()