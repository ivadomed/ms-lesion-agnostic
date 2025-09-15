"""
This file is used to segment the spinal cord in the MSD dataset on the train and test set
It checks if the SC file already exists and copies it. If not, it runs the segmentation with SCT.
It also dilates the spinal cord mask by 2 mm. 

Input:
    -i: the path to the MSD dataset
    -o: the output directory where the segmented spinal cord will be saved

Output:
    None

Example usage:
    python segment_sc.py -i /path/to/msd_dataset -o /path/to/output_directory

Author: Pierre-Louis Benveniste
"""
import os
from pathlib import Path
import argparse
import json
from tqdm import tqdm
import shutil
from image import Image, get_dimension


def parse_args():
    parser = argparse.ArgumentParser(description="Segment the spinal cord in the MSD dataset")
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to the MSD dataset")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output directory for the segmented spinal cord")
    return parser.parse_args()


def dilate_sc_seg(sc_seg_path: str, dilation_mm=2):
    """
    Dilates the spinal cord segmentation mask by a specified number of millimeters.
    
    Args:
        sc_seg_path (Path): Path to the spinal cord segmentation mask.
        dilation_mm (int): Number of millimeters to dilate the mask.
    """
    # Load the spinal cord segmentation mask
    sc_seg = Image(sc_seg_path)

    # We get the orientation of the image
    image_orientation = sc_seg.orientation
    # print("Seg orientation:", image_orientation)

    # We check which is the S-I direction
    s_i_direction = None
    if 'S' in image_orientation:
        s_i_direction = image_orientation.index('S')
    if 'I' in image_orientation:
        s_i_direction = image_orientation.index('I')

    # We also want the R-L direction
    r_l_direction = None
    if 'R' in image_orientation:
        r_l_direction = image_orientation.index('R')
    if 'L' in image_orientation:
        r_l_direction = image_orientation.index('L')
    # print("S-I direction:", s_i_direction)
    # print("R-L direction:", r_l_direction)

    # Finally we also want the A-P direction
    a_p_direction = None
    if 'A' in image_orientation:
        a_p_direction = image_orientation.index('A')
    if 'P' in image_orientation:
        a_p_direction = image_orientation.index('P')
    # print("A-P direction:", a_p_direction)

    # Now we want to identify the number of voxels to dilate in the R-L axis (it should be close to 2mm)
    resolution_r_l = get_dimension(Image(sc_seg))[4+ r_l_direction]
    # print("Resolution R-L:", resolution_r_l)
    vox_dilate_r_l = max(1,int(dilation_mm / resolution_r_l))  # 2mm dilation in the R-L axis
    # print("Voxels to dilate in R-L axis:", vox_dilate_r_l)

    # same for the A-P axis
    resolution_a_p = get_dimension(Image(sc_seg))[4+ a_p_direction]
    # print("Resolution A-P:", resolution_a_p)
    vox_dilate_a_p = max(1,int(dilation_mm / resolution_a_p)) # 2mm dilation in the A-P axis
    # print("Voxels to dilate in A-P axis:", vox_dilate_a_p)

    # Build the output path
    sc_seg_dilated_path = sc_seg_path.replace(".nii.gz", "_dilated.nii.gz")

    # We dilate the SC mask around the axial plane (S-I direction) with a radius of 2 mm computed on the R-L axis
    assert os.system(f"sct_maths -i {sc_seg_path} -dilate {vox_dilate_r_l} -shape disk -dim {s_i_direction} -o {sc_seg_dilated_path}")==0
    # We dilate the SC mask around the sagittal plane (R-L direction) with a radius of 2 mm computed on the A-P axis
    assert os.system(f"sct_maths -i {sc_seg_dilated_path} -dilate {vox_dilate_a_p} -shape disk -dim {r_l_direction} -o {sc_seg_dilated_path}")==0

    return None


def main():

    # Parse arguments
    args = parse_args()
    msd_file_path = Path(args.input)
    output_dir = Path(args.output)

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the MSD dataset
    with open(msd_file_path, 'r') as f:
        msd_data_split = json.load(f)

    # Get the files to segment
    files_to_segment = msd_data_split['train'] + msd_data_split['validation'] + msd_data_split['test']

    # Initialize a counter for existing spinal cord segmentations
    count_existing = 0
    for file in tqdm(files_to_segment):
        # Check if the SC seg exist
        ## First we build the path to the sc_seg file
        img = file['image']
        img_parts = Path(img).parts
        sc_seg_path = Path(*img_parts[:7]) / "derivatives" / "labels" / str(Path(*img_parts[7:])).replace(".nii.gz", "_seg-manual.nii.gz")
        # Build the output path for the spinal cord segmentation
        output_sc_seg_path = output_dir / file['site'] / sc_seg_path.name
        output_sc_seg_path.parent.mkdir(parents=True, exist_ok=True)
        ## Check if sc seg exists
        if sc_seg_path.exists():
            count_existing+=1
            # Copy the existing segmentation to the output directory
            shutil.copy(sc_seg_path, output_sc_seg_path)
        else:
            # If the segmentation does not exist, run SCT to segment the spinal cord
            print(f"Segmenting spinal cord for {img}...")
            assert os.system(f"sct_deepseg spinalcord -i {img} -o {output_sc_seg_path}") == 0, "SCT segmentation failed"
        # Dilate the spinal cord segmentation by 2 mm
        dilate_sc_seg(str(output_sc_seg_path))

    print(f"Found {count_existing} existing spinal cord segmentations.")


if __name__ == "__main__":
    main()