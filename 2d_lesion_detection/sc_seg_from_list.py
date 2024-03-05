"""
Tool for generating missing spinal cord segmentations for a list of scans.
For every scan in the given list, checks if a sc segmentation file exists in the BIDS database.
If the file is missing, generates it using the spinal cord toolbox.

Make sure to activate the sct env before running:
source ~/sct_6.2/python/envs/venv_sct/bin/activate
"""

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import json
from pathlib import Path
import subprocess


def segment_spinal_cord(image_path:str, labels_path:str, seg_name:str):
    """
    Uses the spinal cord toolbox to generate a spinal cord segmentation file.

    Args:
        image_path (str): Path to the image containing the spinal cord
        labels_path (str): Path to the corresponding labels folder. 
                           This is where the segmentation file will be saved.
        seg_name (str): Name of the segmentation file
    """

    command = [
    "sct_deepseg_sc",
    "-i", image_path,
    "-c", "t2",
    "-ofolder", labels_path,
    "-o", seg_name
    ]
    subprocess.run(command, check=True)


def main():
    parser = ArgumentParser(
    prog = 'sc_seg_from_list',
    description = 'Generate missing spinal cord segmentations for the given list of scans.',
    formatter_class = ArgumentDefaultsHelpFormatter)
    parser.add_argument('-j', '--json-list',
                        required = True,
                        type = Path,
                        help = 'path to json list of scans to process')
    parser.add_argument('-d', '--database',
                        required = True,
                        type = Path,
                        help = 'path to BIDS database')
    
    args = parser.parse_args()

    with open(args.json_list, "r") as json_file:
        data = json.load(json_file)

    scan_list = [item for sublist in data.values() for item in sublist]

    error_list = []
    for scan in scan_list:
        patient = scan.split("_")[0]
        ses = scan.split("_")[1]

        spinal_seg_nii_path = args.database/ "derivatives"/ "labels"/ patient/ ses/ "anat"/ (scan+"_seg-manual.nii.gz")

        if not spinal_seg_nii_path.exists():
            image_path = args.database/ patient/ ses/ "anat"/ (scan+".nii.gz")
            labels_path = spinal_seg_nii_path.parent
            seg_name = spinal_seg_nii_path.name

            # When the segmentation fails, add name to error_list and skip it
            try:
                segment_spinal_cord(image_path, labels_path, seg_name)
            except Exception as e:
                print(f"Error processing scan {scan}: {e}")
                error_list.append(scan)

    print(f"All done! Here are the scans that couldn't be processed: {error_list}")



if __name__ == "__main__":
    main()
