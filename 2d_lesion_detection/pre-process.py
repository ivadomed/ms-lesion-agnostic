"""
Generates a YOLO format dataset from a BIDS database.
Only processes scans in the given json list.

Here is how the json list should be formatted:
    {"train": ["sub-cal056_ses-M12_STIR",
                "sub-edm011_ses-M0_PSIR",
                "sub-cal072_ses-M0_STIR"],
    "val": ["sub-cal157_ses-M12_STIR"],
    "test": ["sub-edm076_ses-M0_PSIR"]}

    
The YOLO dataset is formatted as follows:
    dataset/
    │
    ├── images/
    │   ├── train/
    │   │   ├── sub-cal056_ses-M12_STIR_0.png
    │   │   ├── sub-cal056_ses-M12_STIR_1.png
    │   │   └── ...
    │   ├── val/
    │   │   ├── sub-tor006_ses-M12_PSIR_0.png
    │   │   └── ...
    │   └── test/
    │       ├── sub-tor007_ses-M12_PSIR_0.png
    │       └── ...
    │
    ├── labels/
    │   ├── train/
    │   │   ├── sub-cal056_ses-M12_STIR_0.txt
    │   │   ├── sub-cal056_ses-M12_STIR_1.txt
    │   │   └── ...
    │   ├── val/
    │   │   ├── sub-tor006_ses-M12_PSIR_0.txt
    │   │   └── ...
    │   └── test/
    │       ├── sub-tor007_ses-M12_PSIR_0.txt
    │       └── ...
    │
    └── data.yaml

"""

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import glob
import json
from pathlib import Path
from data_utils import nifti_to_png, labels_from_nifti


def process_scan(nii_volume:str, output_dir:Path, database:Path, set:str):
    """
    For every slice in input scan:
        - Extracts bounding boxes from label file
        - Saves bounding box coordinates in .txt file
        - Saves image slice as .png file 

    Output files are saved in the specified set's folder within the output_dir.

    Args:
        nii_volume (str): Name of the scan, including session number and contrast type
            example format --> sub-cal056_ses-M12_STIR
        output_dir (Path): Path to the YOLO dataset folder
        database (Path): Path to the BIDS database
        set (str): Dataset name -- must be one of ["train", "test", "val"]
    """

    patient = nii_volume.split("_")[0]
    ses = nii_volume.split("_")[1]

    print(f"Processing scan {patient}")

    # 1- get bounding boxes from segmentation and save to txt file
    # Check if scan has already been processed
    pattern = output_dir / "labels"/ set/(patient+"*")
    matching_slices = glob.glob(str(pattern))
    if matching_slices == []:
        lesion_nii_path = database/ "derivatives"/ "labels"/ patient/ ses/ "anat"/ (nii_volume+"_lesion-manual.nii.gz")
        labels_from_nifti(lesion_nii_path, output_dir / "labels"/ set)

    # 2- save spinal cord slices as pngs
    # Check if scan has already been processed
    pattern = output_dir / "images"/ set/(patient+"*")
    matching_slices = glob.glob(str(pattern))
    if matching_slices == []:
        image_nii_path = database/ patient/ ses/ "anat"/ (nii_volume+".nii.gz")
        spinal_cord_nii_path = database/ "derivatives"/ "labels"/ patient/ ses/ "anat"/ (nii_volume+"_seg-manual.nii.gz")
        nifti_to_png(image_nii_path, output_dir / "images"/ set, spinal_cord_nii_path)


def _main():
    parser = ArgumentParser(
    prog = 'pre-process',
    description = 'Generates YOLO format dataset from a list of scans and a BIDS database.',
    formatter_class = ArgumentDefaultsHelpFormatter)
    parser.add_argument('-j', '--json-list',
                        required = True,
                        type = Path,
                        help = 'Path to json list of images to process')
    parser.add_argument('-d', '--database',
                        required = True,
                        type = Path,
                        help = 'Path to BIDS database')
    parser.add_argument('-o', '--output-dir',
                        required = True,
                        type = Path,
                        help = 'Output directory for YOLO dataset')

    args = parser.parse_args()
    
    with open(args.json_list, "r") as json_file:
        data = json.load(json_file)

    training_list = data["train"]
    validation_list = data["val"]
    test_list = data["test"]

    for volume in training_list:
        process_scan(volume, args.output_dir, args.database, "train")

    for volume in validation_list:
        process_scan(volume, args.output_dir, args.database, "val")

    for volume in test_list:
        process_scan(volume, args.output_dir, args.database, "test")

    # Create yaml file
    # TODO
    


if __name__ == "__main__":
    _main()