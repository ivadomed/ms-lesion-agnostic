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
import logging
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path

import glob
import json
import ruamel.yaml

from data_utils import nifti_to_png, labels_from_nifti

logging.basicConfig(filename='pre_process_warning.log', level=logging.WARNING, filemode='w')

def process_scan(nii_volume:str, output_dir:Path, database:Path, set_name:str):
    """
    For every slice in input scan:
        - Extracts bounding boxes from label file
        - Saves bounding box coordinates in .txt file
        - Saves image slice as .png file 

    Output files are saved in the specified set's folder (test, train or val) within the output_dir.

    Args:
        nii_volume (str): Name of the scan, including session number and contrast type
            example format --> sub-cal056_ses-M12_STIR
        output_dir (Path): Path to the YOLO dataset folder
        database (Path): Path to the BIDS database
        set_name (str): Dataset name -- must be one of ["train", "test", "val"]
    """

    patient = nii_volume.split("_")[0]
    ses = nii_volume.split("_")[1]

    print(f"Processing scan {patient}")

    # 1- get bounding boxes from segmentation and save to txt file
    # Check if scan has already been processed
    label_pattern = output_dir / "labels"/ set_name/(patient+"*")
    matching_slices = glob.glob(str(label_pattern))

    if matching_slices == []: # if no slices are found, process scan
        lesion_nii_path = database/ "derivatives"/ "labels"/ patient/ ses/ "anat"/ (nii_volume+"_lesion-manual.nii.gz")
        labels_from_nifti(lesion_nii_path, output_dir / "labels"/ set_name)

    # 2- save spinal cord slices as pngs
    # Check if scan has already been processed
    img_pattern = output_dir / "images"/ set_name/(patient+"*")
    matching_slices = glob.glob(str(img_pattern))

    if matching_slices == []: # if no slices are found, process scan
        image_nii_path = database/ patient/ ses/ "anat"/ (nii_volume+".nii.gz")
        spinal_cord_nii_path = database/ "derivatives"/ "labels"/ patient/ ses/ "anat"/ (nii_volume+"_seg-manual.nii.gz")
        nifti_to_png(image_nii_path, output_dir / "images"/ set_name, spinal_cord_nii_path)

    # Check that all txt files have a corresponding png
    # This was implemented after noticing that some sc segmentations were blank
    list_of_img_slices = [Path(file).name.replace(".png","") for file in glob.glob(str(img_pattern))]

    for file in glob.glob(str(label_pattern)):
        filename = Path(file).name.replace(".txt","")

        try:
            assert filename in list_of_img_slices
        except AssertionError as e:
            # If a txt file has no corresponding png, all slices from that volume are saved
            # and a warning is logged
            logging.warning(f"{e}: {filename} has a segmentation file, but no corresponding image. Saving all slices")
            parts = filename.split('_')
            nii_name = '_'.join(parts[:-1])

            image_nii_path = database/ filename.split("_")[0]/ ses/ "anat"/ (nii_name+".nii.gz")
            nifti_to_png(image_nii_path, output_dir / "images"/ set_name) #save all slices


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
    dir_name = args.output_dir.name

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


    # Create yml file
    yml_str = f"""\
    path: "{dir_name}"
    train: "images/train"
    val: "images/val"
    test: "images/test"

    nc: 1
    names: ["lesion"]
    """

    yaml = ruamel.yaml.YAML(pure=True)
    yaml.preserve_quotes = True
    data = yaml.load(yml_str)

    yaml.dump(data, args.output_dir/(dir_name+".yml"))


if __name__ == "__main__":
    _main()
