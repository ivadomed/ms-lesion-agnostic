"""
This file creates the MSD-style JSON datalist to train an nnunet model using monai. 
The datasets used are CanProCo, Bavaria-quebec, basel and sct-testing-large.

Arguments:
    -pd, --path-data: Path to the data set directory
    -pj, --path-joblib: Path to joblib file from ivadomed containing the dataset splits.
    -po, --path-out: Path to the output directory where dataset json is saved
    --contrast: Contrast to use for training
    --label-type: Type of labels to use for training
    --seed: Seed for reproducibility

Example:
    python create_msd_data.py -pd /path/dataset -po /path/output 

TO DO: 
    *

Pierre-Louis Benveniste
"""

import os
import json
from tqdm import tqdm
import yaml
import argparse
from loguru import logger
from sklearn.model_selection import train_test_split
from datetime import date
from pathlib import Path
import nibabel as nib
import numpy as np
import skimage


def get_parser():
    """
    Get parser for script create_msd_data.py

    Input:
        None

    Returns:
        parser : argparse object
    """

    parser = argparse.ArgumentParser(description='Code for MSD-style JSON datalist for lesion-agnostic nnunet model training.')

    parser.add_argument('-pd', '--path-data', required=True, type=str, help='Path to the folder containing the datasets')
    parser.add_argument('-po', '--path-out', type=str, help='Path to the output directory where dataset json is saved')
    parser.add_argument('--lesion-only', action='store_true', help='Use only masks which contain some lesions')
    parser.add_argument('--seed', default=42, type=int, help="Seed for reproducibility")

    return parser


def count_lesion(label_file):
    """
    This function takes a label file and counts the number of lesions in it.

    Input:
        label_file : str : Path to the label file
    
    Returns:
        count : int : Number of lesions in the label file
        total_volume : float : Total volume of lesions in the label file
    """

    label = nib.load(label_file)
    label_data = label.get_fdata()

    # get the total volume of the lesions
    total_volume = np.sum(label_data)
    resolution = label.header.get_zooms()
    total_volume = total_volume * np.prod(resolution)

    # get the number of lesions
    _, nb_lesions = skimage.measure.label(label_data, connectivity=2, return_num=True)
    
    return  total_volume, nb_lesions


def main():
    """
    This is the main function of the script.

    Input:
        None
    
    Returns:
        None
    """
    # Get the arguments
    parser = get_parser()
    args = parser.parse_args()

    root = args.path_data
    seed = args.seed

    # Get all subjects
    canproco_path = Path(os.path.join(root, "canproco"))
    basel_path = Path(os.path.join(root, "basel-mp2rage"))
    bavaria_path = Path(os.path.join(root, "bavaria-quebec-spine-ms"))
    sct_testing_path = Path(os.path.join(root, "sct-testing-large"))

    subjects_canproco = list(canproco_path.rglob('*_lesion-manual.nii.gz'))
    subjects_basel = list(basel_path.rglob('*UNIT1.nii.gz'))
    subjects_sct = list(sct_testing_path.rglob('*_lesion-manual.nii.gz'))
    subjects_bavaria = list(bavaria_path.rglob('*T2w.nii.gz'))

    subjects = subjects_canproco + subjects_basel + subjects_sct + subjects_bavaria
    # logger.info(f"Total number of subjects in the root directory: {len(subjects)}")

    # create one json file with 60-20-20 train-val-test split
    train_ratio, val_ratio, test_ratio = 0.6, 0.2, 0.2
    train_subjects, test_subjects = train_test_split(subjects, test_size=test_ratio, random_state=args.seed)
    # Use the training split to further split into training and validation splits
    train_subjects, val_subjects = train_test_split(train_subjects, test_size=val_ratio / (train_ratio + val_ratio),
                                                    random_state=args.seed, )
    # sort the subjects
    train_subjects = sorted(train_subjects)
    val_subjects = sorted(val_subjects)
    test_subjects = sorted(test_subjects)

    # logger.info(f"Number of training subjects: {len(train_subjects)}")
    # logger.info(f"Number of validation subjects: {len(val_subjects)}")
    # logger.info(f"Number of testing subjects: {len(test_subjects)}")

    # dump train/val/test splits into a yaml file
    with open(f"{args.path_out}/data_split_{str(date.today())}_seed{seed}.yaml", 'w') as file:
        yaml.dump({'train': train_subjects, 'val': val_subjects, 'test': test_subjects}, file, indent=2, sort_keys=True)

    # keys to be defined in the dataset_0.json
    params = {}
    params["description"] = "ms-lesion-agnostic"
    params["labels"] = {
        "0": "background",
        "1": "ms-lesion-seg"
        }
    params["license"] = "plb"
    params["modality"] = {
        "0": "MRI"
        }
    params["name"] = "ms-lesion-agnostic"
    params["seed"] = args.seed
    params["reference"] = "NeuroPoly"
    params["tensorImageSize"] = "3D"

    train_subjects_dict = {"train": train_subjects}
    val_subjects_dict = {"validation": val_subjects}
    test_subjects_dict =  {"test": test_subjects}
    all_subjects_list = [train_subjects_dict, val_subjects_dict, test_subjects_dict]

    # iterate through the train/val/test splits and add those which have both image and label
    for subjects_dict in tqdm(all_subjects_list, desc="Iterating through train/val/test splits"):

        for name, subs_list in subjects_dict.items():

            temp_list = []
            for subject_no, subject in enumerate(subs_list):

                temp_data_canproco = {}
                temp_data_basel = {}
                temp_data_sct = {}
                temp_data_bavaria = {}

                # Canproco
                if 'canproco' in str(subject):
                    temp_data_canproco["label"] = str(subject)
                    temp_data_canproco["image"] = str(subject).replace('_lesion-manual.nii.gz', '.nii.gz').replace('derivatives/labels/', '')
                    if os.path.exists(temp_data_canproco["label"]) and os.path.exists(temp_data_canproco["image"]):
                        total_lesion_volume, nb_lesions = count_lesion(temp_data_canproco["label"])
                        temp_data_canproco["total_lesion_volume"] = total_lesion_volume
                        temp_data_canproco["nb_lesions"] = nb_lesions
                        if args.lesion_only and nb_lesions == 0:
                            continue
                        temp_list.append(temp_data_canproco)
                
                # Basel
                elif 'basel-mp2rage' in str(subject):
                    relative_path = subject.relative_to(basel_path).parent
                    temp_data_basel["image"] = str(subject)
                    temp_data_basel["label"] = str(basel_path / 'derivatives' / 'labels' /  relative_path / str(subject).replace('UNIT1.nii.gz', 'UNIT1_desc-rater3_label-lesion_seg.nii.gz'))
                    if os.path.exists(temp_data_basel["label"]) and os.path.exists(temp_data_basel["image"]):
                        total_lesion_volume, nb_lesions = count_lesion(temp_data_basel["label"])
                        temp_data_basel["total_lesion_volume"] = total_lesion_volume
                        temp_data_basel["nb_lesions"] = nb_lesions
                        if args.lesion_only and nb_lesions == 0:
                            continue
                        temp_list.append(temp_data_basel)

                # sct-testing-large
                elif 'sct-testing-large' in str(subject):
                    temp_data_sct["label"] = str(subject)
                    temp_data_sct["image"] = str(subject).replace('_lesion-manual.nii.gz', '.nii.gz').replace('derivatives/labels/', '')
                    if os.path.exists(temp_data_sct["label"]) and os.path.exists(temp_data_sct["image"]):
                        total_lesion_volume, nb_lesions = count_lesion(temp_data_sct["label"])
                        temp_data_sct["total_lesion_volume"] = total_lesion_volume
                        temp_data_sct["nb_lesions"] = nb_lesions
                        if args.lesion_only and nb_lesions == 0:
                            continue
                        temp_list.append(temp_data_sct)
                        

                # Bavaria-quebec
                elif 'bavaria-quebec-spine-ms' in str(subject):
                    relative_path = subject.relative_to(bavaria_path).parent
                    temp_data_bavaria["image"] = str(subject)
                    temp_data_bavaria["label"] = str(bavaria_path / 'derivatives' / 'labels' / relative_path / subject.name.replace('T2w.nii.gz', 'T2w_lesion-manual.nii.gz'))
                    if os.path.exists(temp_data_bavaria["label"]) and os.path.exists(temp_data_bavaria["image"]):
                        total_lesion_volume, nb_lesions = count_lesion(temp_data_bavaria["label"])
                        temp_data_bavaria["total_lesion_volume"] = total_lesion_volume
                        temp_data_bavaria["nb_lesions"] = nb_lesions
                        if args.lesion_only and nb_lesions == 0:
                            continue
                        temp_list.append(temp_data_bavaria)
                
            params[name] = temp_list
            logger.info(f"Number of images in {name} set: {len(temp_list)}")
    params["numTest"] = len(params["test"])
    params["numTraining"] = len(params["train"])
    params["numValidation"] = len(params["validation"])
    # Print total number of images
    logger.info(f"Total number of images in the dataset: {params['numTest'] + params['numTraining'] + params['numValidation']}")

    final_json = json.dumps(params, indent=4, sort_keys=True)
    if not os.path.exists(args.path_out):
        os.makedirs(args.path_out, exist_ok=True)
    if args.lesion_only:
        jsonFile = open(args.path_out + "/" + f"dataset_{str(date.today())}_seed{seed}_lesionOnly.json", "w")
    else:
        jsonFile = open(args.path_out + "/" + f"dataset_{str(date.today())}_seed{seed}.json", "w")
    jsonFile.write(final_json)
    jsonFile.close()

    return None


if __name__ == "__main__":
    main()