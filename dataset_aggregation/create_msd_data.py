"""
This file creates the MSD-style JSON datalist to train an nnunet model using monai. 

Arguments:
    -pd, --path-data: Path to the data set directory
    -po, --path-out: Path to the output directory where dataset json is saved
    --lesion-only: Use only masks which contain some lesions
    --seed: Seed for reproducibility
    --canproco-exclude: Path to the file containing the list of subjects to exclude from CanProCo
    --exclude: Path to the file containing the list of subjects to exclude from the dataset
    --all-train: Use the data to train the model (except for the external test data)

Example:
    python 1_create_msd_data.py -pd /path/dataset -po /path/output --lesion-only --seed 42 --canproco-exclude /path/exclude_list.txt --exclude /path/exclude_list.txt

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
from utils.image import Image
import pandas as pd


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
    parser.add_argument('--canproco-exclude', type=str, help='Path to the file containing the list of subjects to exclude from CanProCo')
    parser.add_argument('--exclude', type=str, help='Path to the file containing the list of subjects to exclude from the dataset')
    parser.add_argument('--lesion-only', action='store_true', help='Use only masks which contain some lesions')
    parser.add_argument('--seed', default=42, type=int, help="Seed for reproducibility")
    parser.add_argument('--all-train', action='store_true', help='Use the data to train the model (except for the external test data)')
    parser.add_argument('--list-contrast-distribution', action='store_true', help='Print the distribution of contrasts in the dataset and exit')

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


def get_acquisition_resolution_and_dimension(image_path):
    """
    This function takes an image file as input and returns its acquisition, resolution and dimension.

    Input:
        image_path : str : Path to the image file

    Returns:
        acquisition : str : Acquisition of the image
        resolution : list : Resolution of the image
        dimension : list : Dimension of the image
    """
    img = Image(str(image_path))
    img.change_orientation('RPI')
    # Get pixdim
    pixdim = img.dim[4:7]
    # If all are the same, the image is 3D
    if np.allclose(pixdim, pixdim[0], atol=1e-3):
        acquisition = '3D'
    # Elif, the lowest arg is 0 then the acquisition is sagittal
    elif np.argmax(pixdim) == 0:
        acquisition = 'sag'
    # Elif, the lowest arg is 1 then the acquisition is coronal
    elif np.argmax(pixdim) == 1:
        acquisition = 'cor'
    # Else the acquisition is axial
    else:
        acquisition = 'ax'
    # Get the resolution
    resolution = list(img.dim[4:7])
    # Get image dimension
    dimension = list(img.dim[0:3])
    return acquisition, resolution, dimension


def print_dataset_contrasts_distribution(derivatives, dataset_name):
    """
    This function takes a list of derivatives and prints the distribution of contrasts in the dataset.
    Input:
        derivatives : list : List of derivatives
    Returns:
        None
    """
    # Get the contrasts
    contrasts = []
    for derivative in derivatives:
        if 'basel-mp2rage' in str(derivative):
            contrast = str(derivative).replace('_desc-rater3_label-lesion_seg.nii.gz', '.nii.gz').split('_')[-1].replace('.nii.gz', '')
            contrasts.append(contrast)
        elif 'nih-ms-mp2rage' in str(derivative):
            contrast = str(derivative).replace('_desc-rater1_label-lesion_seg.nii.gz', '.nii.gz').split('_')[-1].replace('.nii.gz', '')
            contrasts.append(contrast)
        else:
            contrast = str(derivative).replace('_lesion-manual.nii.gz', '.nii.gz').split('_')[-1].replace('.nii.gz', '')
            contrasts.append(contrast)        
        
    # Get the unique contrasts
    unique_contrasts = set(contrasts)
    # Print the distribution
    logger.info(f"Distribution of contrasts in {dataset_name}:")
    for contrast in unique_contrasts:
        count = contrasts.count(contrast)
        logger.info(f"{contrast}: {count}")
    return contrasts


def split_dataset(derivatives, test_size=0.1, random_state=42):
    """
    This function takes a list of derivatives and splits it into train, validation and test sets based on the subjects.

    Input:
        derivatives : list : List of derivatives
        test_size : float : Size of the test set
        random_state : int : Random state for reproducibility

    Returns:
        train : list : List of training derivatives
        val : list : List of validation derivatives
        test : list : List of test derivatives
    """
    df_derivatives = pd.DataFrame(derivatives, columns=['derivative'])
    df_derivatives['subject'] = df_derivatives['derivative'].apply(lambda x: x.name.split('_')[0])
    df_derivatives_subjects = df_derivatives['subject'].unique() 
    df_derivatives_subjects_train, df_derivatives_subjects_test = train_test_split(df_derivatives_subjects, test_size=test_size, random_state=random_state)
    df_derivatives_subjects_train, df_derivatives_subjects_val = train_test_split(df_derivatives_subjects_train, test_size=test_size, random_state=random_state)
    train = df_derivatives[df_derivatives['subject'].isin(df_derivatives_subjects_train)]["derivative"].tolist()
    val = df_derivatives[df_derivatives['subject'].isin(df_derivatives_subjects_val)]["derivative"].tolist()
    test = df_derivatives[df_derivatives['subject'].isin(df_derivatives_subjects_test)]["derivative"].tolist()
    return train, val, test


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

    test_size = 0.1

    # Get all subjects
    path_basel_mp2rage = Path(os.path.join(root, "basel-mp2rage"))
    path_bavaria_unstitched = Path(os.path.join(root, "bavaria-quebec-spine-ms-unstitched"))
    path_canproco = Path(os.path.join(root, "canproco"))
    path_basel_2018 = Path(os.path.join(root, "ms-basel-2018"))
    path_basel_2020 = Path(os.path.join(root, "ms-basel-2020"))
    path_karo = Path(os.path.join(root, "ms-karolinska-2020"))
    path_nih = Path(os.path.join(root, "nih-ms-mp2rage"))
    path_nyu = Path(os.path.join(root, "ms-nyu"))
    path_sct_testing = Path(os.path.join(root, "sct-testing-large"))

    derivatives_basel_mp2rage = list(path_basel_mp2rage.rglob('*_desc-rater3_label-lesion_seg.nii.gz'))
    derivatives_basel_mp2rage = [derivative for derivative in derivatives_basel_mp2rage if 'labels-ms-spinal' in str(derivative)]
    derivatives_bavaria_unstitched = list(path_bavaria_unstitched.rglob('*_lesion-manual.nii.gz'))
    derivatives_bavaria_unstitched = [derivative for derivative in derivatives_bavaria_unstitched if 'labels-ms-spinal' in str(derivative)]
    derivatives_canproco = list(path_canproco.rglob('*_lesion-manual.nii.gz'))
    derivatives_canproco = [derivative for derivative in derivatives_canproco if 'labels-ms-spinal' in str(derivative)]
    derivatives_basel_2018 = list(path_basel_2018.rglob('*_lesion-manual.nii.gz'))
    derivatives_basel_2020 = list(path_basel_2020.rglob('*_lesion-manual.nii.gz'))
    # Remove PD files from basel_2020
    derivatives_basel_2020 = [derivative for derivative in derivatives_basel_2020 if '_PD_' not in str(derivative)]
    derivatives_karo = list(path_karo.rglob('*_lesion-manual.nii.gz'))
    derivatives_nih = list(path_nih.rglob('*_desc-rater1_label-lesion_seg.nii.gz'))
    derivatives_nih = [derivative for derivative in derivatives_nih if 'labels-ms-spinal' in str(derivative)]
    derivatives_nyu = list(path_nyu.rglob('*_lesion-manual.nii.gz'))
    derivatives_nyu = [derivative for derivative in derivatives_nyu if 'labels-ms-spinal' in str(derivative)]
    derivatives_sct = list(path_sct_testing.rglob('*_lesion-manual.nii.gz'))
    derivatives_sct = [derivative for derivative in derivatives_sct if 'labels-ms-spinal' in str(derivative)]

    # Path to the file containing the list of subjects to exclude from CanProCo
    if args.canproco_exclude is not None:
       with open(args.canproco_exclude, 'r') as file:
            canproco_exclude_list = yaml.load(file, Loader=yaml.FullLoader)
    # only keep the contrast psir and stir
    canproco_exclude_list = canproco_exclude_list['PSIR'] + canproco_exclude_list['STIR']

    # Path to the file containing the list of subjects to exclude from the datasets
    if args.exclude is not None:
         with open(args.exclude, 'r') as file:
                exclude_list = yaml.load(file, Loader=yaml.FullLoader)
    exclude_list = exclude_list['EXCLUDED']

    # Print the distribution of contrasts in the datasets
    contrasts_basel_mp2rage = print_dataset_contrasts_distribution(derivatives_basel_mp2rage, "Basel MP2RAGE")
    contrasts_bavaria = print_dataset_contrasts_distribution(derivatives_bavaria_unstitched, "Bavaria Quebec")
    contrasts_canproco = print_dataset_contrasts_distribution(derivatives_canproco, "CanProCo")
    contrasts_basel_2018 = print_dataset_contrasts_distribution(derivatives_basel_2018, "Basel 2018")
    contrasts_basel_2020 = print_dataset_contrasts_distribution(derivatives_basel_2020, "Basel 2020")
    logger.info(f"We decided not to include the PD images because of the quality of the manual segmentations.")
    contrasts_karo = print_dataset_contrasts_distribution(derivatives_karo, "Karolinska")
    contrasts_nih = print_dataset_contrasts_distribution(derivatives_nih, "NIH")
    contrasts_nyu = print_dataset_contrasts_distribution(derivatives_nyu, "NYU")
    contrasts_sct = print_dataset_contrasts_distribution(derivatives_sct, "SCT Testing")

    all_contrasts = contrasts_basel_mp2rage + contrasts_bavaria + contrasts_canproco + contrasts_basel_2018 + contrasts_basel_2020 + contrasts_karo + contrasts_nih + contrasts_nyu + contrasts_sct
    # Change MEGRE TO T2star in the list
    all_contrasts = [contrast.replace('MEGRE', 'T2star') for contrast in all_contrasts]
    # create a dictionnary which stores the counts for each contrast
    contrast_counts = {}
    for contrast in all_contrasts:
        if contrast in contrast_counts:
            contrast_counts[contrast] += 1
        else:
            contrast_counts[contrast] = 1
    # Print the counts
    logger.info(f"Distribution of contrasts in the dataset:")
    for contrast, count in contrast_counts.items():
        logger.info(f"{contrast}: {count}")

    if args.list_contrast_distribution:
        # We print the distribution of contrasts in the datasets and exit
        return None

    # The splitting should be done on the subjects and not on the images. It should also be done per site.
    ## To do so, we build a df of the subjects and then split it
    basel_mp2rage_train, basel_mp2rage_val, basel_mp2rage_test = split_dataset(derivatives_basel_mp2rage, test_size=test_size, random_state=args.seed)
    bavaria_unstitched_train, bavaria_unstitched_val, bavaria_unstitched_test = split_dataset(derivatives_bavaria_unstitched, test_size=test_size, random_state=args.seed)
    canproco_train, canproco_val, canproco_test = split_dataset(derivatives_canproco, test_size=test_size, random_state=args.seed)
    basel_2018_train, basel_2018_val, basel_2018_test = split_dataset(derivatives_basel_2018, test_size=test_size, random_state=args.seed)
    nih_train, nih_val, nih_test = split_dataset(derivatives_nih, test_size=test_size, random_state=args.seed)
    nyu_train, nyu_val, nyu_test = split_dataset(derivatives_nyu, test_size=test_size, random_state=args.seed)
    sct_train, sct_val, sct_test = split_dataset(derivatives_sct, test_size=test_size, random_state=args.seed)

    # Gather the splittings
    train_derivatives = basel_mp2rage_train + bavaria_unstitched_train + canproco_train + basel_2018_train + nih_train + nyu_train + sct_train
    val_derivatives = basel_mp2rage_val + bavaria_unstitched_val + canproco_val + basel_2018_val + nih_val + nyu_val + sct_val
    test_derivatives = basel_mp2rage_test + bavaria_unstitched_test + canproco_test + basel_2018_test + nih_test + nyu_test + sct_test
    # As for the external datasets, we don't split them
    external_derivatives = derivatives_karo

    # If the flag --all-train is set, use all the data for training
    if args.all_train:
        train_derivatives = train_derivatives + val_derivatives + test_derivatives
        val_derivatives = []
        test_derivatives = []
    
    # sort the subjects
    train_derivatives = sorted(train_derivatives)
    val_derivatives = sorted(val_derivatives)
    test_derivatives = sorted(test_derivatives)

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

    train_derivatives_dict = {"train": train_derivatives}
    val_derivatives_dict = {"validation": val_derivatives}
    test_derivatives_dict =  {"test": test_derivatives}
    all_derivatives_list = [train_derivatives_dict, val_derivatives_dict, test_derivatives_dict]

    # iterate through the train/val/test splits and add those which have both image and label
    subjects_basel = []
    subjects_bavaria = []
    subjects_canproco = []
    subjects_nih = []
    subjects_nyu = []
    subjects_sct = []
    for derivatives_dict in tqdm(all_derivatives_list, desc="Iterating through train/val/test splits"):

        for name, derivs_list in derivatives_dict.items():

            temp_list = []
            for subject_no, derivative in tqdm(enumerate(derivs_list)):

                temp_data_basel = {}
                temp_data_bavaria = {}
                temp_data_canproco = {}
                temp_data_nih = {}
                temp_data_sct = {}
                temp_data_nyu = {}
                
                # Basel
                if 'basel-mp2rage' in str(derivative):
                    if str(derivative).split('/')[-1] in exclude_list:
                        print("excluded")
                        continue
                    relative_path = derivative.relative_to(path_basel_mp2rage).parent
                    temp_data_basel["label"] = str(derivative)
                    temp_data_basel["image"] = str(derivative).replace('_desc-rater3_label-lesion_seg.nii.gz', '.nii.gz').replace('derivatives/labels-ms-spinal-cord-only/', '')
                    if os.path.exists(temp_data_basel["label"]) and os.path.exists(temp_data_basel["image"]):
                        total_lesion_volume, nb_lesions = count_lesion(temp_data_basel["label"])
                        temp_data_basel["total_lesion_volume"] = total_lesion_volume
                        temp_data_basel["nb_lesions"] = nb_lesions
                        temp_data_basel["site"]='basel-mp2rage'
                        temp_data_basel["contrast"] = str(derivative).replace('_desc-rater3_label-lesion_seg.nii.gz', '.nii.gz').split('_')[-1].replace('.nii.gz', '')
                        acquisition, resolution, dimension = get_acquisition_resolution_and_dimension(temp_data_basel["image"])
                        temp_data_basel["acquisition"] = acquisition
                        # Convert each value to float64
                        resolution = [np.float64(i) for i in resolution]
                        temp_data_basel["resolution"] = resolution
                        temp_data_basel["dimension"] = dimension
                        if args.lesion_only and nb_lesions == 0:
                            continue
                        temp_list.append(temp_data_basel)
                        # Get the subject
                        subject = str(derivative).split('/')[-1].split('_')[0]
                        subjects_basel.append(subject)
            
                # Bavaria-quebec
                elif 'bavaria-quebec-spine-ms' in str(derivative):
                    if str(derivative).split('/')[-1] in exclude_list:
                        continue
                    temp_data_bavaria["label"] = str(derivative)
                    temp_data_bavaria["image"] = str(derivative).replace('_lesion-manual.nii.gz', '.nii.gz').replace('derivatives/labels-ms-spinal-cord-only/', '')
                    if os.path.exists(temp_data_bavaria["label"]) and os.path.exists(temp_data_bavaria["image"]):
                        total_lesion_volume, nb_lesions = count_lesion(temp_data_bavaria["label"])
                        temp_data_bavaria["total_lesion_volume"] = total_lesion_volume
                        temp_data_bavaria["nb_lesions"] = nb_lesions
                        temp_data_bavaria["site"]='bavaria-quebec-spine-ms'
                        temp_data_bavaria["contrast"] = str(derivative).replace('_lesion-manual.nii.gz', '.nii.gz').split('_')[-1].replace('.nii.gz', '')
                        acquisition, resolution, dimension = get_acquisition_resolution_and_dimension(temp_data_bavaria["image"])
                        temp_data_bavaria["acquisition"] = acquisition
                        resolution = [np.float64(i) for i in resolution]
                        temp_data_bavaria["resolution"] = resolution
                        temp_data_bavaria["dimension"] = dimension
                        if args.lesion_only and nb_lesions == 0:
                            continue
                        temp_list.append(temp_data_bavaria)
                        # Get the subject
                        subject = str(derivative).split('/')[-1].split('_')[0]
                        subjects_bavaria.append(subject)
                
                # Canproco
                elif 'canproco' in str(derivative):
                    subject_id = derivative.name.replace('_PSIR_lesion-manual.nii.gz', '')
                    subject_id = subject_id.replace('_STIR_lesion-manual.nii.gz', '')
                    if subject_id in canproco_exclude_list:
                        continue  
                    temp_data_canproco["label"] = str(derivative)
                    temp_data_canproco["image"] = str(derivative).replace('_lesion-manual.nii.gz', '.nii.gz').replace('derivatives/labels-ms-spinal-cord-only/', '')
                    if os.path.exists(temp_data_canproco["label"]) and os.path.exists(temp_data_canproco["image"]):
                        total_lesion_volume, nb_lesions = count_lesion(temp_data_canproco["label"])
                        temp_data_canproco["total_lesion_volume"] = total_lesion_volume
                        temp_data_canproco["nb_lesions"] = nb_lesions
                        temp_data_canproco["site"]='canproco'
                        temp_data_canproco["contrast"] = str(derivative).replace('_lesion-manual.nii.gz', '.nii.gz').split('_')[-1].replace('.nii.gz', '')
                        acquisition, resolution, dimension = get_acquisition_resolution_and_dimension(temp_data_canproco["image"])
                        temp_data_canproco["acquisition"] = acquisition
                        resolution = [np.float64(i) for i in resolution]
                        temp_data_canproco["resolution"] = resolution
                        temp_data_canproco["dimension"] = dimension
                        if args.lesion_only and nb_lesions == 0:
                            continue
                        temp_list.append(temp_data_canproco)
                        # Get the subject
                        subject = str(derivative).split('/')[-1].split('_')[0]
                        subjects_canproco.append(subject)

                # nih-ms-mp2rage
                elif 'nih-ms-mp2rage' in str(derivative):
                    if str(derivative).split('/')[-1] in exclude_list:
                        continue
                    temp_data_nih["label"] = str(derivative)
                    temp_data_nih["image"] = str(derivative).replace('_desc-rater1_label-lesion_seg.nii.gz', '.nii.gz').replace('derivatives/labels-ms-spinal-cord-only/', '')
                    if os.path.exists(temp_data_nih["label"]) and os.path.exists(temp_data_nih["image"]):
                        total_lesion_volume, nb_lesions = count_lesion(temp_data_nih["label"])
                        temp_data_nih["total_lesion_volume"] = total_lesion_volume
                        temp_data_nih["nb_lesions"] = nb_lesions
                        temp_data_nih["site"]='nih-ms-mp2rage'
                        temp_data_nih["contrast"] = str(derivative).replace('_desc-rater1_label-lesion_seg.nii.gz', '.nii.gz').split('_')[-1].replace('.nii.gz', '')
                        acquisition, resolution, dimension = get_acquisition_resolution_and_dimension(temp_data_nih["image"])
                        temp_data_nih["acquisition"] = acquisition
                        resolution = [np.float64(i) for i in resolution]
                        temp_data_nih["resolution"] = resolution
                        temp_data_nih["dimension"] = dimension
                        if args.lesion_only and nb_lesions == 0:
                            continue
                        temp_list.append(temp_data_nih)
                        # Get the subject
                        subject = str(derivative).split('/')[-1].split('_')[0]
                        subjects_nih.append(subject)

                # ms-nyu   
                elif 'ms-nyu' in str(derivative):
                    if str(derivative).split('/')[-1] in exclude_list:
                        continue
                    temp_data_nyu["label"] = str(derivative)
                    temp_data_nyu["image"] = str(derivative).replace('_lesion-manual.nii.gz', '.nii.gz').replace('derivatives/labels-ms-spinal-cord-only/', '')
                    if os.path.exists(temp_data_nyu["label"]) and os.path.exists(temp_data_nyu["image"]):
                        total_lesion_volume, nb_lesions = count_lesion(temp_data_nyu["label"])
                        temp_data_nyu["total_lesion_volume"] = total_lesion_volume
                        temp_data_nyu["nb_lesions"] = nb_lesions
                        temp_data_nyu["site"]='ms-nyu'
                        temp_data_nyu["contrast"] = str(derivative).replace('_lesion-manual.nii.gz', '.nii.gz').split('_')[-1].replace('.nii.gz', '')
                        acquisition, resolution, dimension = get_acquisition_resolution_and_dimension(temp_data_nyu["image"])
                        temp_data_nyu["acquisition"] = acquisition
                        resolution = [np.float64(i) for i in resolution]
                        temp_data_nyu["resolution"] = resolution
                        temp_data_nyu["dimension"] = dimension
                        if args.lesion_only and nb_lesions == 0:
                            continue
                        temp_list.append(temp_data_nyu)
                        # Get the subject
                        subject = str(derivative).split('/')[-1].split('_')[0]
                        subjects_nyu.append(subject)

                # sct-testing-large
                elif 'sct-testing-large' in str(derivative):
                    if str(derivative).split('/')[-1] in exclude_list:
                        continue
                    temp_data_sct["label"] = str(derivative)
                    temp_data_sct["image"] = str(derivative).replace('_lesion-manual.nii.gz', '.nii.gz').replace('derivatives/labels-ms-spinal-cord-only/', '')
                    if os.path.exists(temp_data_sct["label"]) and os.path.exists(temp_data_sct["image"]):
                        total_lesion_volume, nb_lesions = count_lesion(temp_data_sct["label"])
                        temp_data_sct["total_lesion_volume"] = total_lesion_volume
                        temp_data_sct["nb_lesions"] = nb_lesions
                        # For the site, we use the name of the dataset and the beginning of the subject id
                        ## remove common path between sct-testing-large and the root
                        site = str(os.path.relpath(Path(temp_data_sct["image"]),root)).split('/')[1].replace("sub-","")
                        ## remove the subject number:
                        site=''.join([i for i in site if not i.isdigit()])
                        temp_data_sct["site"]='sct-testing-large--'+site
                        temp_data_sct["contrast"] = str(derivative).replace('_lesion-manual.nii.gz', '.nii.gz').split('_')[-1].replace('.nii.gz', '')
                        acquisition, resolution, dimension = get_acquisition_resolution_and_dimension(temp_data_sct["image"])
                        temp_data_sct["acquisition"] = acquisition
                        resolution = [np.float64(i) for i in resolution]
                        temp_data_sct["resolution"] = resolution
                        temp_data_sct["dimension"] = dimension
                        if args.lesion_only and nb_lesions == 0:
                            continue
                        temp_list.append(temp_data_sct)
                        # Get the subject
                        subject = str(derivative).split('/')[-1].split('_')[0]
                        subjects_sct.append(subject)
                        
        params[name] = temp_list
        logger.info(f"Number of images in {name} set: {len(temp_list)}")
    params["numTest"] = len(params["test"])
    params["numTraining"] = len(params["train"])
    params["numValidation"] = len(params["validation"])
    params["numSubjectsTrainValTest"] = len(set(subjects_basel)) + len(set(subjects_bavaria)) + len(set(subjects_canproco)) + len(set(subjects_nih)) + len(set(subjects_nyu)) + len(set(subjects_sct))
    # Print the info of numbers
    logger.info(f"Number of subjects in TrainValTest: {params['numSubjectsTrainValTest']}")

    # Now for the external validation datasets:
    temp_list = []
    subjects_external = []
    for derivative in external_derivatives:

        temp_data = {}
        temp_data["label"] = str(derivative)
        temp_data["image"] = str(derivative).replace('_lesion-manual.nii.gz', '.nii.gz').replace('derivatives/labels/', '').replace('derivatives/labels-ms-spinal-cord-only/', '')
        if os.path.exists(temp_data["label"]) and os.path.exists(temp_data["image"]):
            total_lesion_volume, nb_lesions = count_lesion(temp_data["label"])
            temp_data["total_lesion_volume"] = total_lesion_volume
            temp_data["nb_lesions"] = nb_lesions
            site = str(os.path.relpath(Path(temp_data["image"]),root)).split('/')[0]
            temp_data["site"]=site
            temp_data["contrast"] = str(derivative).replace('_lesion-manual.nii.gz', '.nii.gz').split('_')[-1].replace('.nii.gz', '')
            acquisition, resolution, dimension = get_acquisition_resolution_and_dimension(temp_data["image"])
            temp_data["acquisition"] = acquisition
            resolution = [np.float64(i) for i in resolution]
            temp_data["resolution"] = resolution
            temp_data["dimension"] = dimension
            if args.lesion_only and nb_lesions == 0:
                continue
            temp_list.append(temp_data)
            # Get the subject
            subject = str(derivative).split('/')[-1].split('_')[0]
            subjects_external.append(site+"-"+subject)

    params["externalValidation"] = temp_list
    params["numExternalValidation"] = len(params["externalValidation"])
    params["numSubjectsExternalValidation"] = len(set(subjects_external))
    # Print the number of subjects in the external validation set
    logger.info(f"Number of images in the external validation set: {params['numExternalValidation']}")
    logger.info(f"Number of subjects in the external validation set: {params['numSubjectsExternalValidation']}")

    # Print total number of images
    logger.info(f"Total number of images in the dataset: {params['numTest'] + params['numTraining'] + params['numValidation'] + params['numExternalValidation']}")
    final_json = json.dumps(params, indent=4, sort_keys=True)
    if not os.path.exists(args.path_out):
        os.makedirs(args.path_out, exist_ok=True)
    if args.lesion_only and args.all_train:
        jsonFile = open(args.path_out + "/" + f"dataset_{str(date.today())}_seed{seed}_lesionOnly_allTrain.json", "w")
    elif args.lesion_only:
        jsonFile = open(args.path_out + "/" + f"dataset_{str(date.today())}_seed{seed}_lesionOnly.json", "w")
    elif args.all_train:
        jsonFile = open(args.path_out + "/" + f"dataset_{str(date.today())}_seed{seed}_allTrain.json", "w")
    else:
        jsonFile = open(args.path_out + "/" + f"dataset_{str(date.today())}_seed{seed}.json", "w")
    jsonFile.write(final_json)
    jsonFile.close()

    return None


if __name__ == "__main__":
    main()