"""
This file creates the MSD-style JSON datalist to train an nnunet model using monai. 
The datasets used are CanProCo, Bavaria-quebec, basel and sct-testing-large.

Arguments:
    -pd, --path-data: Path to the data set directory
    -po, --path-out: Path to the output directory where dataset json is saved
    --lesion-only: Use only masks which contain some lesions
    --seed: Seed for reproducibility
    --canproco-exclude: Path to the file containing the list of subjects to exclude from CanProCo

Example:
    python 1_create_msd_data.py -pd /path/dataset -po /path/output --lesion-only --seed 42 --canproco-exclude /path/exclude_list.txt

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
from utils.image import Image


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
    derivatives_bavaria_unstitched = list(path_bavaria_unstitched.rglob('*_lesion-manual.nii.gz'))
    derivatives_canproco = list(path_canproco.rglob('*_lesion-manual.nii.gz'))
    derivatives_basel_2018 = list(path_basel_2018.rglob('*_lesion-manual.nii.gz'))
    derivatives_basel_2020 = list(path_basel_2020.rglob('*_lesion-manual.nii.gz'))
    derivatives_karo = list(path_karo.rglob('*_lesion-manual.nii.gz'))
    derivatives_nih = list(path_nih.rglob('*_desc-rater1_label-lesion_seg.nii.gz'))
    derivatives_nyu = list(path_nyu.rglob('*_lesion-manual.nii.gz'))
    derivatives_sct = list(path_sct_testing.rglob('*_lesion-manual.nii.gz'))

    # Path to the file containing the list of subjects to exclude from CanProCo
    if args.canproco_exclude is not None:
       with open(args.canproco_exclude, 'r') as file:
            canproco_exclude_list = yaml.load(file, Loader=yaml.FullLoader)
    # only keep the contrast psir and stir
    canproco_exclude_list = canproco_exclude_list['PSIR'] + canproco_exclude_list['STIR']

    # Print the number of derivatives in each dataset
    logger.info(f"Number of derivatives in basel-mp2rage: {len(derivatives_basel_mp2rage)}")
    logger.info(f"Number of derivatives in bavaria-quebec: {len(derivatives_bavaria_unstitched)}")
    logger.info(f"Number of derivatives in canproco: {len(derivatives_canproco)}")
    logger.info(f"Number of derivatives in ms-basel-2018: {len(derivatives_basel_2018)}")
    logger.info(f"Number of derivatives in ms-basel-2020: {len(derivatives_basel_2020)}")
    logger.info(f"Number of derivatives in ms-karolinska-2020: {len(derivatives_karo)}")
    logger.info(f"Number of derivatives in nih-ms-mp2rage: {len(derivatives_nih)}")
    logger.info(f"Number of derivatives in ms-nyu: {len(derivatives_nyu)}")
    logger.info(f"Number of derivatives in sct-testing-large: {len(derivatives_sct)}")

    derivatives = derivatives_basel_mp2rage + derivatives_bavaria_unstitched + derivatives_canproco + derivatives_basel_2018 + derivatives_basel_2020 + derivatives_nih + derivatives_nyu + derivatives_sct
    external_derivatives = derivatives_basel_2018 + derivatives_basel_2020 + derivatives_karo
    logger.info(f"Total number of derivatives in the root directory: {len(derivatives)}")

    # create one json file with 60-20-20 train-val-test split
    train_ratio, val_ratio, test_ratio = 0.6, 0.2, 0.2
    train_derivatives, test_derivatives = train_test_split(derivatives, test_size=test_ratio, random_state=args.seed)
    # Use the training split to further split into training and validation splits
    train_derivatives, val_derivatives = train_test_split(train_derivatives, test_size=val_ratio / (train_ratio + val_ratio),
                                                    random_state=args.seed, )
    # sort the subjects
    train_derivatives = sorted(train_derivatives)
    val_derivatives = sorted(val_derivatives)
    test_derivatives = sorted(test_derivatives)

    # logger.info(f"Number of training subjects: {len(train_derivatives)}")
    # logger.info(f"Number of validation subjects: {len(val_derivatives)}")
    # logger.info(f"Number of testing subjects: {len(test_derivatives)}")

    # # dump train/val/test splits into a yaml file
    # with open(f"{args.path_out}/data_split_{str(date.today())}_seed{seed}.yaml", 'w') as file:
    #     yaml.dump({'train': train_derivatives, 'val': val_derivatives, 'test': test_derivatives}, file, indent=2, sort_keys=True)

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
    for derivatives_dict in tqdm(all_derivatives_list, desc="Iterating through train/val/test splits"):

        for name, derivs_list in derivatives_dict.items():

            temp_list = []
            for subject_no, derivative in enumerate(derivs_list):

                
                temp_data_basel = {}
                temp_data_bavaria = {}
                temp_data_canproco = {}
                temp_data_nih = {}
                temp_data_sct = {}

                # # Basel
                # if 'basel-mp2rage' in str(derivative):
                #     relative_path = derivative.relative_to(path_basel_mp2rage).parent
                #     temp_data_basel["label"] = str(derivative)
                #     temp_data_basel["image"] = str(derivative).replace('_desc-rater3_label-lesion_seg.nii.gz', '.nii.gz').replace('derivatives/labels/', '')
                #     if os.path.exists(temp_data_basel["label"]) and os.path.exists(temp_data_basel["image"]):
                #         total_lesion_volume, nb_lesions = count_lesion(temp_data_basel["label"])
                #         temp_data_basel["total_lesion_volume"] = total_lesion_volume
                #         temp_data_basel["nb_lesions"] = nb_lesions
                #         temp_data_basel["site"]='basel-mp2rage'
                #         temp_data_basel["contrast"] = str(derivative).replace('_desc-rater3_label-lesion_seg.nii.gz', '.nii.gz').split('_')[-1].replace('.nii.gz', '')
                #         acquisition, resolution, dimension = get_acquisition_resolution_and_dimension(temp_data_basel["image"])
                #         temp_data_basel["acquisition"] = acquisition
                #         temp_data_basel["resolution"] = resolution
                #         temp_data_basel["dimension"] = dimension
                #         if args.lesion_only and nb_lesions == 0:
                #             continue
                #         temp_list.append(temp_data_basel)
            

                # # Bavaria-quebec
                # elif 'bavaria-quebec-spine-ms' in str(derivative):
                #     temp_data_bavaria["label"] = str(derivative)
                #     temp_data_bavaria["image"] = str(derivative).replace('_lesion-manual.nii.gz', '.nii.gz').replace('derivatives/labels/', '')
                #     if os.path.exists(temp_data_bavaria["label"]) and os.path.exists(temp_data_bavaria["image"]):
                #         total_lesion_volume, nb_lesions = count_lesion(temp_data_bavaria["label"])
                #         temp_data_bavaria["total_lesion_volume"] = total_lesion_volume
                #         temp_data_bavaria["nb_lesions"] = nb_lesions
                #         temp_data_bavaria["site"]='bavaria-quebec'
                #         temp_data_bavaria["contrast"] = str(derivative).replace('_lesion-manual.nii.gz', '.nii.gz').split('_')[-1].replace('.nii.gz', '')
                #         temp_data_bavaria["orientation"] = get_orientation(temp_data_bavaria["image"])
                #         if args.lesion_only and nb_lesions == 0:
                #             continue
                #         temp_list.append(temp_data_bavaria)
                
                # # Canproco
                # elif 'canproco' in str(derivative):
                #     subject_id = derivative.name.replace('_PSIR_lesion-manual.nii.gz', '')
                #     subject_id = subject_id.replace('_STIR_lesion-manual.nii.gz', '')
                #     if subject_id in canproco_exclude_list:
                #         continue  
                #     temp_data_canproco["label"] = str(derivative)
                #     temp_data_canproco["image"] = str(derivative).replace('_lesion-manual.nii.gz', '.nii.gz').replace('derivatives/labels/', '')
                #     if os.path.exists(temp_data_canproco["label"]) and os.path.exists(temp_data_canproco["image"]):
                #         total_lesion_volume, nb_lesions = count_lesion(temp_data_canproco["label"])
                #         temp_data_canproco["total_lesion_volume"] = total_lesion_volume
                #         temp_data_canproco["nb_lesions"] = nb_lesions
                #         temp_data_canproco["site"]='canproco'
                #         temp_data_canproco["contrast"] = str(derivative).replace('_lesion-manual.nii.gz', '.nii.gz').split('_')[-1].replace('.nii.gz', '')
                #         acquisition, resolution, dimension = get_acquisition_resolution_and_dimension(temp_data_canproco["image"])
                #         temp_data_canproco["acquisition"] = acquisition
                #         temp_data_canproco["resolution"] = resolution
                #         temp_data_canproco["dimension"] = dimension
                #         if args.lesion_only and nb_lesions == 0:
                #             continue
                #         temp_list.append(temp_data_canproco)

                # # nih-ms-mp2rage
                # elif 'nih-ms-mp2rage' in str(derivative):
                #     temp_data_nih["label"] = str(derivative)
                #     temp_data_nih["image"] = str(derivative).replace('_desc-rater1_label-lesion_seg.nii.gz', '.nii.gz').replace('derivatives/labels/', '')
                #     if os.path.exists(temp_data_nih["label"]) and os.path.exists(temp_data_nih["image"]):
                #         total_lesion_volume, nb_lesions = count_lesion(temp_data_nih["label"])
                #         temp_data_nih["total_lesion_volume"] = total_lesion_volume
                #         temp_data_nih["nb_lesions"] = nb_lesions
                #         temp_data_nih["site"]='nih-ms-mp2rage'
                #         temp_data_nih["contrast"] = str(derivative).replace('_desc-rater1_label-lesion_seg.nii.gz', '.nii.gz').split('_')[-1].replace('.nii.gz', '')
                #         acquisition, resolution, dimension = get_acquisition_resolution_and_dimension(temp_data_nih["image"])
                #         temp_data_nih["acquisition"] = acquisition
                #         temp_data_nih["resolution"] = resolution
                #         temp_data_nih["dimension"] = dimension
                #         if args.lesion_only and nb_lesions == 0:
                #             continue
                #         temp_list.append(temp_data_nih)

                # sct-testing-large
                if 'sct-testing-large' in str(derivative):
                    temp_data_sct["label"] = str(derivative)
                    temp_data_sct["image"] = str(derivative).replace('_lesion-manual.nii.gz', '.nii.gz').replace('derivatives/labels/', '')
                    if os.path.exists(temp_data_sct["label"]) and os.path.exists(temp_data_sct["image"]):
                        total_lesion_volume, nb_lesions = count_lesion(temp_data_sct["label"])
                        temp_data_sct["total_lesion_volume"] = total_lesion_volume
                        temp_data_sct["nb_lesions"] = nb_lesions
                        # For the site, we use the name of the dataset and the beginning of the subject id
                        ## remove common path between sct-testing-large and the root
                        site = str(os.path.relpath(derivative, Path(temp_data_sct["image"]))).split('/')[1].replace("sub-","")
                        print(site)
                        ## remove the subject number:
                        site=''.join([i for i in site if not i.isdigit()])
                        print(site)
                        temp_data_sct["site"]='sct-testing-large'
                        temp_data_sct["contrast"] = str(derivative).replace('_lesion-manual.nii.gz', '.nii.gz').split('_')[-1].replace('.nii.gz', '')
                        # temp_data_sct["orientation"] = get_orientation(temp_data_sct["image"])
                        if args.lesion_only and nb_lesions == 0:
                            continue
                        temp_list.append(temp_data_sct)
                    break

                        
    #         params[name] = temp_list
    #         logger.info(f"Number of images in {name} set: {len(temp_list)}")
    # params["numTest"] = len(params["test"])
    # params["numTraining"] = len(params["train"])
    # params["numValidation"] = len(params["validation"])
    # # Print total number of images
    # logger.info(f"Total number of images in the dataset: {params['numTest'] + params['numTraining'] + params['numValidation']}")

    # final_json = json.dumps(params, indent=4, sort_keys=True)
    # if not os.path.exists(args.path_out):
    #     os.makedirs(args.path_out, exist_ok=True)
    # if args.lesion_only:
    #     jsonFile = open(args.path_out + "/" + f"dataset_{str(date.today())}_seed{seed}_lesionOnly.json", "w")
    # else:
    #     jsonFile = open(args.path_out + "/" + f"dataset_{str(date.today())}_seed{seed}.json", "w")
    # jsonFile.write(final_json)
    # jsonFile.close()

    return None


if __name__ == "__main__":
    main()