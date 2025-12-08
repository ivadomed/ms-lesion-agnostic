"""
This file creates the MSD-style JSON datalist to train an nnunet model using monai. 

Arguments:
    -pd, --path-data: Path to the data set directory
    -po, --path-out: Path to the output directory where dataset json is saved
    --canproco-exclude: Path to the file containing the list of subjects to exclude from CanProCo
    --exclude: Path to the file containing the list of subjects to exclude from the dataset
    --canproco-only: If set, only use CanProCo dataset

Example:
    python 1_create_msd_data.py -pd /path/to/dataset -po /path/to/output --canproco-exclude /path/to/canproco_exclude.yaml --exclude /path/to/exclude.yaml

Author: Pierre-Louis Benveniste
"""

import os
import json
from tqdm import tqdm
import yaml
import argparse
from loguru import logger
from datetime import date
from pathlib import Path
import numpy as np
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
    parser.add_argument('--canproco-only', action='store_true', help='If set, only use CanProCo dataset')
    return parser


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
    seed = 42

    # Save the logger to a file:
    os.makedirs(args.path_out, exist_ok=True)
    logger.add(os.path.join(args.path_out, f'logger_{str(date.today())}.log'))
    # Add the command line to the logger
    logger.info(f"Command line: python {' '.join(os.sys.argv)}")

    # Get all subjects
    path_bavaria_unstitched = Path(os.path.join(root, "bavaria-quebec-spine-ms-unstitched"))
    path_canproco = Path(os.path.join(root, "canproco"))
    path_basel_2018 = Path(os.path.join(root, "ms-basel-2018"))
    # path_dresden = Path(os.path.join(root, "ms-dresden-mp2rage-2025"))
    path_karo = Path(os.path.join(root, "ms-karolinska-2020"))
    path_umass1 = Path(os.path.join(root, "umass-ms-siemens-espree1.5"))
    path_umass2 = Path(os.path.join(root, "umass-ms-ge-pioneer3"))
    path_umass3 = Path(os.path.join(root, "umass-ms-ge-excite1.5"))
    path_umass4 = Path(os.path.join(root, "umass-ms-ge-hdxt1.5"))
    path_beijing = Path(os.path.join(root, "ms-nmo-beijing"))

    # Open the exclude files
    if args.canproco_exclude is not None:
       with open(args.canproco_exclude, 'r') as file:
            canproco_exclude_list = yaml.load(file, Loader=yaml.FullLoader)
    canproco_exclude_list = canproco_exclude_list['PSIR'] + canproco_exclude_list['STIR']
    # Add exclude file for all datasets
    if args.exclude is not None:
         with open(args.exclude, 'r') as file:
                exclude_list = yaml.load(file, Loader=yaml.FullLoader)
    exclude_list = exclude_list['EXCLUDED']

    # For each dataset, we select images which contain multiple time points
    
    # BAVARIA
    dict_bavaria = {}
    ### For bavaria, we keep all T2w images
    imgs_bavaria = list(path_bavaria_unstitched.rglob("*T2w.nii.gz"))
    imgs_bavaria = [str(i) for i in imgs_bavaria]
    ### Dictionnary is bavaria[subject_id][session_id] = [list of images]
    for img in tqdm(imgs_bavaria):
        subject_id = img.split("/")[-1].split("_")[0]
        session_id = img.split("/")[-1].split("_")[1]
        if "ses-" not in session_id:
            continue
        if subject_id not in dict_bavaria:
            dict_bavaria[subject_id] = {}
        if session_id not in dict_bavaria[subject_id]:
            dict_bavaria[subject_id][session_id] = []
        dict_bavaria[subject_id][session_id].append(img)
    # Then we remove from the dictionnary subjects with only one session
    dict_bavaria = {k: v for k, v in dict_bavaria.items() if len(v) > 1}
    # sort the sessions for each subject
    for subject_id in dict_bavaria:
        dict_bavaria[subject_id] = dict(sorted(dict_bavaria[subject_id].items(), key=lambda x: x[0]))
    logger.info(f"Number of subjects in Bavaria: {len(dict_bavaria)}")

    ## CANPROCO
    dict_canproco = {}
    ### For canproco, we keep all PSIR and STIR images
    imgs_canproco = list(path_canproco.rglob("*PSIR.nii.gz")) + list(path_canproco.rglob("*STIR.nii.gz"))
    imgs_canproco = [str(i) for i in imgs_canproco]
    ### Dictionnary is canproco[subject_id][session_id] = [list of images]
    for img in tqdm(imgs_canproco):
        subject_id = img.split("/")[-1].split("_")[0]
        session_id = img.split("/")[-1].split("_")[1]
        if "ses-" not in session_id:
            continue
        if subject_id+"_"+session_id in canproco_exclude_list:
            continue
        if subject_id not in dict_canproco:
            dict_canproco[subject_id] = {}
        if session_id not in dict_canproco[subject_id]:
            dict_canproco[subject_id][session_id] = []
        dict_canproco[subject_id][session_id].append(img)
    ### Then we remove from the dictionnary subjects with only one session
    dict_canproco = {k: v for k, v in dict_canproco.items() if len(v) > 1}
    # sort the sessions for each subject
    for subject_id in dict_canproco:
        dict_canproco[subject_id] = dict(sorted(dict_canproco[subject_id].items(), key=lambda x: x[0]))
    logger.info(f"Number of subjects in CanProCo: {len(dict_canproco)}")

    ## BASEL 2018
    dict_basel = {}
    ### For basel, we keep all nifti files except derivatives
    imgs_basel = list(path_basel_2018.rglob("*.nii.gz"))
    imgs_basel = [i for i in imgs_basel if 'derivatives' not in str(i)]
    imgs_basel = [i for i in imgs_basel if 'SHA256' not in str(i)] # We remove controls
    imgs_basel = [str(i) for i in imgs_basel]
    ### Dictionnary is basel[subject_id][session_id] = [list of images]
    for img in tqdm(imgs_basel):
        subject_id = img.split("/")[-1].split("_")[0]
        session_id = img.split("/")[-1].split("_")[1]
        if "ses-" not in session_id:
            continue
        if subject_id not in dict_basel:
            dict_basel[subject_id] = {}
        if session_id not in dict_basel[subject_id]:
            dict_basel[subject_id][session_id] = []
        dict_basel[subject_id][session_id].append(img)
    # Then we remove from the dictionnary subjects with only one session
    dict_basel = {k: v for k, v in dict_basel.items() if len(v) > 1}
    # Sort the sessions for each subject
    for subject_id in dict_basel:
        dict_basel[subject_id] = dict(sorted(dict_basel[subject_id].items(), key=lambda x: x[0]))
    logger.info(f"Number of subjects in Basel: {len(dict_basel)}")

    ## MS KAROLINSKA
    dict_karo = {}
    ### For karo, we keep all anat images except derivatives, FLAIR acq-isoMPR and acq-isoMpr
    imgs_karo = list(path_karo.rglob("*.nii.gz"))
    imgs_karo = [i for i in imgs_karo if 'derivatives' not in str(i)]
    imgs_karo = [i for i in imgs_karo if 'FLAIR' not in str(i)]
    imgs_karo = [i for i in imgs_karo if 'acq-isoMPR' not in str(i)]
    imgs_karo = [i for i in imgs_karo if 'acq-isoMpr' not in str(i)]
    imgs_karo = [i for i in imgs_karo if '/anat/' in str(i)] # We remove controls
    imgs_karo = [str(i) for i in imgs_karo]
    ### Dictionnary is karo[subject_id][session_id] = [list of images]
    for img in tqdm(imgs_karo): #tqdm for progress bar
        subject_id = img.split("/")[-1].split("_")[0]
        session_id = img.split("/")[-1].split("_")[1]
        if "ses-" not in session_id:
            continue
        if subject_id not in dict_karo:
            dict_karo[subject_id] = {}
        if session_id not in dict_karo[subject_id]:
            dict_karo[subject_id][session_id] = []
        dict_karo[subject_id][session_id].append(img)
    # Then we remove from the dictionnary subjects with only one session
    dict_karo = {k: v for k, v in dict_karo.items() if len(v) > 1}
    # Sort the sessions for each subject
    for subject_id in dict_karo:
        dict_karo[subject_id] = dict(sorted(dict_karo[subject_id].items(), key=lambda x: x[0]))
    logger.info(f"Number of subjects in Karo: {len(dict_karo)}")

    ## Umass 1
    dict_umass1 = {}
    ### For umass1, we keep all T1w, acq-FMPIR_T2w, acq-ax_T1w and acq-ax_T2w images
    imgs_umass_1 = list(Path(path_umass1).rglob('*_T1w.nii.gz')) # This is only for images with T1w (not designed for acq-...: there is not acq-... in this case)
    imgs_umass_1 = [i for i in imgs_umass_1 if '_acq-' not in str(i)]
    imgs_umass_1 = [i for i in imgs_umass_1 if 'ce-gad' not in str(i)]
    imgs_umass_1 += list(Path(path_umass1).rglob('*acq-FMPIR_T2w.nii.gz')) # We add acq-FMPIR_T2w (we leave a space in case multiple runs)
    imgs_umass_1 += list(Path(path_umass1).rglob('*acq-ax_T1w.nii.gz')) # We add acq-ax_T1w (we leave a space in case multiple runs)
    imgs_umass_1 += list(Path(path_umass1).rglob('*acq-ax_T2w.nii.gz')) # We add acq-ax_T2w (we leave a space in case multiple runs)
    imgs_umass_1 = [i for i in imgs_umass_1 if 'derivatives' not in str(i)]
    imgs_umass_1 = [str(i) for i in imgs_umass_1]
    ### Dictionnary is umass1[subject_id][session_id] = [list of images]
    for img in tqdm(imgs_umass_1):
        subject_id = img.split("/")[-1].split("_")[0]
        session_id = img.split("/")[-1].split("_")[1]
        if "ses-" not in session_id:
            continue
        if subject_id not in dict_umass1:
            dict_umass1[subject_id] = {}
        if session_id not in dict_umass1[subject_id]:
            dict_umass1[subject_id][session_id] = []
        dict_umass1[subject_id][session_id].append(img)
    # Then we remove from the dictionnary subjects with only one session
    dict_umass1 = {k: v for k, v in dict_umass1.items() if len(v) > 1}
    # Sort the sessions for each subject
    for subject_id in dict_umass1:
        dict_umass1[subject_id] = dict(sorted(dict_umass1[subject_id].items(), key=lambda x: x[0]))
    logger.info(f"Number of subjects in Umass1: {len(dict_umass1)}")

    ## Umass 2
    dict_umass2 = {}
    ### For umass2, we keep all T1w, acq-3D_T1w, acq-STIR_T2w and acq-axial_T2w images
    imgs_umass_2 = list(Path(path_umass2).rglob('*_T1w.nii.gz'))
    imgs_umass_2 = [i for i in imgs_umass_2 if '_ce-gad' not in str(i)]
    imgs_umass_2 = [i for i in imgs_umass_2 if 'acq-3D' not in str(i)]
    imgs_umass_2 += list(Path(path_umass2).rglob('*acq-3D_T1w.nii.gz')) # We add acq-3D_T1w (we leave a space in case multiple runs)
    imgs_umass_2 += list(Path(path_umass2).rglob('*acq-STIR_T2w.nii.gz')) # We add acq-STIR_T2w (we leave a space in case multiple runs)
    imgs_umass_2 += list(Path(path_umass2).rglob('*acq-axial_T2w.nii.gz')) # We add acq-axial_T2w (we leave a space in case multiple runs)
    imgs_umass_2 = [i for i in imgs_umass_2 if 'derivatives' not in str(i)]
    imgs_umass_2 = [i for i in imgs_umass_2 if 'SHA256' not in str(i)]
    imgs_umass_2 = [str(i) for i in imgs_umass_2]
    ### Dictionnary is umass2[subject_id][session_id] = [list of images]
    for img in tqdm(imgs_umass_2):
        subject_id = img.split("/")[-1].split("_")[0]
        session_id = img.split("/")[-1].split("_")[1]
        if "ses-" not in session_id:
            continue
        if subject_id not in dict_umass2:
            dict_umass2[subject_id] = {}
        if session_id not in dict_umass2[subject_id]:
            dict_umass2[subject_id][session_id] = []
        dict_umass2[subject_id][session_id].append(img)
    # Then we remove from the dictionnary subjects with only one session
    dict_umass2 = {k: v for k, v in dict_umass2.items() if len(v) > 1}
    # Sort the sessions for each subject
    for subject_id in dict_umass2:
        dict_umass2[subject_id] = dict(sorted(dict_umass2[subject_id].items(), key=lambda x: x[0]))
    logger.info(f"Number of subjects in Umass2: {len(dict_umass2)}")
    
    ## Umass 3
    dict_umass3 = {}
    ### For umass3, we keep all T1w, acq-3D_T1w, acq-STIR_T2w and acq-ax_T2w images
    imgs_umass_3 = list(Path(path_umass3).rglob('*T1w.nii.gz'))
    imgs_umass_3 = [i for i in imgs_umass_3 if '_ce-gad' not in str(i)]
    imgs_umass_3 += list(Path(path_umass3).rglob('*T2w.nii.gz')) # We add acq-3D_T1w (we leave a space in case multiple runs)
    imgs_umass_3 = [i for i in imgs_umass_3 if 'acq-STIR'not in str(i) and 'acq-ax' not in str(i)] # We keep only acq-STIR and acq-ax (we leave a space in case multiple runs)
    imgs_umass_3 += list(Path(path_umass3).rglob('*acq-STIR_T2w.nii.gz')) # We add acq-STIR_T2w (we leave a space in case multiple runs)
    imgs_umass_3 += list(Path(path_umass3).rglob('*acq-ax_T2w.nii.gz')) # We add acq-ax_T2w (we leave a space in case multiple runs)
    imgs_umass_3 = [i for i in imgs_umass_3 if 'derivatives' not in str(i)]
    imgs_umass_3 = [i for i in imgs_umass_3 if 'SHA256' not in str(i)]
    imgs_umass_3 = [str(i) for i in imgs_umass_3]
    ### Dictionnary is umass3[subject_id][session_id] = [list of images]
    for img in tqdm(imgs_umass_3):
        subject_id = img.split("/")[-1].split("_")[0]
        session_id = img.split("/")[-1].split("_")[1]
        if "ses-" not in session_id:
            continue
        if subject_id not in dict_umass3:
            dict_umass3[subject_id] = {}
        if session_id not in dict_umass3[subject_id]:
            dict_umass3[subject_id][session_id] = []
        dict_umass3[subject_id][session_id].append(img)
    # Then we remove from the dictionnary subjects with only one session
    dict_umass3 = {k: v for k, v in dict_umass3.items() if len(v) > 1}
    # Sort the sessions for each subject
    for subject_id in dict_umass3:
        dict_umass3[subject_id] = dict(sorted(dict_umass3[subject_id].items(), key=lambda x: x[0]))
    logger.info(f"Number of subjects in Umass3: {len(dict_umass3)}")

    ## Umass 4
    dict_umass4 = {}
    ### For umass4, we keep all T1w, acq-3D_T1w, acq-STIR_T2w and acq-ax_T2w images
    imgs_umass_4 = list(Path(path_umass4).rglob('*T1w.nii.gz'))
    imgs_umass_4 = [i for i in imgs_umass_4 if '_ce-gad' not in str(i)]
    imgs_umass_4 += list(Path(path_umass4).rglob('*T2w.nii.gz')) # We add acq-3D_T1w (we leave a space in case multiple runs)
    imgs_umass_4 = [i for i in imgs_umass_4 if 'acq-STIR'not in str(i) and 'acq-ax' not in str(i)] # We keep only acq-STIR and acq-ax (we leave a space in case multiple runs)
    imgs_umass_4 += list(Path(path_umass4).rglob('*acq-STIR_T2w.nii.gz')) # We add acq-STIR_T2w (we leave a space in case multiple runs)
    imgs_umass_4 += list(Path(path_umass4).rglob('*acq-ax_T2w.nii.gz')) # We add acq-ax_T2w (we leave a space in case multiple runs)
    imgs_umass_4 = [str(i) for i in imgs_umass_4]
    imgs_umass_4 = [i for i in imgs_umass_4 if 'SHA256' not in str(i)]
    ### Dictionnary is umass4[subject_id][session_id] = [list of images]
    for img in tqdm(imgs_umass_4):
        subject_id = img.split("/")[-1].split("_")[0]
        session_id = img.split("/")[-1].split("_")[1]
        if "ses-" not in session_id:
            continue
        if subject_id not in dict_umass4:
            dict_umass4[subject_id] = {}
        if session_id not in dict_umass4[subject_id]:
            dict_umass4[subject_id][session_id] = []
        dict_umass4[subject_id][session_id].append(img)
    # Then we remove from the dictionnary subjects with only one session
    dict_umass4 = {k: v for k, v in dict_umass4.items() if len(v) > 1}
    # Sort the sessions for each subject
    for subject_id in dict_umass4:
        dict_umass4[subject_id] = dict(sorted(dict_umass4[subject_id].items(), key=lambda x: x[0]))
    logger.info(f"Number of subjects in Umass4: {len(dict_umass4)}")

    ## BEIJING
    dict_beijing = {}
    ### For beijing, we keep all acq-sag_T1w, axTseRst_T2w and sagTseRst_T2w images
    ## MS-NMO-BEIJING
    imgs_beijing = list(Path(path_beijing).rglob('*acq-sag_*T1w.nii.gz'))  # We add acq-sag_T1w (we leave a space in case multiple runs)
    imgs_beijing += list(Path(path_beijing).rglob('*axTseRst_*T2w.nii.gz'))  # We add axTseRst_T2w (we leave a space in case multiple runs)
    imgs_beijing += list(Path(path_beijing).rglob('*sagTseRst_*T2w.nii.gz')) # We add sagTseRst_T2w (we leave a space in case multiple runs)
    imgs_beijing = [i for i in imgs_beijing if 'ocalizer' not in str(i)]
    imgs_beijing = [i for i in imgs_beijing if 'sub-MS' in str(i)]
    imgs_beijing = [str(i) for i in imgs_beijing]
    ### Dictionnary is beijing[subject_id][session_id] = [list of images]
    for img in tqdm(imgs_beijing):
        subject_id = img.split("/")[-1].split("_")[0]
        session_id = img.split("/")[-1].split("_")[1]
        if "ses-" not in session_id:
            continue
        if subject_id not in dict_beijing:
            dict_beijing[subject_id] = {}
        if session_id not in dict_beijing[subject_id]:
            dict_beijing[subject_id][session_id] = []
        dict_beijing[subject_id][session_id].append(img)
    # Then we remove from the dictionnary subjects with only one session
    dict_beijing = {k: v for k, v in dict_beijing.items() if len(v) > 1}
    # Sort the sessions for each subject
    for subject_id in dict_beijing:
        dict_beijing[subject_id] = dict(sorted(dict_beijing[subject_id].items(), key=lambda x: x[0]))
    logger.info(f"Number of subjects in Beijing: {len(dict_beijing)}")

    # We merge all dictionnaries
    dict_all = {**dict_bavaria, **dict_canproco, **dict_basel, **dict_karo, **dict_umass1, **dict_umass2, **dict_umass3, **dict_umass4, **dict_beijing}
    logger.info(f"Total number of subjects before exclusion: {len(dict_all)}")
    # We exclude subjects in the exclude list
    for sub in exclude_list:
        if sub.split("_")[0] in dict_all:
            del dict_all[sub.split("_")[0]]
    logger.info(f"Total number of subjects after exclusion: {len(dict_all)}")

    # Now we create the MSD-style JSON datalist
    json_dict = {}
    json_dict['name'] = 'Longitudinal_MS_lesion-agnostic'
    json_dict['description'] = 'This is a longitudinal multiple sclerosis dataset aggregated from multiple sources.'
    json_dict['modality'] = {"0": "MRI"}
    json_dict['data'] = dict_canproco if args.canproco_only else dict_all

    # We save the json file
    os.makedirs(args.path_out, exist_ok=True)
    json_path = os.path.join(args.path_out, f'dataset_{str(date.today())}.json')
    if args.canproco_only:
        json_path = os.path.join(args.path_out, f'dataset_canproco_only_{str(date.today())}.json')
    with open(json_path, 'w') as f:
        json.dump(json_dict, f, indent=4)
    logger.info(f"JSON file saved at {json_path}")

    return None


if __name__ == "__main__":
    main()