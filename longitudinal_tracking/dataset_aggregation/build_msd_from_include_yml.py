"""
This file creates the MSD-style JSON datalist to train an nnunet model using monai. 

Arguments:
    --include-yml: path to the yaml file containing the list of subjects to include
    --dataset-path: path to the root dataset folder
    --output-path: path to the output folder

Example:
    python build_msd_from_include_yml.py --include-yml /path/to/include.yml --dataset-path /path/to/dataset --output-path /path/to/output/folder
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

    parser = argparse.ArgumentParser(description="Create MSD-style JSON datalist for longitudinal MS dataset aggregation.")
    parser.add_argument("--include-yml", type=str, required=False, default=None,
                        help="Path to the yaml file containing the list of subjects to include.")
    parser.add_argument("--dataset-path", type=str, required=True,
                        help="Path to the root dataset folder containing all datasets.")
    parser.add_argument("-o", "--path-out", type=str, required=True,
                        help="Path to the output directory where the JSON file will be saved.")
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
    seed = 42

    # Save the logger to a file:
    os.makedirs(args.path_out, exist_ok=True)
    logger.add(os.path.join(args.path_out, f'logger_{str(date.today())}.log'))
    # Add the command line to the logger
    logger.info(f"Command line: python {' '.join(os.sys.argv)}")
    
    # Load the include yml file
    with open(args.include_yml, 'r') as f:
        include_dict = yaml.safe_load(f)
    include_list = include_dict["include"]

    # For each file in the include list, we found the corresponding image in the dataset path
    dict_imgs = {}
    for img in include_list:
        # Extract info:
        contrast = img.split("_")[-1]
        subject_id = img.split("_")[0]
        # We find the first timepoint
        baseline_img = os.path.join(args.dataset_path, "canproco", subject_id, "ses-M0", "anat",  f"{subject_id}_ses-M0_{contrast}.nii.gz")
        # We find the follow-up timepoint
        followup_img = os.path.join(args.dataset_path, "canproco", subject_id, "ses-M12","anat", f"{subject_id}_ses-M12_{contrast}.nii.gz")
        # We add to the dict
        dict_imgs[subject_id] = {}
        dict_imgs[subject_id]["ses-M0"] = [baseline_img]
        dict_imgs[subject_id]["ses-M12"] = [followup_img]
    logger.info(f"Number of subjects in Bavaria: {len(dict_imgs)}")

    # Now we create the MSD-style JSON datalist
    json_dict = {}
    json_dict['name'] = 'Longitudinal_MS_lesion-agnostic'
    json_dict['description'] = 'This is a longitudinal multiple sclerosis dataset aggregated from multiple sources.'
    json_dict['modality'] = {"0": "MRI"}
    json_dict['data'] = dict_imgs

    # We save the json file
    os.makedirs(args.path_out, exist_ok=True)
    json_path = os.path.join(args.path_out, f'dataset_include_{str(date.today())}.json')
    with open(json_path, 'w') as f:
        json.dump(json_dict, f, indent=4)
    logger.info(f"JSON file saved at {json_path}")

    return None


if __name__ == "__main__":
    main()