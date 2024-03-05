"""
Makes a list of all the scans in BIDS database that have a lesion segmentation file (*lesion-manual.nii.gz),
Splits that list into 3 datasets (train, test and val).
Saves a json file containing the lists of test, train and val datasets.

Here is an example of the json format:
    {"train": ["sub-cal056_ses-M12_STIR",
                "sub-edm011_ses-M0_PSIR",
                "sub-cal072_ses-M0_STIR"],
    "val": ["sub-cal157_ses-M12_STIR"],
    "test": ["sub-edm076_ses-M0_PSIR"]}
"""

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import glob
import json
import os
from pathlib import Path
from sklearn.model_selection import train_test_split


def _train_test_val_split(list_to_split:list, train_size:float, val_size:float):
    """
    Splits a list into 3 lists (test, train and val) using the specified ratios.

    Args:
        list_to_split (list): List of scan names to split into datasets
        train_size (float): Proportion of scans for training (between 0 and 1)
        val_size (float): Proportion of scans for validation (between 0 and 1)

    Returns:
        train (list): List of scans in training set
        test (list): List of scans in testing set
        val (list): List of scans in validation set
    """

    train, test_val= train_test_split(list_to_split,random_state=0, test_size= round(1-train_size, 3))
    val, test= train_test_split(test_val,random_state=0, test_size= (round((1-train_size - val_size)/(1-train_size), 3)))

    return train, test, val


def _main():
    parser = ArgumentParser(
    prog = 'train_test_val_from_BIDS',
    description = 'Saves a json file containing the lists of test, train and val datasets',
    formatter_class = ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--database',
                        required = True,
                        type = Path,
                        help = 'Path to BIDS database')
    parser.add_argument('-o', '--output-path',
                        required = True,
                        type = Path,
                        help = 'Output path to json file where lists will be saved. Must end with .json')
    parser.add_argument('-t', '--train-size',
                        default = 0.8,
                        type = float,
                        help = 'Proportion of dataset to use for training')
    parser.add_argument('-v', '--val-size',
                        default = 0.1,
                        type = float,
                        help = 'Proportion of dataset to use for validation')

    args = parser.parse_args()

    directory_path = args.input_path

    # Find *_lesion-manual.nii.gz files
    search_pattern = os.path.join(directory_path, "derivatives", "labels", "**", "**", "anat", "*_lesion-manual.nii.gz")
    matching_files = glob.glob(search_pattern)

    # Get volume names
    volume_list = []
    for path in matching_files:
        volume_name = Path(path).stem[:-len("_lesion-manual.nii")]
        volume_list.append(volume_name)
    
    # Split into train, test and val sets
    train, test, val = _train_test_val_split(volume_list, args.train_size, args.val_size)

    data_dict = {"train": train,
                 "test": test,
                 "val": val}

    # Save to json file
    output_dir = args.output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.output_path, "w") as outfile:
        json.dump(data_dict, outfile)


if __name__ == "__main__":
    _main()