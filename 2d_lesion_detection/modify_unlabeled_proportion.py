"""
Script for changing the proportion of unlabeled slices (i.e. slices that contain no lesion)
in the train set from a yolo dataset
"""
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import os
from pathlib import Path
import random
import shutil


def _main():
    parser = ArgumentParser(
    prog = 'modify_unlabeled_proportion',
    description = 'From a yolo dataset, change the proportion of slices that '
                  'contain no lesion (unlabeled) in the train set',
    formatter_class = ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input-path',
                        required = True,
                        type = Path,
                        help = 'Path to existing YOLO database')
    parser.add_argument('-o', '--output-path',
                        required = True,
                        type = Path,
                        help = 'Path to new YOLO database')
    parser.add_argument('-r', '--ratio',
                        default = 0.25,
                        type = float,
                        help = 'Proportion of dataset that should be unlabeled')

    args = parser.parse_args()

    all_train = os.listdir(args.input_path/"images"/"train")
    all_train = [file for file in all_train if not file.startswith(".")] #remove hidden files
    all_train = [filename.replace(".png", "") for filename in all_train] #remove extension
    print(all_train)
    print(f"all_train len: {len(all_train)}")

    all_labels = os.listdir(args.input_path/"labels"/"train")
    all_labels = [file for file in all_labels if not file.startswith(".")] #remove hidden files
    all_labels = [filename.replace(".txt", "") for filename in all_labels] #remove extension
    print(all_labels)
    print(f"all_labels len: {len(all_labels)}")

    all_unlabelled = [filename for filename in all_train if filename not in all_labels]
    print(all_unlabelled)
    print(f"all_unlabelled len: {len(all_unlabelled)}")

    random.seed(10)
    n_unlabelled_to_copy = (len(all_labels)*args.ratio)/(1-args.ratio)
    unlabelled_to_copy = random.sample(all_unlabelled, int(n_unlabelled_to_copy))
    print(unlabelled_to_copy)
    print(f"unlabelled_to_copy len: {len(unlabelled_to_copy)}")

    # Transfer files to new dataset folder
    shutil.copytree(args.input_path/"labels", args.output_path/"labels") # all labels
    shutil.copytree(args.input_path/"images"/"test", args.output_path/"images"/"test") # images in test
    shutil.copytree(args.input_path/"images"/"val", args.output_path/"images"/"val") # images in val

    os.makedirs(args.output_path/"images"/"train", exist_ok=True)
    # copy over the unlabelled images that have been selected
    for file in unlabelled_to_copy:
        print(f"Copying over {file}")
        source_file_path = args.input_path/"images"/"train"/(file + ".png")
        destination_file_path = args.output_path/"images"/"train"/(file + ".png")
        shutil.copy(source_file_path, destination_file_path)

    # copy over all images that are labelled
    for file in all_labels:
        print(f"Copying over {file}")
        source_file_path = args.input_path/"images"/"train"/(file + ".png")
        destination_file_path = args.output_path/"images"/"train"/(file + ".png")
        shutil.copy(source_file_path, destination_file_path)


if __name__ == "__main__":
    _main()
