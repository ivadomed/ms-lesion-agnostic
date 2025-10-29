"""
This script performs registration of two images using multiple methods and computes the score of the registration.
Registration is done using SCT.

Input:
    -i1 : path to the input image at timepoint 1
    -i2 : path to the input image at timepoint 2
    -o : path to the output folder where registration results will be stored

Output:
    None
"""
import os
import argparse
from pathlib import Path
import sys
# Import the functions from utils in parent folder
file_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.abspath(os.path.join(file_path, ".."))
sys.path.insert(0, root_path)
from utils import segment_sc, segment_lesions

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i1', '--input_image1', type=str, required=True, help='Path to the input image at timepoint 1')
    parser.add_argument('-i2', '--input_image2', type=str, required=True, help='Path to the input image at timepoint 2')
    parser.add_argument('-o', '--output_folder', type=str, required=True, help='Path to the output folder where registration results will be stored')
    return parser.parse_args()


def register_multiple_methods(input_image1, input_image2, output_folder):
    """
    This function performs registration of two images using multiple methods.

    Inputs:
        input_image1 : path to the input image at timepoint 1
        input_image2 : path to the input image at timepoint 2
        output_folder : path to the output folder where registration results will be stored

    Outputs:
        None
    """
    # Build output directory
    os.makedirs(output_folder, exist_ok=True)

    return None


def main():
    args = parse_args()
    input_image1 = args.input_image1
    input_image2 = args.input_image2
    output_folder = args.output_folder

    register_multiple_methods(input_image1, input_image2, output_folder)

    return None


if __name__ == "__main__":
    main()