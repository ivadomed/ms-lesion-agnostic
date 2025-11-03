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
import nibabel as nib
import numpy as np
import sklearn.metrics as skl_metrics

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


def register(input_image, reference_image, input_sc_seg, reference_sc_seg, output_folder, method, qc_folder):
    """
    This function performs registration of an input image to a reference image using SCT.

    Inputs:
        input_image : path to the input image to be registered
        reference_image : path to the reference image
        input_sc_seg : path to the spinal cord segmentation of the input image
        reference_sc_seg : path to the spinal cord segmentation of the reference image
        output_folder : path to the output folder where registration results will be stored
        method : registration method to be used
        qc_folder : path to the QC folder

    Outputs:
        None
    """
    os.makedirs(output_folder, exist_ok=True)
    # Build output_file
    output_file = Path(output_folder) / "img2_registered_2_img1.nii.gz"
    
    # Build command
    command = f"sct_register_multimodal -i {input_image} -d {reference_image} -iseg {input_sc_seg} -dseg {reference_sc_seg} -ofolder {output_folder} -o {output_file} -param {method} -qc {qc_folder}"
    # Execute command
    assert os.system(command) == 0

    return output_file


def mutual_information(x, y, nbins=32, normalized=False):
    """
    Compute mutual information
    # This function comes from SCT: https://github.com/spinalcordtoolbox/spinalcordtoolbox/blob/724009e694cafe8d141a890cc9c7145ffe9f983a/spinalcordtoolbox/math.py#L246

    :param x: 1D numpy.array : flatten data from an image
    :param y: 1D numpy.array : flatten data from an image
    :param nbins: number of bins to compute the contingency matrix (only used if normalized=False)
    :return: float non negative value : mutual information
    """
    if normalized:
        mi = skl_metrics.normalized_mutual_info_score(x, y)
    else:
        c_xy = np.histogram2d(x, y, nbins)[0]
        mi = skl_metrics.mutual_info_score(None, None, contingency=c_xy)
    return mi


def compute_registration_score(registered_image, reference_image):
    """
    This function computes a registration score between the registered image and the reference image.

    Inputs:
        registered_image : path to the registered image
        reference_image : path to the reference image
    Outputs:
        mse : mean squared error between the registered image and the reference image
        mi : mutual information between the registered image and the reference image
    """
    # Here we use a simple metric: mean squared error between the two images
    reg_img = nib.load(registered_image).get_fdata()
    ref_img = nib.load(reference_image).get_fdata()

    # Ensure both images have the same shape
    assert reg_img.shape == ref_img.shape, "Registered image and reference image must have the same shape"

    # Compute mean squared error
    mse = np.mean((reg_img - ref_img) ** 2)

    # To compute MI, we need to flatten the images
    reg_img_flat = reg_img.flatten()
    ref_img_flat = ref_img.flatten()
    # Compute mutual information also
    mi = mutual_information(ref_img_flat, reg_img_flat)
    return mse, mi


def register_multiple_methods(input_image1, input_image2, output_folder, qc_folder):
    """
    This function performs registration of two images using multiple methods.

    Inputs:
        input_image1 : path to the input image at timepoint 1
        input_image2 : path to the input image at timepoint 2
        output_folder : path to the output folder where registration results will be stored

    Outputs:
        scores : list of tuples containing (mse, mi, registered_file, method) for each registration method
    """
    # Build output directory
    os.makedirs(output_folder, exist_ok=True)
    # Build the QC folder
    os.makedirs(qc_folder, exist_ok=True)

    # First we segment the spinal cord in both images
    image_1_name = Path(input_image1).name
    sc_seg_1 = os.path.join(output_folder, image_1_name.replace('.nii.gz', '_sc_seg.nii.gz'))
    image_2_name = Path(input_image2).name
    sc_seg_2 = os.path.join(output_folder, image_2_name.replace('.nii.gz', '_sc_seg.nii.gz'))
    # Segment the spinal cord
    segment_sc(input_image1, sc_seg_1)
    segment_sc(input_image2, sc_seg_2)
    
    # Then we perform registration using multiple strategies
    methods = ['step=1,type=im,algo=dl',
               'step=1,type=seg,algo=slicereg,metric=MeanSquares',
               'step=1,type=seg,algo=slicereg,metric=MeanSquares:step=2,type=im,algo=dl',
               'step=1,type=seg,algo=slicereg,metric=MeanSquares:step=2,type=seg,algo=affine,metric=MeanSquares,gradStep=0.2:step=3,type=im,algo=syn,metric=MI,iter=5,shrink=2']
    # Initialize variables to store best scores
    scores = []
    # For each method, we register image 2 to image 1
    for i, method in enumerate(methods):
        output_register_method_i = os.path.join(output_folder, f'registered_method_{i+1}')
        registered_file = register(input_image2, input_image1, sc_seg_2, sc_seg_1, output_register_method_i, method, qc_folder)
        # Now we compute the registration "score"
        mse, mi = compute_registration_score(registered_file, input_image1)
        scores.append((mse, mi, registered_file, f'method_{i+1}'))

    return scores


def save_registration_results(input_image1, input_image2, scores, output_folder):
    """
    This function saves the registration results to a csv file.

    Inputs:
        input_image1 : path to the input image at timepoint 1
        input_image2 : path to the input image at timepoint 2
        scores : list of tuples containing (mse, mi, registered_file, method) for each registration method
        output_folder : path to the output folder where registration results will be stored

    Outputs:
        None
    """
    results_file = os.path.join(output_folder, "registration_results.csv")

    with open(results_file, 'w') as f:
        f.write(f"Input Image 1,Input Image 2,Method,MSE,MI\n")
        for score in scores:
            f.write(f"{input_image1},{input_image2},{score[3]},{score[0]},{score[1]}\n")
    return None


def main():
    args = parse_args()
    input_image1 = args.input_image1
    input_image2 = args.input_image2
    output_folder = args.output_folder

    # Create a QC folder inside the output folder
    qc_folder = os.path.join(output_folder, "qc")
    os.makedirs(qc_folder, exist_ok=True)

    # Perform registration using multiple methods
    scores = register_multiple_methods(input_image1, input_image2, output_folder, qc_folder)

    # Print and format the scores
    for i, score in enumerate(scores):
        print(f"Method {i+1}: MSE = {score[0]}, MI = {score[1]}, Registered File = {score[2]}, Method = {score[3]}")

    # Save registration results
    save_registration_results(input_image1, input_image2, scores, output_folder)

    return None


if __name__ == "__main__":
    main()