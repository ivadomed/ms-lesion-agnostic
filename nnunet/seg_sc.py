""" Spinal cord segmentation of SC MS lesion images

This script segments the spinal cord of all the images in the dataset which have a lesion mask. 
It uses the contrasts agnostic model to do it. 

TIP: 
    For canproco : suffix-lesion = '_lesion-manual', suffix-sc = '_seg-manual'
    For bavaria : suffix-lesion = '_lesions-manual', suffix-sc = '_seg-manual'
    For basel : suffix-lesion = '_lesion-manualNeuroPoly', suffix-sc = '_label-SC_seg'
    For sct-testing-large : suffix-lesion = '_lesion-manual', suffix-sc = '_seg-manual'
    WARNING: for bavaria, the lesion mask is like this : sub-m023917_ses-20130506_acq-ax_lesions-manual_T2w.nii.gz (with the _T2w at the end)

Example of run:
    $ python seg_sc.py --data-path /path/to/data --model-path /path/to/model --qc-path /path/to/qc --suffix-lesion _lesion-manual --suffix-sc _seg-manual

Args:
    --data-path: path to the dataset
    --model-path: path to the model
    --qc-path: path to the qc folder
    --suffix-lesion: suffix of the lesion mask
    --suffix-sc: suffix of the spinal cord mask

Returns:
    None

TODO:
    *

Pierre-Louis Benveniste
"""

import os
import argparse
from pathlib import Path
import shutil
import time

def get_parser():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Segment spinal cord using contrasts agnostic model')
    parser.add_argument('--data-path', help='Path to the dataset folder', required=True ,type=str)
    parser.add_argument('--model-path', help='Path to the model', required=True, type=str)
    parser.add_argument('--qc-path', help='Path to the qc folder', required=True, type=str)
    parser.add_argument('--suffix-lesion', help='Suffix of the lesion mask', required=True, default='_lesion-manual', type=str)
    parser.add_argument('--suffix-sc', help='Suffix of the spinal cord mask', required=True, default='_seg-manual', type=str)
    return parser

def main():
    """
    This function is the main function of the script. 
    It uses the contrasts agnostic model to segment the spinal cord of all the images in the dataset which have a lesion mask.

    Args:
        None
    
    Returns:
        None
    """

    # Parse command line arguments
    parser = get_parser()
    args = parser.parse_args()

    # Get input arguments
    path_data = args.data_path
    path_model = args.model_path
    path_qc = args.qc_path
    suffix_lesion = args.suffix_lesion
    suffix_sc = args.suffix_sc

    # Create the qc folder if it does not exist
    if not os.path.exists(path_qc):
        os.makedirs(path_qc)

    # Get the list of images to segment
    list_lesion_mask = list(Path(path_data).rglob(f'*{suffix_lesion}*'))

    # Get the corresponding images
    list_corresponding_image = [str(lesion_mask).replace(suffix_lesion, '') for lesion_mask in list_lesion_mask]
    list_corresponding_image = [str(image).replace('/derivatives/labels','') for image in list_corresponding_image]

    # Iterate over the images
    for image in list_corresponding_image:
        # Get image name
        image_name = str(os.path.basename(image)).replace('.nii.gz', '')

        print(f'Segmenting {image_name}')

        # Find corresponding lesion mask
        if 'basel' in path_data:
            corresponding_lesion_mask = [str(lesion_mask) for lesion_mask in list_lesion_mask if str(image_name).replace('_T2w','') in str(lesion_mask).replace('_T2w','')][0]
        else :
            corresponding_lesion_mask = [str(lesion_mask) for lesion_mask in list_lesion_mask if str(image_name) in str(lesion_mask)][0]

        # Build temp folder in qc_folder
        tmp_folder_seg = os.path.join(path_qc, f'{image_name}_tmp_seg')
        if not os.path.exists(tmp_folder_seg):
            os.makedirs(tmp_folder_seg)
        
        # copy image to tmp folder
        image_copy = os.path.join(tmp_folder_seg, image_name)
        shutil.copyfile(image, image_copy)

        # Segment the spinal cord
        os.system(f'python nnunet/run_inference_single_image.py --path-img {image_copy}  --chkp-path {path_model} --path-out {tmp_folder_seg} --use-tta --remove-small-objects 10')

        # Build prediction file name and mask file name
        pred_file = os.path.join(tmp_folder_seg, f'{str(image_copy).split("/")[-1].split(".")[0]}_pred.nii.gz')
        if 'bavaria' in path_data:  
            mask_file_name= f'{str(image_copy).split("/")[-1].replace("_T2w", "seg-manual_T2w")}'
        
        else: 
            mask_file_name =  f'{str(image_copy).split("/")[-1].split(".")[0]}{suffix_sc}.nii.gz'

        # Build path to mask file
        mask_file = os.path.join(os.path.dirname(corresponding_lesion_mask), mask_file_name)

        # Copy prediction file to mask file location
        shutil.copyfile(pred_file, mask_file)

        # Remove tmp folder
        shutil.rmtree(tmp_folder_seg)
        
        # Create QC folder
        os.system(f'sct_qc -i {image_copy} -p "sct_deepseg_sc" -s {mask_file} -qc {path_qc}')
    
    return None


if __name__ == "__main__":
    main()








    