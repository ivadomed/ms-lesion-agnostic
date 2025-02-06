"""
This script takes an MSD dataset and converts the external test set to the nnU-Net format with reorientation.

Arguments:
    -i: Path to the MSD dataset
    -o: Output directory
    --taskname: Name of the task
    --tasknumber: Number of the task

Returns:
    None

Example:
    python convert_msd_to_nnunet.py -i /path/to/MSD -o /path/to/output --taskname msLesionAgnostic --tasknumber 101

TODO: This script could be integrated in the script convert_msd_to_nnunet_reorient.py

Author: Pierre-Louis Benveniste    
"""

import os
import shutil
import argparse
import json
from pathlib import Path
import tqdm
import nibabel as nib
import numpy as np
from collections import OrderedDict



def parse_args():
    parser = argparse.ArgumentParser(description='Convert MSD dataset to nnU-Net format')
    parser.add_argument('-i', '--input', type=str, required=True, help='Path to the MSD dataset json file')
    parser.add_argument('-o', '--output', type=str, required=True, help='Output directory. It will create a folder called imagesExternalTs in the output directory and labelsExternalTs')
    parser.add_argument('--taskname', type=str, help='Name of the task', default='msLesionAgnostic')    
    return parser.parse_args()


def main():
    # Parse arguments
    args = parse_args()
    input_msd = args.input
    output_dir = args.output

    # Load the MSD dataset
    with open(input_msd, 'r') as f:
        msd_data_split = json.load(f)

    path_out = Path(output_dir)
    
    # Define paths for train and test folders 
    path_out_imagesExternalTs = Path(os.path.join(path_out, 'imagesExternalTs'))
    path_out_labelsExternalTs = Path(os.path.join(path_out, 'labelsExternalTs'))

    # Make the directories
    path_out.mkdir(parents=True, exist_ok=True)
    path_out_imagesExternalTs.mkdir(parents=True, exist_ok=True)
    path_out_labelsExternalTs.mkdir(parents=True, exist_ok=True)

    # Initialise the conversion dict
    conversion_dict = {}
    
    # initialise the count of images
    scan_cnt_ext = 0
    
    # Load the json file
    external_data = msd_data_split['externalValidation']

    # Initialise the lists for the external test set
    ext_images = []
    ext_labels = []


    # Iterate over all  training images
    for img_dict in tqdm.tqdm(external_data):

        scan_cnt_ext += 1

        image_file_nnunet = os.path.join(path_out_imagesExternalTs,f'{args.taskname}_{scan_cnt_ext:03d}_0000.nii.gz')
        label_file_nnunet = os.path.join(path_out_labelsExternalTs,f'{args.taskname}_{scan_cnt_ext:03d}.nii.gz')
        
        ext_images.append(str(image_file_nnunet))
        ext_labels.append(str(label_file_nnunet))

        # Instead of copying we will reorient the image to RPI
        assert os.system(f"sct_image -i {img_dict['image']} -setorient RPI -o {image_file_nnunet}") ==0

        # Binarize the label and save it to the adequate path
        label = nib.load(img_dict['label']).get_fdata()
        label[label > 0] = 1
        label = nib.Nifti1Image(label, nib.load(img_dict['label']).affine)
        nib.save(label, label_file_nnunet)
        # Then we reorient the label to RPI
        assert os.system(f"sct_image -i {label_file_nnunet} -setorient RPI -o {label_file_nnunet}") ==0

        # Update the conversion dict
        conversion_dict[str(os.path.abspath(img_dict['image']))] = image_file_nnunet
        conversion_dict[str(os.path.abspath(img_dict['label']))] = label_file_nnunet

        # For each label fils, we reorient them to the same orientation as the image using sct_register_multimodal -identity 1
        assert os.system(f"sct_register_multimodal -i {str(label_file_nnunet)} -d {str(image_file_nnunet)} -identity 1 -o {str(label_file_nnunet)} -owarp file_to_delete.nii.gz -owarpinv file_to_delete_2.nii.gz ") ==0
        # Remove the other useless files
        assert os.system("rm file_to_delete.nii.gz file_to_delete_2.nii.gz") ==0
        other_file_to_remove = str(label_file_nnunet).replace('.nii.gz', '_inv.nii.gz')
        assert os.system(f"rm {other_file_to_remove}") ==0

        # Then we binarize the label
        assert os.system(f"sct_maths -i {str(label_file_nnunet)} -bin 0.5 -o {str(label_file_nnunet)}") ==0

    # Display of number of training and number of testing images
    print("Number of images for external validation: " + str(scan_cnt_ext))

    #----------------- CREATION OF THE DICTIONNARY-----------------------------------
    # create dataset_description.json
    json_object = json.dumps(conversion_dict, indent=4)
    # write to dataset description
    conversion_dict_name = f"conversion_dict_external_validation.json"
    with open(os.path.join(path_out, conversion_dict_name), "w") as outfile:
        outfile.write(json_object)

    return None


if __name__ == '__main__':
    main()