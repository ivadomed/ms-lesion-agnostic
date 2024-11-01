"""
This script takes an MSD dataset and converts it to the nnU-Net format.

Arguments:
    -i: Path to the MSD dataset
    -o: Output directory
    --taskname: Name of the task
    --tasknumber: Number of the task

Returns:
    None

Example:
    python convert_msd_to_nnunet.py -i /path/to/MSD -o /path/to/output --taskname msLesionAgnostic --tasknumber 101

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
    parser.add_argument('-o', '--output', type=str, required=True, help='Output directory')
    parser.add_argument('--taskname', type=str, help='Name of the task', default='msLesionAgnostic')
    parser.add_argument('--tasknumber', type=int, required=True, help='Number of the task')
    return parser.parse_args()


def main():
    # Parse arguments
    args = parse_args()
    input_msd = args.input
    output_dir = args.output

    # Load the MSD dataset
    with open(input_msd, 'r') as f:
        msd_data_split = json.load(f)

    # Define the output path
    path_out = Path(os.path.join(output_dir, f'Dataset{args.tasknumber}_{args.taskname}'))
    
    # Define paths for train and test folders 
    path_out_imagesTr = Path(os.path.join(path_out, 'imagesTr'))
    path_out_imagesTs = Path(os.path.join(path_out, 'imagesTs'))
    path_out_labelsTr = Path(os.path.join(path_out, 'labelsTr'))
    path_out_labelsTs = Path(os.path.join(path_out, 'labelsTs'))

    # Load both train and validation set into the train images as nnunet uses cross-fold-validation
    train_images, train_labels = [], []
    test_images, test_labels = [], []

    # Make the directories
    path_out.mkdir(parents=True, exist_ok=True)
    path_out_imagesTr.mkdir(parents=True, exist_ok=True)
    path_out_imagesTs.mkdir(parents=True, exist_ok=True)
    path_out_labelsTr.mkdir(parents=True, exist_ok=True)
    path_out_labelsTs.mkdir(parents=True, exist_ok=True)

    # Initialise the conversion dict
    conversion_dict = {}

    # Initialise the number of scans in train and in test folder
    scan_cnt_train, scan_cnt_test = 0, 0
    
    # Load the json file
    train_data = msd_data_split['train'] + msd_data_split['validation']
    test_data = msd_data_split['test']

    # Iterate over all  training images
    for img_dict in tqdm.tqdm(train_data):
        #we update the count of images of the training set
        scan_cnt_train+= 1

        image_file_nnunet = os.path.join(path_out_imagesTr,f'{args.taskname}_{scan_cnt_train:03d}_0000.nii.gz')
        label_file_nnunet = os.path.join(path_out_labelsTr,f'{args.taskname}_{scan_cnt_train:03d}.nii.gz')
        
        train_images.append(str(image_file_nnunet))
        train_labels.append(str(label_file_nnunet))

        # Copy the image to the new structure
        shutil.copyfile(img_dict['image'], image_file_nnunet)

        # Binarize the label and save it to the adequate path
        label = nib.load(img_dict['label']).get_fdata()
        label[label > 0] = 1
        label = nib.Nifti1Image(label, nib.load(img_dict['label']).affine)
        nib.save(label, label_file_nnunet)

        # Update the conversion dict
        conversion_dict[str(os.path.abspath(img_dict['image']))] = image_file_nnunet
        conversion_dict[str(os.path.abspath(img_dict['label']))] = label_file_nnunet

        # For each label fils, we reorient them to the same orientation as the image using sct_register_multimodal -identity 1
        os.system(f"sct_register_multimodal -i {str(label_file_nnunet)} -d {str(image_file_nnunet)} -identity 1 -o {str(label_file_nnunet)} -owarp file_to_delete.nii.gz -owarpinv file_to_delete_2.nii.gz ")
        # Remove the other useless files
        os.system("rm file_to_delete.nii.gz file_to_delete_2.nii.gz")
        other_file_to_remove = str(label_file_nnunet).replace('.nii.gz', '_inv.nii.gz')
        os.system(f"rm {other_file_to_remove}")

        # Then we binarize the label
        os.system(f"sct_maths -i {str(label_file_nnunet)} -bin 0.5 -o {str(label_file_nnunet)}")

    
    # Iterate over all test images
    for img_dict in tqdm.tqdm(test_data):
        #we update the count of images of the test set
        scan_cnt_test+= 1

        image_file_nnunet = os.path.join(path_out_imagesTs,f'{args.taskname}_{scan_cnt_test:03d}_0000.nii.gz')
        label_file_nnunet = os.path.join(path_out_labelsTs,f'{args.taskname}_{scan_cnt_test:03d}.nii.gz')
        
        test_images.append(str(image_file_nnunet))
        test_labels.append(str(label_file_nnunet))

        # Copy the image to the new structure
        shutil.copyfile(img_dict['image'], image_file_nnunet)

        # Binarize the label and save it to the adequate path
        label = nib.load(img_dict['label']).get_fdata()
        label[label > 0] = 1
        label = nib.Nifti1Image(label, nib.load(img_dict['label']).affine)
        nib.save(label, label_file_nnunet)

        # Update the conversion dict
        conversion_dict[str(os.path.abspath(img_dict['image']))] = image_file_nnunet
        conversion_dict[str(os.path.abspath(img_dict['label']))] = label_file_nnunet

        # For each label fils, we reorient them to the same orientation as the image using sct_register_multimodal -identity 1
        os.system(f"sct_register_multimodal -i {str(label_file_nnunet)} -d {str(image_file_nnunet)} -identity 1 -o {str(label_file_nnunet)} -owarp file_to_delete.nii.gz -owarpinv file_to_delete_2.nii.gz ")
        # Remove the other useless files
        os.system("rm file_to_delete.nii.gz file_to_delete_2.nii.gz")
        other_file_to_remove = str(label_file_nnunet).replace('.nii.gz', '_inv.nii.gz')
        os.system(f"rm {other_file_to_remove}")

        # Then we binarize the label
        os.system(f"sct_maths -i {str(label_file_nnunet)} -bin 0. -o {str(label_file_nnunet)}")

    # Display of number of training and number of testing images
    print("Number of images for training: " + str(scan_cnt_train))
    print("Number of images for testing: " + str(scan_cnt_test))

    #----------------- CREATION OF THE DICTIONNARY-----------------------------------
    # create dataset_description.json
    json_object = json.dumps(conversion_dict, indent=4)
    # write to dataset description
    conversion_dict_name = f"conversion_dict.json"
    with open(os.path.join(path_out, conversion_dict_name), "w") as outfile:
        outfile.write(json_object)

    # c.f. dataset json generation. This contains the metadata for the dataset that nnUNet uses during preprocessing and training
    # general info : https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/dataset_conversion/utils.py
    # example: https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/dataset_conversion/Task055_SegTHOR.py

    json_dict = OrderedDict()
    json_dict['name'] = args.taskname
    json_dict['description'] = args.taskname
    json_dict['tensorImageSize'] = "3D"
    json_dict['reference'] = "TBD"
    json_dict['licence'] = "TBD"
    json_dict['release'] = "0.0"
    
    # Because only using one modality  
    ## was changed from 'modality' to 'channel_names'
    json_dict['channel_names'] = {
            "0": "MRI",
        }
    
     # 0 is always the background. Any class labels should start from 1.
    json_dict['labels'] = {
        "background" : 0,
        "lesion" : 1,
    }
   
    # json_dict['regions_class_order'] = [1,2]

    json_dict['numTraining'] = scan_cnt_train
    json_dict['numTest'] = scan_cnt_test
    #Newly required field in the json file with v2
    json_dict["file_ending"] = ".nii.gz"

    json_dict['training'] = [{'image': str(train_labels[i]).replace("labelsTr", "imagesTr") , "label": train_labels[i] }
                                 for i in range(len(train_images))]
    # Note: See https://github.com/MIC-DKFZ/nnUNet/issues/407 for how this should be described

    #Removed because useless in this case
    json_dict['test'] = [{'image': str(test_labels[i]).replace("labelsTs", "imagesTs") , "label": test_labels[i] }
                                 for i in range(len(test_images))]

    # create dataset_description.json
    json_object = json.dumps(json_dict, indent=4)
    # write to dataset description
    # nn-unet requires it to be "dataset.json"
    dataset_dict_name = f"dataset.json"
    with open(os.path.join(path_out, dataset_dict_name), "w") as outfile:
        outfile.write(json_object)

    return None


if __name__ == '__main__':
    main()







    

    

