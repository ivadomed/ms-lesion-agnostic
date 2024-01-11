"""
This file contains the code for the data preprocessing to the nnUNet format.
It takes all available MS SC segmentation data and converts it to the nnUNet format.
It also creates the SC segmentation files using the contrast-agnostic model if the SC seg is not available.
The datasets used are the following:
- Canproco (PSIR, STIR)
- sct-testing-large (T1, T2 and T2*)
- bavaria-quebec-spine-ms : dataset is subject to change soon (T2w)
- nih-ms-mp2rage (MP2RAGE) : !!!!! Problem : lesions are note segmented for now !!!!!
- basel-mp2rage (MP2RAGE)

Args:
    --data-path : path to the datasets folder
    --datasets : list of datasets to use (sep = ",")
    --output-folder : path to the output folder
    --sc-seg-output : path to the SC segmentation output folder
    --taskname: Specify the task name - usually the anatomy to be segmented, e.g. Hippocampus
    --tasknumber : Specify the task number, has to be greater than 100 but less than 999
    --exclude-file : Path to the file containing the list of subjects to exclude from the dataset (default=None)

Returns:
    None

Example:
    python nnunet/convert_BIDS_to_nnunet.py --data-path /path/to/datasets --datasets canproco,sct-testing-large,bavaria-quebec-spine-ms,nih-ms-mp2rage,basel-mp2rage --output-folder /path/to/output
      --sc-seg-output /path/to/sc-seg-output --taskname MsLesionAgnostic --tasknumber 101 --exclude-file /path/to/exclude-file

TODO:
    *

Pierre-Louis Benveniste
"""

import os
import argparse
import numpy as np
import nibabel as nib
import shutil
import pathlib
from pathlib import Path
import json
from collections import OrderedDict
import run_inference_single_image


def get_parser():
    """
    This function parses the command line arguments and returns an argparse object.

    Input:
        None

    Returns:
        parser : argparse object
    """
    parser = argparse.ArgumentParser(description='Convert datasets to nnUNet format.')
    parser.add_argument('--data-path', type=str, required=True, help='Path to the datasets folder.')
    parser.add_argument('--datasets', type=str, required=True, help='List of datasets to use (sep = ",")')
    parser.add_argument('--output-folder', type=str, required=True, help='Path to the output folder.')
    parser.add_argument('--sc-seg-output', type=str, required=True, help='Path to the SC segmentation output folder.')
    parser.add_argument('--taskname', type=str, required=True, help='Specify the task name - usually the anatomy to be segmented, e.g. Hippocampus.')
    parser.add_argument('--tasknumber', type=int, required=True, help='Specify the task number, has to be greater than 100 but less than 999.')
    parser.add_argument('--exclude-file', type=str, required=False, default=None, help='Path to the file containing the list of subjects to exclude from the dataset (default=None).')   

    return parser


def create_sc_seg(image_file, sc_seg_output, path_to_model="/Users/plbenveniste/Documents/NeuroPoly/model_2023-09-18"):
    """
    This functions uses teh contrast-agnostic model to create the SC segmentation file.

    Input:
        image_file : path to the image file
        sc_seg_output : path to the SC segmentation output folder
    
    Returns:
        sc_seg_file : path to the SC segmentation file
    """

    # Create temporary output folder
    tmp_folder_seg = os.path.join(sc_seg_output, "tmp")
    pathlib.Path(tmp_folder_seg).mkdir(parents=True, exist_ok=True)

    os.system(f'python nnunet/run_inference_single_image.py --path-img {image_file}  --chkp-path {path_to_model} --path-out {tmp_folder_seg} --use-tta --')

    # Build output file name
    sc_seg_file = os.path.join(tmp_folder_seg, f'{str(image_file).split("/")[-1].split(".")[0]}_pred.nii.gz')

    return sc_seg_file


def main():
    """
    This function is the main function of the script.
    It parses the command line arguments, calls the functions to convert the datasets to nnUNet format and creates the SC segmentation files.

    Input:
        None

    Returns:
        None
    """
    # Parse the command line arguments
    parser = get_parser()
    args = parser.parse_args()

    # Get the command line arguments
    data_path = args.data_path
    datasets = args.datasets.split(",")
    output_folder = args.output_folder
    sc_seg_output = args.sc_seg_output
    taskname = args.taskname
    tasknumber = args.tasknumber
    exclude_file = args.exclude_file

    # Create the output folders
    path_out_imagesTr = Path(os.path.join(output_folder, 'imagesTr'))
    path_out_imagesTs = Path(os.path.join(output_folder, 'imagesTs'))
    path_out_labelsTr = Path(os.path.join(output_folder, 'labelsTr'))
    path_out_labelsTs = Path(os.path.join(output_folder, 'labelsTs'))

    # We load both train and validation set into the train images as nnunet uses cross-fold-validation
    train_images, train_labels = [], []
    test_images, test_labels = [], []

    # Make the directories
    pathlib.Path(output_folder).mkdir(parents=True, exist_ok=True)
    pathlib.Path(path_out_imagesTr).mkdir(parents=True, exist_ok=True)
    pathlib.Path(path_out_imagesTs).mkdir(parents=True, exist_ok=True)
    pathlib.Path(path_out_labelsTr).mkdir(parents=True, exist_ok=True)
    pathlib.Path(path_out_labelsTs).mkdir(parents=True, exist_ok=True)

    # Initialise the conversion dict
    conversion_dict = {}

    # Initialise the number of scans in train and in test folder
    scan_cnt_train, scan_cnt_test = 0, 0

    # Convert the datasets to nnUNet format
    for dataset in datasets:
        dataset_path = os.path.join(data_path, dataset)

        # Get all MS lesion segmtation files
        ms_lesion_files = list(Path(dataset_path).rglob(f'*_lesion-manual.nii.gz'))
        print(f"Found {len(ms_lesion_files)} MS lesion segmentation files in {dataset}")

        # For each file find corresponding image and SC segmentation file
        for lesion_seg_file in ms_lesion_files:
            # Get the subject name
            subject_name = lesion_seg_file.name.split("_lesion-manual.nii.gz")[0]
            
            # Find the corresponding image file
            image_file = list(Path(dataset_path).rglob(f'{subject_name}.nii.gz'))
            if len(image_file) == 0:
                print(f"Could not find image file for {subject_name}.")
                continue

            # Find the corresponding SC segmentation file
            sc_seg_file = list(Path(sc_seg_output).rglob(f'{subject_name}_seg-manual.nii.gz'))
            if len(sc_seg_file) == 0:
                # If no SC segmentation file is found, create it using the contrast-agnostic model
                sc_seg_file = create_sc_seg(image_file[0], sc_seg_output)

            # Chose if subject in test set or train set
            if np.random.rand() > 0.2:
                # Add to the train set
                scan_cnt_train+= 1
                
                image_file_nnunet_channel_1 = os.path.join(path_out_imagesTr,f'{args.taskname}_{scan_cnt_train:03d}_0000.nii.gz')
                image_file_nnunet_channel_2 = os.path.join(path_out_imagesTr,f'{args.taskname}_{scan_cnt_train:03d}_0001.nii.gz')
                label_file_nnunet = os.path.join(path_out_labelsTr,f'{args.taskname}_{scan_cnt_train:03d}.nii.gz')
                
                train_images.append(str(image_file_nnunet_channel_1))
                train_labels.append(str(label_file_nnunet))

                # Copy the image file
                shutil.copyfile(image_file[0], image_file_nnunet_channel_1)

                # Modify the header of channel 2 to have same spacing as channel 1
                spacing = nib.load(image_file_nnunet_channel_1).header.get_zooms()
                sc_seg_img = nib.load(sc_seg_file)
                sc_seg_img.header.set_zooms(spacing)
                nib.save(sc_seg_img, image_file_nnunet_channel_2)

                # Copy the label file AND image file
                shutil.copyfile(lesion_seg_file, label_file_nnunet)
            
                # Update the conversion dict (for label we only point to the lesion mask)
                conversion_dict[str(image_file)] = image_file_nnunet_channel_1
                conversion_dict[str(sc_seg_file)] = image_file_nnunet_channel_2
                conversion_dict[str(lesion_seg_file)] = label_file_nnunet
            
            else:
                # Add to the test set
                scan_cnt_test += 1

                # create the new convention names
                image_file_nnunet_channel_1 = os.path.join(path_out_imagesTs,f'{args.taskname}_{scan_cnt_test:03d}_0000.nii.gz')
                image_file_nnunet_channel_2 = os.path.join(path_out_imagesTs,f'{args.taskname}_{scan_cnt_test:03d}_0001.nii.gz')
                label_file_nnunet = os.path.join(path_out_labelsTs,f'{args.taskname}_{scan_cnt_test:03d}.nii.gz')
                
                test_images.append(str(image_file_nnunet_channel_1))
                test_labels.append(str(label_file_nnunet))
                
                # Copy the image file
                shutil.copyfile(image_file[0], image_file_nnunet_channel_1)

                # Modify the header of channel 2 to have same spacing as channel 1
                spacing = nib.load(image_file_nnunet_channel_1).header.get_zooms()
                sc_seg_img = nib.load(sc_seg_file)
                sc_seg_img.header.set_zooms(spacing)
                nib.save(sc_seg_img, image_file_nnunet_channel_2)

                # Copy the label file AND image file
                shutil.copyfile(lesion_seg_file, label_file_nnunet)
            
                # Update the conversion dict (for label we only point to the lesion mask)
                conversion_dict[str(image_file)] = image_file_nnunet_channel_1
                conversion_dict[str(sc_seg_file)] = image_file_nnunet_channel_2
                conversion_dict[str(lesion_seg_file)] = label_file_nnunet

    # Display of number of training and number of testing images
    print("Number of images for training: " + str(scan_cnt_train))
    print("Number of images for testing: " + str(scan_cnt_test))

    #----------------- CREATION OF THE DICTIONNARY-----------------------------------
    # Create dataset_description.json
    json_object = json.dumps(conversion_dict, indent=4)
    # Write to dataset description
    conversion_dict_name = f"conversion_dict.json"
    with open(os.path.join(output_folder, conversion_dict_name), "w") as outfile:
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
            "0": "MRI image (PSIR, STIR, T2w, MP2RAGE)",
            "1": "SC_seg",
        }
    
     # 0 is always the background. Any class labels should start from 1.
    json_dict['labels'] = {
        "background" : 0,
        "Lesion" : 1,
    }

    json_dict['numTraining'] = scan_cnt_train
    json_dict['numTest'] = scan_cnt_test
    #Newly required field in the json file with v2
    json_dict["file_ending"] = ".nii.gz"

    json_dict['training'] = [{'image': str(train_labels[i]).replace("labelsTr", "imagesTr") , "label": train_labels[i] }
                                 for i in range(len(train_images))]
    # Note: See https://github.com/MIC-DKFZ/nnUNet/issues/407 for how this should be described

    json_dict['test'] = [{'image': str(test_labels[i]).replace("labelsTs", "imagesTs") , "label": test_labels[i] }
                                 for i in range(len(test_images))]

    # create dataset_description.json
    json_object = json.dumps(json_dict, indent=4)
    # write to dataset description
    # nn-unet requires it to be "dataset.json"
    dataset_dict_name = f"dataset.json"
    with open(os.path.join(output_folder, dataset_dict_name), "w") as outfile:
        outfile.write(json_object)
    
    return None



if __name__ == "__main__":
    main()