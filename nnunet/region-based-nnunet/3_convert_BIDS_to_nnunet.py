"""
Convert data from BIDS to nnU-Net format
This python script converts data from the BIDS format to the nnU-Net format in order to be able to perform pre-processing, training and inference.

Example of run:
    $ python convert_BIDS_to_nnunet.py --path-data-json /path/to/datasplit/json --path-out /path/to/nnUNet_raw --taskname TASK-NAME --tasknumber DATASET-ID

Arguments:
    --path-data-json : Path to BIDS structured dataset. Accepts both cross-sectional and longitudinal datasets
    --path-out : Path to output directory.
    --taskname: Specify the task name - usually the anatomy to be segmented, e.g. Hippocampus
    --tasknumber : Specify the task number, has to be greater than 100 but less than 999

Returns:
    None
    
Todo:
    * 
Pierre-Louis Benveniste
"""

import argparse
import pathlib
from pathlib import Path
import json
import os
import shutil
from collections import OrderedDict

import numpy as np
import tqdm
import nibabel as nib


def get_parser():
    """
    This function parses the command line arguments and returns an argparse object.

    Input:
        None

    Returns:
        parser : argparse object
    """
    parser = argparse.ArgumentParser(description='Convert BIDS-structured database to nnUNet format.')
    parser.add_argument('--path-data-json', required=True,
                        help='Path to the json file describing the training and testing split')
    parser.add_argument('--path-out', help='Path to output directory.', required=True)
    parser.add_argument('--taskname', default='MSSpineLesion', type=str,
                        help='Specify the task name - usually the anatomy to be segmented, e.g. Hippocampus',)
    parser.add_argument('--tasknumber', default=501,type=int, 
                        help='Specify the task number, has to be greater than 500 but less than 999. e.g 502')

    return parser

def  create_multi_label_mask(lesion_mask_file, sc_seg_file, label_file_nnunet, label_threshold=0.5):
    """
    This function creates a multi-label mask for a given lesion mask and spinal cord segmentation mask.
    It also removes the lesions and the spinal cord segmentation which are above the first vertebral level.
    It saves the multi-label mask in the destination folder.

    Input:
        lesion_mask_file : Path to the lesion mask file
        sc_seg_file : Path to the spinal cord segmentation mask file
        label_file_nnunet : Path to the multi-label mask file

    Returns:
        None
    """
    # first for both spinal cord and lesion, we threshold and binarize everything at 1
    # for the lesion mask
    lesion_mask = nib.load(lesion_mask_file)
    lesion_affine = lesion_mask.affine
    lesion_header = lesion_mask.header
    lesion_mask = np.asarray(lesion_mask.dataobj)
    lesion_mask = np.where(lesion_mask > label_threshold, 1, 0)

    #we load the disc levels mask
    sc_seg = nib.load(sc_seg_file)
    sc_seg = np.asarray(sc_seg.dataobj)
    sc_seg = np.where(sc_seg > label_threshold, 1, 0)
    
    #we create the multi-label
    multi_label = np.zeros(lesion_mask.shape,dtype=np.int16)
    multi_label[sc_seg==1] = 1
    multi_label[lesion_mask==1] = 2

    #we save it in the destination folder
    multi_label_file = nib.Nifti1Image(multi_label, lesion_affine, lesion_header)
    nib.save(multi_label_file, str(label_file_nnunet))
    return None


def main():
    """
    This functions builds a dataset for training in the nnunet format.

    Input:
        args : Arguments of the script
    
    Returns:
        None
    """
    args = get_parser().parse_args()
    #------------- DEFINITION OF THE PATHS --------------------------
    path_data_json = Path(args.path_data_json)
    path_out = Path(os.path.join(os.path.abspath(args.path_out), f'Dataset{args.tasknumber}_{args.taskname}'))

    # define paths for train and test folders 
    path_out_imagesTr = Path(os.path.join(path_out, 'imagesTr'))
    path_out_imagesTs = Path(os.path.join(path_out, 'imagesTs'))
    path_out_labelsTr = Path(os.path.join(path_out, 'labelsTr'))
    path_out_labelsTs = Path(os.path.join(path_out, 'labelsTs'))

    # we load both train and validation set into the train images as nnunet uses cross-fold-validation
    train_images, train_labels = [], []
    test_images, test_labels = [], []

    # make the directories
    pathlib.Path(path_out).mkdir(parents=True, exist_ok=True)
    pathlib.Path(path_out_imagesTr).mkdir(parents=True, exist_ok=True)
    pathlib.Path(path_out_imagesTs).mkdir(parents=True, exist_ok=True)
    pathlib.Path(path_out_labelsTr).mkdir(parents=True, exist_ok=True)
    pathlib.Path(path_out_labelsTs).mkdir(parents=True, exist_ok=True)

    #make the directories for the test set
    pathlib.Path(os.path.join(path_out_imagesTs, 'canproco')).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.join(path_out_labelsTs, 'canproco')).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.join(path_out_imagesTs, 'basel')).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.join(path_out_labelsTs, 'basel')).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.join(path_out_imagesTs, 'sct-testing')).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.join(path_out_labelsTs, 'sct-testing')).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.join(path_out_imagesTs, 'bavaria')).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.join(path_out_labelsTs, 'bavaria')).mkdir(parents=True, exist_ok=True)

    #initialise the conversion dict
    conversion_dict = {}

    #--------------- DISPACTH OF LABELLED IMAGES IN TRAIN OR TEST SET ------------------- 
    
    #Initialise the number of scans in train and in test folder
    scan_cnt_train, scan_cnt_test = 0, 0
    
    #we load the json file
    data_split = json.load(open(path_data_json, 'r'))
    train_data = data_split['training']
    test_data = data_split['testing']
    
    #we iterate over all  training images
    for image_file in tqdm.tqdm(train_data):
        #we update the count of images of the training set
        scan_cnt_train+= 1

        image_file_nnunet = os.path.join(path_out_imagesTr,f'{args.taskname}_{scan_cnt_train:03d}_0000.nii.gz')
        label_file_nnunet = os.path.join(path_out_labelsTr,f'{args.taskname}_{scan_cnt_train:03d}.nii.gz')
        
        train_images.append(str(image_file_nnunet))
        train_labels.append(str(label_file_nnunet))

        # copy the image to new structure
        shutil.copyfile(image_file, image_file_nnunet)

        # Here we build the multi-label label file
        lesion_seg_file = train_data[image_file]['lesion']
        sc_seg_file = train_data[image_file]['sc']
        create_multi_label_mask(lesion_seg_file, sc_seg_file, label_file_nnunet)

        #we update the conversion dict (for label we only point to the lesion mask)
        conversion_dict[str(os.path.abspath(image_file))] = image_file_nnunet
        conversion_dict[str(os.path.abspath(lesion_seg_file))] = label_file_nnunet

    #we iterate over all  testing images
    for image_file in tqdm.tqdm(test_data):
        #we update the count of images of the test set
        scan_cnt_test+= 1

        # we create test folders for each dataset
        if 'canproco' in image_file:
            image_file_nnunet = os.path.join(path_out_imagesTs,'canproco',f'{args.taskname}_{scan_cnt_test:03d}_0000.nii.gz')
            label_file_nnunet = os.path.join(path_out_labelsTs,'canproco',f'{args.taskname}_{scan_cnt_test:03d}.nii.gz')
        elif 'basel' in image_file:
            image_file_nnunet = os.path.join(path_out_imagesTs,'basel',f'{args.taskname}_{scan_cnt_test:03d}_0000.nii.gz')
            label_file_nnunet = os.path.join(path_out_labelsTs,'basel',f'{args.taskname}_{scan_cnt_test:03d}.nii.gz')
        elif 'sct-testing-large' in image_file:
            image_file_nnunet = os.path.join(path_out_imagesTs,'sct-testing',f'{args.taskname}_{scan_cnt_test:03d}_0000.nii.gz')
            label_file_nnunet = os.path.join(path_out_labelsTs,'sct-testing',f'{args.taskname}_{scan_cnt_test:03d}.nii.gz')
        elif 'bavaria' in image_file:
            image_file_nnunet = os.path.join(path_out_imagesTs,'bavaria',f'{args.taskname}_{scan_cnt_test:03d}_0000.nii.gz')
            label_file_nnunet = os.path.join(path_out_labelsTs,'bavaria',f'{args.taskname}_{scan_cnt_test:03d}.nii.gz')

        # copy the image to new structure
        shutil.copyfile(image_file, image_file_nnunet)

        # Here we build the multi-label label file
        lesion_seg_file = train_data[image_file]['lesion']
        sc_seg_file = train_data[image_file]['sc']
        create_multi_label_mask(lesion_seg_file, sc_seg_file, label_file_nnunet)
        
        test_images.append(str(image_file_nnunet))
        test_labels.append(str(label_file_nnunet))

    #Display of number of training and number of testing images
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
        "Spinal cord" : [1, 2] ,
        "Lesion" : 2,
    }
   
    json_dict['regions_class_order'] = [1,2]

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