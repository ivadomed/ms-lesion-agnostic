"""
This file was created to analyze the msd dataset used for training and testing our dataset. 
It takes as input the msd dataset and analysis the properties of the dataset.

Input:
    --msd-data-path
    --dataset-path
    --output-folder

Output:
    None

Example:
    python dataset_analysis/msd_data_analysis.py --msd-data-path /path/to/msd/data --dataset-path /path/to/folder/of/datasets --output-folder /path/to/output/folder

Author: Pierre-Louis Benveniste
"""

import argparse
import os
import json
import nibabel as nib
import numpy as np
from pathlib import Path
from image import Image
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--msd-data-path', type=str, required=True, help='Path to the MSD dataset')
    parser.add_argument('--output-folder', type=str, required=True, help='Path to the output folder')
    parser.add_argument('--dataset-path', type=str, required=True, help='Path to the folder containing the datasets')
    return parser.parse_args()


def main():
    # Parse arguments
    args = parse_args()
    msd_data_path = args.msd_data_path
    output_folder = args.output_folder
    dataset_path = args.dataset_path
    
    # Build the output folder
    os.makedirs(output_folder, exist_ok=True)

    # Load the dataset
    with open(msd_data_path, 'r') as f:
        msd_data = json.load(f)
    
    # Get data
    data = msd_data['train'] + msd_data['validation'] + msd_data['test']

    print("Number of images: ", len(data))
    print("Number of images for training: ", (msd_data['numTraining']))
    print("Number of images for validation: ", (msd_data['numValidation']))
    print("Number of images for testing: ", (msd_data['numTest']))

    # Count the number of images per countrast
    contrast_count = {}
    for image in data:
        contrast = image['contrast']
        if contrast not in contrast_count:
            contrast_count[contrast] = 0
        contrast_count[contrast] += 1
    
    print("Number of images per contrast: ", contrast_count)

    # Now we will look at the orientation of the images: 
    print("PSIR are 2D sagital images: count PSIR images:", contrast_count['PSIR'])
    print("STIR are 2D sagital images: count STIR images:", contrast_count['STIR'])
    print("UNIT1 are 3D images: count UNIT1 images:", contrast_count['UNIT1'])
    print("T1w are 3D images: count T1w images:", contrast_count['T1w'])

    # Now for more complex cases: T1w, T2w and T2star
    ## For T1w easy: there is only one: 
    ### After looking at the data, we haev decided to remove it
    ## For T2w:
    # get one example of T2w
    t2w = [image for image in data if image['contrast'] == 'T2w']
    count_t2w_ax = 0
    count_t2w_sag = 0

    # for file in t2w files, if ax not in name print file name
    for file in t2w:
        if 'acq-ax' in file['image'].split('/')[-1]:
            count_t2w_ax += 1
        elif 'acq-sag' in file['image'].split('/')[-1]:
            count_t2w_sag += 1
        elif 'amuVirg' in file['image'].split('/')[-1]:
            count_t2w_sag += 1
        elif 'sub-nyuShepherd' in file['image'].split('/')[-1]:
            count_t2w_ax += 1
        elif 'uclCiccarelli' in file['image'].split('/')[-1]:
            count_t2w_sag += 1
        else:
            print(file['image'].split('/')[-1])
    
    print('For T2w, we have only 2D images: ', count_t2w_ax, ' axial images and ', count_t2w_sag, ' sagital images')
            
    ## For T2star:
    t2star = [image for image in data if image['contrast'] == 'T2star']
    count_t2star_ax = 0
    count_t2star_sag = 0

    # for file in t2star files, if ax not in name print file name
    for file in t2star:
        if 'acq-sag' in file['image'].split('/')[-1]:
            count_t2star_sag += 1
        else:
            count_t2star_ax += 1
    print('For T2star, we have only 2D images: ', count_t2star_ax, ' axial images and ', count_t2star_sag, ' sagital images')

    print("Total number of sagital images: ", count_t2star_sag + count_t2w_sag + contrast_count['PSIR'] + contrast_count['STIR'])
    print("Total number of axial images: ", count_t2star_ax + count_t2w_ax )
    print("Total number of 3D images: ", contrast_count['UNIT1'] + contrast_count['T1w'])

    # Now we count the number of subjects
    subjects = []
    for image in data:
        sub = image['image'].split('/')[-1].split('_')[0]
        dataset = image['image'].split('/data/')[1].split('/')[0]
        subject = dataset + '/' + sub
        subjects.append(subject)
    
    print("Number of subjects: ", len(set(subjects)))    

    # # Now we will look at the average resolution of the images
    # ## Iterate over the images
    # resolutions = []
    # for image in tqdm(data):
    #     image_reoriented = Image(image['image']).change_orientation('RPI')
    #     resolution = image_reoriented.dim[4:7]
    #     resolution = [float(res) for res in resolution]
    #     resolutions.append(resolution)
        
    # print("Average resolution: ", np.mean(resolutions, axis=0))
    # print("Std resolution: ", np.std(resolutions, axis=0))
    # print("Median resolution: ", np.median(resolutions, axis=0))
    # print("Minimum pixel dimension: ", np.min(resolutions))
    # print("Maximum pixel dimension: ", np.max(resolutions))

    print("-------------------------------------")



    # #############################################################
    # Now we can look at the external testing dataset ms-basel-2018 and ms-basel-2020
    path_ms_basel_2018 = os.path.join(dataset_path, 'ms-basel-2018')
    path_ms_basel_2020 = os.path.join(dataset_path, 'ms-basel-2020')

    # We count all the segmentation files in both using rglob
    ms_basel_2018 = list(Path(path_ms_basel_2018).rglob("*lesion-manual.nii.gz"))
    ms_basel_2020 = list(Path(path_ms_basel_2020).rglob("*lesion-manual.nii.gz"))

    list_contrast_ms_basel_2018 = [str(image).split("/")[-1].split('_')[-2] for image in ms_basel_2018]

    print("Number of images in ms-basel-2018: ", len(ms_basel_2018))
    print("Contrast in ms-basel-2018: ", set(list_contrast_ms_basel_2018))
    # Print the number of each contrast
    contrast_count_ms_basel_2018 = {}
    for contrast in set(list_contrast_ms_basel_2018):
        contrast_count_ms_basel_2018[contrast] = list_contrast_ms_basel_2018.count(contrast)
    print("Number of images per contrast in ms-basel-2018: ", contrast_count_ms_basel_2018)

    list_contrast_ms_basel_2020 = [str(image).split("/")[-1].split('_')[-2] for image in ms_basel_2020]

    print("Number of images in ms-basel-2020: ", len(ms_basel_2020))
    print("Contrast in ms-basel-2020: ", set(list_contrast_ms_basel_2020))
    # Print the number of each contrast
    contrast_count_ms_basel_2020 = {}
    for contrast in set(list_contrast_ms_basel_2020):
        contrast_count_ms_basel_2020[contrast] = list_contrast_ms_basel_2020.count(contrast)
    print("Number of images per contrast in ms-basel-2020: ", contrast_count_ms_basel_2020)

    # Now we look at the orientation of the images
    print("Total number of images:", len(ms_basel_2018) + len(ms_basel_2020))
    ## The PD and T2w images are 2D sagittal images
    print("2D sagital images:", contrast_count_ms_basel_2020['PD'] + contrast_count_ms_basel_2018['T2w'])
    ## The T1w images are 3D images
    print("3D images:", contrast_count_ms_basel_2018['T1w'])
    

    # Now we count the number pf subjects
    ## For ms-basel-2018
    subjects_ms_basel_2018 = []
    for image in ms_basel_2018:
        sub = str(image).split("/")[-1].split('_')[0]
        subjects_ms_basel_2018.append(sub)
    print("Number of subjects in ms-basel-2018: ", len(set(subjects_ms_basel_2018)))

    ## For ms-basel-2020
    subjects_ms_basel_2020 = []
    for image in ms_basel_2020:
        sub = str(image).split("/")[-1].split('_')[0]
        subjects_ms_basel_2020.append(sub)
    print("Number of subjects in ms-basel-2020: ", len(set(subjects_ms_basel_2020)))

    # Now we will look at the average resolution of the images
    basel_data = ms_basel_2018 + ms_basel_2020
    ## Iterate over the images
    resolutions_basel = []
    for image in basel_data:
        image_reoriented = Image(str(image)).change_orientation('RPI')
        resolution = image_reoriented.dim[4:7]
        resolution = [float(res) for res in resolution]
        resolutions_basel.append(resolution)
        
    print("Average resolution: ", np.mean(resolutions_basel, axis=0))
    print("Std resolution: ", np.std(resolutions_basel, axis=0))
    print("Median resolution: ", np.median(resolutions_basel, axis=0))
    print("Minimum pixel dimension: ", np.min(resolutions_basel))
    print("Maximum pixel dimension: ", np.max(resolutions_basel))

    print("-------------------------------------")
    # #############################################################
    # Now we can look at the external testing dataset umass*
    path_umass_1 = os.path.join(dataset_path, 'umass-ms-ge-hdxt1.5')
    path_umass_2 = os.path.join(dataset_path, 'umass-ms-ge-pioneer3')
    path_umass_3 = os.path.join(dataset_path, 'umass-ms-siemens-espree1.5')
    path_umass_4 = os.path.join(dataset_path, 'umass-ms-ge-excite1.5')

    # We count all the segmentation files in both using rglob
    umass_1 = list(Path(path_umass_1).rglob("*.nii.gz"))
    umass_2 = list(Path(path_umass_2).rglob("*.nii.gz"))
    umass_3 = list(Path(path_umass_3).rglob("*.nii.gz"))
    umass_4 = list(Path(path_umass_4).rglob("*.nii.gz"))

    umass_1 = [image for image in umass_1 if 'SHA256E' not in str(image)]
    umass_2 = [image for image in umass_2 if 'SHA256E' not in str(image)]
    umass_3 = [image for image in umass_3 if 'SHA256E' not in str(image)]
    umass_4 = [image for image in umass_4 if 'SHA256E' not in str(image)]

    # for each we remove the 3 images 'sub-ms1115_ses-01_acq-ax_ce-gad_T1w', 'sub-ms1098_ses-01_acq-ax_ce-gad_T1w' and 'sub-ms1234_ses-03_acq-ax_ce-gad_T1w'
    umass_1 = [image for image in umass_1 if 'sub-ms1115_ses-01_acq-ax_ce-gad_T1w' not in str(image)]
    umass_1 = [image for image in umass_1 if 'sub-ms1098_ses-01_acq-ax_ce-gad_T1w' not in str(image)]
    umass_1 = [image for image in umass_1 if 'sub-ms1234_ses-03_acq-ax_ce-gad_T1w' not in str(image)]
    umass_2 = [image for image in umass_2 if 'sub-ms1115_ses-01_acq-ax_ce-gad_T1w' not in str(image)]
    umass_2 = [image for image in umass_2 if 'sub-ms1098_ses-01_acq-ax_ce-gad_T1w' not in str(image)]
    umass_2 = [image for image in umass_2 if 'sub-ms1234_ses-03_acq-ax_ce-gad_T1w' not in str(image)]
    umass_3 = [image for image in umass_3 if 'sub-ms1115_ses-01_acq-ax_ce-gad_T1w' not in str(image)]
    umass_3 = [image for image in umass_3 if 'sub-ms1098_ses-01_acq-ax_ce-gad_T1w' not in str(image)]
    umass_3 = [image for image in umass_3 if 'sub-ms1234_ses-03_acq-ax_ce-gad_T1w' not in str(image)]
    umass_4 = [image for image in umass_4 if 'sub-ms1115_ses-01_acq-ax_ce-gad_T1w' not in str(image)]
    umass_4 = [image for image in umass_4 if 'sub-ms1098_ses-01_acq-ax_ce-gad_T1w' not in str(image)]
    umass_4 = [image for image in umass_4 if 'sub-ms1234_ses-03_acq-ax_ce-gad_T1w' not in str(image)]

    # Initialize the seed
    seed = np.random.RandomState(42)

    # We randomly keep 5 images from each dataset
    umass_1 = seed.choice(umass_1, 5, replace=False)
    umass_1 = list(umass_1)
    umass_2 = seed.choice(umass_2, 5, replace=False)
    umass_2 = list(umass_2)
    umass_3 = seed.choice(umass_3, 5, replace=False)
    umass_3 = list(umass_3)
    umass_4 = seed.choice(umass_4, 5, replace=False)
    umass_4 = list(umass_4)

    # Concatenate all the lists
    umass = umass_1 + umass_2 + umass_3 + umass_4

    list_contrast_umass = [str(image).split("/")[-1].split('_')[-1].split('.')[0] for image in umass]

    print("Number of images in umass: ", len(umass))
    print("Contrast in umass: ", set(list_contrast_umass))
    # Print the number of each contrast
    contrast_count_umass = {}
    for contrast in set(list_contrast_umass):
        contrast_count_umass[contrast] = list_contrast_umass.count(contrast)
    print("Number of images per contrast in umass: ", contrast_count_umass)

    # Now we count the number of subjects
    subjects_umass = []
    for image in umass:
        sub = str(image).split("/")[-1].split('_')[0]
        dataset = str(image).split('/data/')[1].split('/')[0]
        subject = dataset + '/' + sub
        subjects_umass.append(subject)

    print("Number of subjects in umass: ", len(set(subjects_umass)))

    # Now we look at orientation of the images
    count_umass_ax = 0
    count_umass_sag = 0
    count_umass_3d = 0
    for image in umass:
        if 'acq-ax' in str(image):
            count_umass_ax += 1
        else:
            # we can open the json sidecar
            sidecar = str(image).replace('.nii.gz', '.json')
            with open(sidecar, 'r') as f:
                metadata = json.load(f)
            if metadata['MRAcquisitionType'] == '3D':
                count_umass_3d += 1
            elif 'sag' in metadata['SeriesDescription'] or 'Sag' in metadata['SeriesDescription'] or 'SAG' in metadata['SeriesDescription']:
                count_umass_sag += 1
            else:
                print("Unknown orientation: ", image)
    print('For umass, we have ', count_umass_ax, ' axial images, ', count_umass_sag, ' sagital images and ', count_umass_3d, ' 3D images')


    # Now we will look at the average resolution of the images
    ## Iterate over the images
    resolutions_umass = []
    for image in umass:
        image_reoriented = Image(str(image)).change_orientation('RPI')
        resolution = image_reoriented.dim[4:7]
        resolution = [float(res) for res in resolution]
        resolutions_umass.append(resolution)
        
    print("Average resolution: ", np.mean(resolutions_umass, axis=0))
    print("Std resolution: ", np.std(resolutions_umass, axis=0))
    print("Median resolution: ", np.median(resolutions_umass, axis=0))
    print("Minimum pixel dimension: ", np.min(resolutions_umass))
    print("Maximum pixel dimension: ", np.max(resolutions_umass))

    print("-------------------------------------")

    #############################################################
    # Now we can look at the external testing dataset ms-nmo-beijing
    path_beijing = os.path.join(dataset_path, 'ms-nmo-beijing')
    
    beijing = list(Path(path_beijing).rglob("*/anat/*.nii.gz"))

    # Keep only T1w images
    beijing = [image for image in beijing if 'T1w' in str(image)]
    # Remove Localizer images
    beijing = [image for image in beijing if 'Localizer' not in str(image)]
    beijing = [image for image in beijing if 'localizer' not in str(image)]
    # Keep only the MS patients
    beijing = [image for image in beijing if 'sub-MS' in str(image)]

    # We only keep 20 images from the dataset
    seed = np.random.RandomState(42)
    beijing = seed.choice(beijing, 20, replace=False)

    list_contrast_beijing = [str(image).split("/")[-1].split('_')[-1].split('.')[0] for image in beijing]

    print("Number of images in beijing: ", len(beijing))
    print("Contrast in beijing: ", set(list_contrast_beijing))
    # Print the number of each contrast
    contrast_count_beijing = {}
    for contrast in set(list_contrast_beijing):
        contrast_count_beijing[contrast] = list_contrast_beijing.count(contrast)
    print("Number of images per contrast in beijing: ", contrast_count_beijing)

    # We look at the orientation
    count_beijing_ax = 0
    count_beijing_sag = 0
    count_beijing_3d = 0
    for image in beijing:
        # open json sidecar
        sidecar = str(image).replace('.nii.gz', '.json')
        with open(sidecar, 'r') as f:
            metadata = json.load(f)
        if metadata['MRAcquisitionType'] == '3D':
            count_beijing_3d += 1
        elif 'tra' in metadata['SeriesDescription']:
            count_beijing_ax += 1
        elif 'sag' in metadata['SeriesDescription'] or 'SAG' in metadata['SeriesDescription']:
            count_beijing_sag += 1
        else:
            print("Unknown orientation: ", image)
    print('For beijing, we have ', count_beijing_ax, ' axial images, ', count_beijing_sag, ' sagital images and ', count_beijing_3d, ' 3D images')

    # Now we count the number of subjects
    subjects_beijing = []
    for image in beijing:
        sub = str(image).split("/")[-1].split('_')[0]
        subjects_beijing.append(sub)
    print("Number of subjects in beijing: ", len(set(subjects_beijing)))

    # Now we will look at the average resolution of the images
    ## Iterate over the images
    resolutions_beijing = []
    for image in beijing:
        image_reoriented = Image(str(image)).change_orientation('RPI')
        resolution = image_reoriented.dim[4:7]
        resolution = [float(res) for res in resolution]
        resolutions_beijing.append(resolution)

        
    print("Average resolution: ", np.mean(resolutions_beijing, axis=0))
    print("Std resolution: ", np.std(resolutions_beijing, axis=0))
    print("Median resolution: ", np.median(resolutions_beijing, axis=0))
    print("Minimum pixel dimension: ", np.min(resolutions_beijing))
    print("Maximum pixel dimension: ", np.max(resolutions_beijing))

    print("-------------------------------------")
    print("-------------------------------------")

    print("Total number of images: ", len(data) + len(ms_basel_2018) + len(ms_basel_2020) + len(umass) + len(beijing))
    print("Total number of subjects: ", len(set(subjects)) + len(set(subjects_ms_basel_2018)) + len(set(subjects_ms_basel_2020)) + len(set(subjects_umass)) + len(set(subjects_beijing)))

    print("Total number of sagital images: ", count_t2star_sag + count_t2w_sag + contrast_count['PSIR'] + contrast_count['STIR'] + contrast_count_ms_basel_2020['PD'] + contrast_count_ms_basel_2018['T2w'] +  count_umass_sag + count_beijing_sag)
    print("Total number of axial images: ", count_t2star_ax + count_t2w_ax + count_umass_ax + count_beijing_ax)
    print("Total number of 3D images: ", contrast_count['UNIT1'] + contrast_count['T1w'] + contrast_count_ms_basel_2018['T1w'] + count_umass_3d + count_beijing_3d)

    # Field strength
    field_strength = []
    for image in data:
        sidecar = image['image'].replace('.nii.gz', '.json')
        # if the sidecar does not exist, we skip the image
        if not os.path.exists(sidecar):
            continue
        with open(sidecar, 'r') as f:
            try:
                metadata = json.load(f)  # Remplacez 'response' par votre source de données
            except json.JSONDecodeError as e:
                continue
        # if field "MagneticFieldStrength" does not exist, we skip the image
        if "MagneticFieldStrength" not in metadata:
            continue
        field_strength.append(metadata["MagneticFieldStrength"])
        # if field strength is 1.5 print the image
        if metadata["MagneticFieldStrength"] == 1.5:
            print(image['image'])
    print("Field strength for MSD dataset: ", set(field_strength))

    return None


if __name__ == '__main__':
    main()