"""
This script agregrates data from multiple datasets into a json file. 
The datasets are umass* (4 datasets), ms-nmo-beijing and ms-mayo-critical-lesions.
For inspiration, I used: https://github.com/ivadomed/ms-lesion-agnostic/blob/r20250626/dataset_analysis/msd_data_analysis.py

Input:
    -data: path to the folder containing the datasets
    -output: path to the output json file
    -exclude-mayo: path to the file containing the list of subjects to exclude from the mayo dataset
Output:
    None

Example:
    python agregate_unannotated_data.py -data /path/to/data -output /path/to/output.json -exclude-mayo /path/to/exclude_mayo.yml

Author: Pierre-Louis Benveniste
"""
import os
import json
import argparse
from pathlib import Path
import yaml
from utils.image import Image
import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate unannotated data from multiple datasets into a json file.")
    parser.add_argument("-data", type=str, required=True, help="Path to the folder containing the datasets.")
    parser.add_argument("-output", type=str, required=True, help="Path to the output json file.")
    parser.add_argument("-exclude-mayo", type=str, required=True, help="Path to the file containing the list of subjects to exclude from the mayo dataset.")
    return parser.parse_args()


def get_acquisition_resolution_and_dimension(image_path, site):
    """
    This function takes an image file as input and returns its acquisition, resolution and dimension.

    Input:
        image_path : str : Path to the image file

    Returns:
        acquisition : str : Acquisition of the image
        orientation : str : Orientation of the image
        resolution : list : Resolution of the image
        dimension : list : Dimension of the image
        field_strength : str : Field strength of the image
        manufacturer : str : Manufacturer of the image
    """
    img = Image(str(image_path))
    img.change_orientation('RPI')
    # Get the resolution
    resolution = list(img.dim[4:7])
    # Get image dimension
    dimension = list(img.dim[0:3])

    # Get image name
    image_name = image_path.split('/')[-1]
    if 'ax' in image_name:
        orientation = 'ax'
    elif 'sag' in image_name:
        orientation = 'sag'
    if '3D' in image_name:
        acquisition = '3D'
    if "mayo" in site:
        acquisition = '2D'
        orientation = 'ax'
        field_strength = 'Missing'
        manufacturer = 'Missing'
    
    # Check if there is a json file
    json_path = image_path.replace('.nii.gz', '.json')
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            json_data = json.load(f)
            if 'SAG' in json_data.get('SeriesDescription') or 'sag' in json_data.get('SeriesDescription') or 'Sag' in json_data.get('SeriesDescription'):
                orientation = 'sag'
            elif 'AX' in json_data.get('SeriesDescription') or 'ax' in json_data.get('SeriesDescription') or 'Ax' in json_data.get('SeriesDescription'):
                orientation = 'ax'
            acquisition = json_data.get('MRAcquisitionType')
            field_strength = json_data.get('MagneticFieldStrength')
            manufacturer = json_data.get('Manufacturer')
    
    return acquisition, orientation, resolution, dimension, field_strength, manufacturer


def main():
    # Parse arguments
    args = parse_args()
    data_path = Path(args.data)
    output_path = Path(args.output)
    mayo_exclude = Path(args.exclude_mayo)

    # If output directory does not exist, create it
    os.makedirs(output_path, exist_ok=True)

    # Load the exclude file from the mayo dataset
    with open(mayo_exclude, 'r') as file:
            mayo_exclude = yaml.load(file, Loader=yaml.FullLoader)
    mayo_exclude = mayo_exclude['slice_motion'] + mayo_exclude['intensity_spikes'] + mayo_exclude['contrast_issues'] 

    # List dataset paths
    mayo_path = os.path.join(data_path, "ms-mayo-critical-lesions") # It contains T2w images
    beijing_path = os.path.join(data_path, "ms-nmo-beijing") # It contains T1w images
    path_umass_1 = os.path.join(data_path, 'umass-ms-ge-hdxt1.5')
    path_umass_2 = os.path.join(data_path, 'umass-ms-ge-pioneer3')
    path_umass_3 = os.path.join(data_path, 'umass-ms-siemens-espree1.5')
    path_umass_4 = os.path.join(data_path, 'umass-ms-ge-excite1.5')

    # Aggregate data
    ## MS-MAYO
    imgs_mayo = list(Path(mayo_path).rglob('*_T2w.nii.gz'))
    imgs_mayo = [i for i in imgs_mayo if 'derivatives' not in str(i)]
    imgs_mayo = [i for i in imgs_mayo if str(i).split('/')[-1] not in mayo_exclude]
    imgs_mayo = [str(i) for i in imgs_mayo]
    print(f"Number of images in mayo dataset: {len(imgs_mayo)}")

    ## MS-NMO-BEIJING
    imgs_beijing = list(Path(beijing_path).rglob('*acq-sag_*T1w.nii.gz'))  # We add acq-sag_T1w (we leave a space in case multiple runs)
    imgs_beijing += list(Path(beijing_path).rglob('*axTseRst_*T2w.nii.gz'))  # We add axTseRst_T2w (we leave a space in case multiple runs)
    imgs_beijing += list(Path(beijing_path).rglob('*sagTseRst_*T2w.nii.gz')) # We add sagTseRst_T2w (we leave a space in case multiple runs)
    imgs_beijing = [i for i in imgs_beijing if 'ocalizer' not in str(i)]
    imgs_beijing = [i for i in imgs_beijing if 'sub-MS' in str(i)]
    imgs_beijing = [str(i) for i in imgs_beijing]
    print(f"Number of images in beijing dataset: {len(imgs_beijing)}")

    ## UMASS 1
    imgs_umass_1 = list(Path(path_umass_1).rglob('*_T1w.nii.gz')) # This is only for images with T1w (not designed for acq-...: there is not acq-... in this case)
    imgs_umass_1 = [i for i in imgs_umass_1 if '_acq-' not in str(i)]
    imgs_umass_1 = [i for i in imgs_umass_1 if 'ce-gad' not in str(i)]
    imgs_umass_1 += list(Path(path_umass_1).rglob('*acq-FMPIR_T2w.nii.gz')) # We add acq-FMPIR_T2w (we leave a space in case multiple runs)
    imgs_umass_1 += list(Path(path_umass_1).rglob('*acq-ax_T1w.nii.gz')) # We add acq-ax_T1w (we leave a space in case multiple runs)
    imgs_umass_1 += list(Path(path_umass_1).rglob('*acq-ax_T2w.nii.gz')) # We add acq-ax_T2w (we leave a space in case multiple runs)
    imgs_umass_1 = [i for i in imgs_umass_1 if 'derivatives' not in str(i)]
    imgs_umass_1 = [str(i) for i in imgs_umass_1]
    print(f"Number of images in umass_1 dataset: {len(imgs_umass_1)}")

    ## UMASS 2
    imgs_umass_2 = list(Path(path_umass_2).rglob('*_T1w.nii.gz'))
    imgs_umass_2 = [i for i in imgs_umass_2 if '_ce-gad' not in str(i)]
    imgs_umass_2 = [i for i in imgs_umass_2 if 'acq-3D' not in str(i)]
    imgs_umass_2 += list(Path(path_umass_2).rglob('*acq-3D_T1w.nii.gz')) # We add acq-3D_T1w (we leave a space in case multiple runs)
    imgs_umass_2 += list(Path(path_umass_2).rglob('*acq-STIR_T2w.nii.gz')) # We add acq-STIR_T2w (we leave a space in case multiple runs)
    imgs_umass_2 += list(Path(path_umass_2).rglob('*acq-axial_T2w.nii.gz')) # We add acq-axial_T2w (we leave a space in case multiple runs)
    imgs_umass_2 = [i for i in imgs_umass_2 if 'derivatives' not in str(i)]
    imgs_umass_2 = [i for i in imgs_umass_2 if 'SHA256' not in str(i)]
    imgs_umass_2 = [str(i) for i in imgs_umass_2]
    print(f"Number of images in umass_2 dataset: {len(imgs_umass_2)}")
    
    
    ## UMASS 3
    imgs_umass_3 = list(Path(path_umass_3).rglob('*T1w.nii.gz'))
    imgs_umass_3 = [i for i in imgs_umass_3 if '_ce-gad' not in str(i)]
    imgs_umass_3 += list(Path(path_umass_3).rglob('*T2w.nii.gz')) # We add acq-3D_T1w (we leave a space in case multiple runs)
    imgs_umass_3 = [i for i in imgs_umass_3 if 'acq-STIR'not in str(i) and 'acq-ax' not in str(i)] # We keep only acq-STIR and acq-ax (we leave a space in case multiple runs)
    imgs_umass_3 += list(Path(path_umass_3).rglob('*acq-STIR_T2w.nii.gz')) # We add acq-STIR_T2w (we leave a space in case multiple runs)
    imgs_umass_3 += list(Path(path_umass_3).rglob('*acq-ax_T2w.nii.gz')) # We add acq-ax_T2w (we leave a space in case multiple runs)
    imgs_umass_3 = [i for i in imgs_umass_3 if 'derivatives' not in str(i)]
    imgs_umass_3 = [i for i in imgs_umass_3 if 'SHA256' not in str(i)]
    imgs_umass_3 = [str(i) for i in imgs_umass_3]
    print(f"Number of images in umass_3 dataset: {len(imgs_umass_3)}")

    ## UMASS 4
    imgs_umass_4 = list(Path(path_umass_4).rglob('*T1w.nii.gz'))
    imgs_umass_4 = [i for i in imgs_umass_4 if '_ce-gad' not in str(i)]
    imgs_umass_4 += list(Path(path_umass_4).rglob('*T2w.nii.gz')) # We add acq-3D_T1w (we leave a space in case multiple runs)
    imgs_umass_4 = [i for i in imgs_umass_4 if 'acq-STIR'not in str(i) and 'acq-ax' not in str(i)] # We keep only acq-STIR and acq-ax (we leave a space in case multiple runs)
    imgs_umass_4 += list(Path(path_umass_4).rglob('*acq-STIR_T2w.nii.gz')) # We add acq-STIR_T2w (we leave a space in case multiple runs)
    imgs_umass_4 += list(Path(path_umass_4).rglob('*acq-ax_T2w.nii.gz')) # We add acq-ax_T2w (we leave a space in case multiple runs)
    imgs_umass_4 = [str(i) for i in imgs_umass_4]
    imgs_umass_4 = [i for i in imgs_umass_4 if 'SHA256' not in str(i)]
    print(f"Number of images in umass_4 dataset: {len(imgs_umass_4)}")

    # Aggregate all images:
    all_imgs = imgs_mayo + imgs_beijing + imgs_umass_1 + imgs_umass_2 + imgs_umass_3 + imgs_umass_4

    # Now we iterate over all images to create a dictionary with the required information
    data_dict = {}
    for img in tqdm(all_imgs):
        # Get the subject ID:
        subject_id = img.split('/')[-1].split('_')[0]
        # Get site ID:
        site = img.split('/data/')[-1].split('/')[0]
        # Contrast:
        contrast = img.split('_')[-1].replace('.nii.gz', '')
        # Except for some particular cases:
        if 'TseRst' in img:
            contrast = 'TseRst_T2w'
        elif 'FMPIR_T2w' in img:
            contrast = 'FMPIR'
        elif 'STIR_T2w' in img:
            contrast = 'STIR'
        acquisition, orientation, resolution, dimension, field_strength, manufacturer = get_acquisition_resolution_and_dimension(img, site)
        resolution = [np.float64(i) for i in resolution]
        img_info = {
            "path": img,
            "subject_id": subject_id,
            "site": site,
            "contrast": contrast,
            "acquisition": acquisition,
            "orientation": orientation,
            "resolution": resolution,
            "dimension": dimension,
            "field_strength": field_strength,
            "manufacturer": manufacturer
        }
        
        # add to the dictionary
        data_dict[img] = img_info

    # For each field, we print the unique values
    print(f"Number of unique contrasts: {set([data_dict[k]['contrast'] for k in data_dict])}")
    print(f"Number of unique acquisitions: {set([data_dict[k]['acquisition'] for k in data_dict])}")
    print(f"Number of unique orientations: {set([data_dict[k]['orientation'] for k in data_dict])}")
    print(f"Number of unique field strengths: {set([data_dict[k]['field_strength'] for k in data_dict])}")

    # save the dictionary as a json file
    json_file_path = os.path.join(output_path, 'unannotated_data.json')
    with open(json_file_path, 'w') as f:
        json.dump(data_dict, f, indent=4)
    print(f"Data saved to {json_file_path}")

if __name__ == "__main__":
    main()