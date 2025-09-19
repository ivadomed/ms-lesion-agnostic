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


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate unannotated data from multiple datasets into a json file.")
    parser.add_argument("-data", type=str, required=True, help="Path to the folder containing the datasets.")
    parser.add_argument("-output", type=str, required=True, help="Path to the output json file.")
    parser.add_argument("-exclude-mayo", type=str, required=True, help="Path to the file containing the list of subjects to exclude from the mayo dataset.")
    return parser.parse_args()


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
    # ## MS-MAYO
    # imgs_mayo = list(Path(mayo_path).rglob('*_T2w.nii.gz'))
    # imgs_mayo = [i for i in imgs_mayo if 'derivatives' not in str(i)]
    # imgs_mayo = [i for i in imgs_mayo if str(i).split('/')[-1] not in mayo_exclude]
    # imgs_mayo = [str(i) for i in imgs_mayo]
    # print(f"Number of images in mayo dataset: {len(imgs_mayo)}")

    # ## MS-NMO-BEIJING
    # imgs_beijing = list(Path(beijing_path).rglob('*acq-sag_*T1w.nii.gz'))  # We add acq-sag_T1w (we leave a space in case multiple runs)
    # imgs_beijing += list(Path(beijing_path).rglob('*axTseRst_*T2w.nii.gz'))  # We add axTseRst_T2w (we leave a space in case multiple runs)
    # imgs_beijing += list(Path(beijing_path).rglob('*sagTseRst_*T2w.nii.gz')) # We add sagTseRst_T2w (we leave a space in case multiple runs)
    # imgs_beijing = [i for i in imgs_beijing if 'ocalizer' not in str(i)]
    # imgs_beijing = [i for i in imgs_beijing if 'sub-MS' in str(i)]
    # imgs_beijing = [str(i) for i in imgs_beijing]
    # print(f"Number of images in beijing dataset: {len(imgs_beijing)}")

    # ## UMASS 1
    # imgs_umass_1 = list(Path(path_umass_1).rglob('*_T1w.nii.gz')) # This is only for images with T1w (not designed for acq-...: there is not acq-... in this case)
    # imgs_umass_1 = [i for i in imgs_umass_1 if '_acq-' not in str(i)]
    # imgs_umass_1 = [i for i in imgs_umass_1 if 'ce-gad' not in str(i)]
    # imgs_umass_1 += list(Path(path_umass_1).rglob('*acq-FMPIR_T2w.nii.gz')) # We add acq-FMPIR_T2w (we leave a space in case multiple runs)
    # imgs_umass_1 += list(Path(path_umass_1).rglob('*acq-ax_T1w.nii.gz')) # We add acq-ax_T1w (we leave a space in case multiple runs)
    # imgs_umass_1 += list(Path(path_umass_1).rglob('*acq-ax_T2w.nii.gz')) # We add acq-ax_T2w (we leave a space in case multiple runs)
    # imgs_umass_1 += list(Path(path_umass_1).rglob('*ce-gad_T1w.nii.gz')) # We add ce-gad_T1w (we leave a space in case multiple runs)
    # imgs_umass_1 = [i for i in imgs_umass_1 if 'acq-ax_ce-gad' not in str(i)]
    # imgs_umass_1 += list(Path(path_umass_1).rglob('*acq-ax_ce-gad_T1w.nii.gz')) # We add acq-ax_ce-gad_T1w (we leave a space in case multiple runs)
    # imgs_umass_1 = [i for i in imgs_umass_1 if 'derivatives' not in str(i)]
    # imgs_umass_1 = [str(i) for i in imgs_umass_1]
    # print(f"Number of images in umass_1 dataset: {len(imgs_umass_1)}")

    ## UMASS 2
    imgs_umass_2 = list(Path(path_umass_2).rglob('*.nii.gz'))
    imgs_umass_2 = [i for i in imgs_umass_2 if 'derivatives' not in str(i)]
    imgs_umass_2 = [i for i in imgs_umass_2 if 'SHA256' not in str(i)] # We remove acq-ax (it is already in umass_1)
    imgs_umass_2 = [str(i) for i in imgs_umass_2]
    for i in imgs_umass_2:
        print(i)
    print(f"Number of images in umass_2 dataset: {len(imgs_umass_2)}")
    
    
    # imgs_umass_1 = list(Path(path_umass_1).rglob('*_T2w.nii.gz'))
    # imgs_umass_2 = list(Path(path_umass_2).rglob('*_T2w.nii.gz'))
    # imgs_umass_3 = list(Path(path_umass_3).rglob('*_T2w.nii.gz'))
    # imgs_umass_4 = list(Path(path_umass_4).rglob('*_T2w.nii.gz'))


if __name__ == "__main__":
    main()