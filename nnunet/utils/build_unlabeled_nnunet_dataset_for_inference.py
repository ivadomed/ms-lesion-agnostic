"""
This file created the nnunet format dataset for inference of some of the unlabeled data.

Arguments:
    -pd, --path-data: Path to the data set directory
    -po, --path-out: Path to the output directory where the nnunet format is stored
    --seed: Seed for reproducibility
    
Example:
    python build_unlabeled_nnunet_dataset_for_inference.py -pd /path/to/data -po /path/to/output

Pierre-Louis Benveniste
"""

import os
import json
from tqdm import tqdm
import argparse
from pathlib import Path


def get_parser():
    """
    Get parser for script create_msd_data.py

    Input:
        None

    Returns:
        parser : argparse object
    """

    parser = argparse.ArgumentParser(description='Code for MSD-style JSON datalist for lesion-agnostic nnunet model training.')

    parser.add_argument('-pd', '--path-data', required=True, type=str, help='Path to the folder containing the datasets')
    parser.add_argument('-po', '--path-out', type=str, help='Path to the output directory where the images are stored')
    parser.add_argument('--seed', default=42, type=int, help="Seed for reproducibility")

    return parser


def main():
    """
    This is the main function of the script.

    Input:
        None
    
    Returns:
        None
    """
    # Get the arguments
    parser = get_parser()
    args = parser.parse_args()

    # Get the arguments
    data = args.path_data
    path_out = args.path_out

    # Get all subjects
    path_umass1 = Path(os.path.join(data, "umass-ms-ge-excite1.5"))
    path_umass2 = Path(os.path.join(data, "umass-ms-ge-hdxt1.5"))
    path_umass3 = Path(os.path.join(data, "umass-ms-ge-pioneer3"))
    path_umass4 = Path(os.path.join(data, "umass-ms-siemens-espree1.5"))
    path_beijing = Path(os.path.join(data, "ms-nmo-beijing"))
    path_rennes = Path(os.path.join(data, "ms-rennes-mp2rage"))

    # In umass dataset we get the STIR_T2w.nii.gz images
    derivatives_umass1 = list(path_umass1.rglob('*acq-STIR_T2w.nii.gz'))
    derivatives_umass2 = list(path_umass2.rglob('*acq-STIR_T2w.nii.gz'))
    derivatives_umass3 = list(path_umass3.rglob('*acq-STIR_T2w.nii.gz'))
    derivatives_umass4 = list(path_umass4.rglob('*acq-STIR_T2w.nii.gz'))
    derivatives_stir = derivatives_umass1 + derivatives_umass2 + derivatives_umass3 + derivatives_umass4
    # In beijing dataset we get the T1w.nii.gz images
    derivatives_t1w = list(path_beijing.rglob('*_T1w.nii.gz'))
    # In rennes dataset we ge the UNIT1.nii.gz images
    derivatives_unit1 = list(path_rennes.rglob('*_UNIT1.nii.gz'))
    # In the umass dataset we get the T1w.nii.gz images _acq-ax_T2w.nii.gz
    derivatives_umass1_t2w = list(path_umass1.rglob('*acq-ax_T2w.nii.gz'))
    derivatives_umass2_t2w = list(path_umass2.rglob('*acq-ax_T2w.nii.gz'))
    derivatives_umass3_t2w = list(path_umass3.rglob('*acq-axial_T2w.nii.gz'))
    derivatives_umass4_t2w = list(path_umass4.rglob('*acq-ax_T2w.nii.gz'))
    derivatives_t2w = derivatives_umass1_t2w + derivatives_umass2_t2w + derivatives_umass3_t2w + derivatives_umass4_t2w

    print(f"Number of STIR images: {len(derivatives_stir)}")
    print(f"Number of T1w images: {len(derivatives_t1w)}")
    print(f"Number of UNIT1 images: {len(derivatives_unit1)}")
    print(f"Number of T2w images: {len(derivatives_t2w)}")

    # We will create a dictionary to store the data
    conversion_dict = {}

    # First image will be named msLesionAgnostic_3866.nii.gz (to continue from the data from the predictions of 901)
    scan_cnt = 3865

    all_images = derivatives_stir + derivatives_t1w + derivatives_unit1 + derivatives_t2w   
    
    # Iterate over all images
    for img_path in tqdm(all_images):
        scan_cnt += 1
        
        image_file_nnunet = os.path.join(path_out,f'msLesionAgnostic_{scan_cnt:03d}_0000.nii.gz')

        # Instead of copying we will reorient the image to RPI
        assert os.system(f"sct_image -i {str(img_path)} -setorient RPI -o {image_file_nnunet}") ==0

        # Update conversion dict
        conversion_dict[str(img_path)] = image_file_nnunet
    
    # create dataset_description.json
    json_object = json.dumps(conversion_dict, indent=4)
    # write to dataset description
    conversion_dict_name = f"conversion_dict_unlabeled_data.json"
    with open(os.path.join(path_out, conversion_dict_name), "w") as outfile:
        outfile.write(json_object)


    return None


if __name__ == "__main__":
    main()