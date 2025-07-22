"""
This script is used to perform inference with the three existing methods in SCT:
    - sct_deepseg_lesion
    - sct_deepseg lesion_ms_axial_t2
    - sct_deepseg lesion_ms_mp2rage
It takes as input the path to the nnUnet format dataset and the path to the output folder.
This script does not perform computation of the metrics.

Input:
    --input-folder: path to the input folder containing the nnUnet format dataset
    --output-folder: path to the output folder where the results will be saved
    --conv-dict: path to the conversion dictionary file
    --msd-dataset: path to the msd dataset
    --training: whether to perform inference on the training or the testing dataset
    --min-idx: minimum index to start processing files
    --max-idx: maximum index to stop processing files

Output:
    None

Example:
    python inference_sct_methods.py --input-folder /path/to/nnunet/dataset --output-folder /path/to/output/folder

Author: Pierre-Louis Benveniste
"""
import os
import argparse
from pathlib import Path
import json
from tqdm import tqdm
import shutil


def parse_arguments():
    parser = argparse.ArgumentParser(description='Perform inference with the three existing methods in SCT')
    parser.add_argument('--input-folder', type=str, required=True, help='Path to the input folder containing the nnUnet format dataset')
    parser.add_argument('--output-folder', type=str, required=True, help='Path to the output folder where the results will be saved')
    parser.add_argument('--conv-dict', type=str, required=True, help='Path to the conversion dictionary file')
    parser.add_argument('--msd-dataset', type=str, required=True, help='Path to the msd dataset')
    parser.add_argument('--training', action='store_true', help='Whether to perform inference on the training dataset')
    parser.add_argument('--min-idx', type=int, required=True, help='Minimum index to start processing files')
    parser.add_argument('--max-idx', type=int, required=True, help='Maximum index to stop processing files')
    args = parser.parse_args()
    return args


def main():
    # Parse arguments
    args = parse_arguments()
    input_folder = Path(args.input_folder)
    output_folder = Path(args.output_folder)
    conv_dict_path = Path(args.conv_dict)
    msd_dataset_path = Path(args.msd_dataset)

    # Create output directory if it doesn't exist
    output_folder.mkdir(parents=True, exist_ok=True)
    # Create subdirectories for each method
    output_folder_lesion = os.path.join(output_folder, 'sct_deepseg_lesion')
    output_folder_lesion_ms_axial_t2 = os.path.join(output_folder, 'sct_deepseg_lesion_ms_axial_t2')
    output_folder_lesion_ms_mp2rage = os.path.join(output_folder, 'sct_deepseg_lesion_ms_mp2rage')
    os.makedirs(output_folder_lesion, exist_ok=True)
    os.makedirs(output_folder_lesion_ms_axial_t2, exist_ok=True)
    os.makedirs(output_folder_lesion_ms_mp2rage, exist_ok=True)

    # Get the files to segment
    files_to_segment = sorted(list(input_folder.rglob("*.nii.gz")))
    # Only keep the files between the two index values
    files_to_segment = [f for f in files_to_segment if int(f.stem.split("_")[-2]) >= args.min_idx and int(f.stem.split("_")[-2]) <= args.max_idx]

    # Load the conversion dictionary
    with open(conv_dict_path, 'r') as f:
        conversion_dict = json.load(f)
    # We keep only training or testing files based on the argument
    conversion_dict = {k: v for k, v in conversion_dict.items() if (args.training and 'imagesTr' in v) or (not args.training and 'imagesTs' in v)}

    # Load the MSD dataset
    with open(msd_dataset_path, 'r') as f:
        msd_data = json.load(f)
    msd_data = msd_data['train'] + msd_data['validation'] if args.training else msd_data['test']

    # Iterate over each file and perform inference
    for file in tqdm(files_to_segment):
        # For each file we find the file orientation and contrast
        for key, value in conversion_dict.items():
            if file.name in value:
                corresponding_file = key
                break
        for data in msd_data:
            if data['image'] == corresponding_file:
                contrast = data['contrast']
                acquisition = data['acquisition']
                break
        print(f"Processing file: {file.name}, Contrast: {contrast}, Acquisition: {acquisition}")
        
        # Build a temp folder for the current file
        temp_folder = os.path.join(output_folder, file.name.replace('.nii.gz', ''))
        os.makedirs(temp_folder, exist_ok=True)

        # Copy the file to the temp folder
        shutil.copy(file, temp_folder)
        inference_file = os.path.join(temp_folder, file.name)
        
        # Perform inference for sct_deepseg_lesion
        ## The command depends on the contrast and acquisition
        if contrast in ['T2star', 'MEGRE']:
            assert os.system(f"sct_deepseg_lesion -i {inference_file} -c t2s -ofolder {temp_folder}")==0
        elif acquisition == 'ax':
            assert os.system(f"sct_deepseg_lesion -i {inference_file} -c t2_ax -ofolder {temp_folder}")==0
        else:
            assert os.system(f"sct_deepseg_lesion -i {inference_file} -c t2 -ofolder {temp_folder}")==0

        ## Copy the output segmentation to the output folder
        output_lesion_seg_path = os.path.join(output_folder_lesion, file.name.replace('_0000.nii.gz', '.nii.gz'))
        shutil.copy(os.path.join(temp_folder, file.name.replace(".nii.gz", "_lesionseg.nii.gz")), output_lesion_seg_path)
        ## Remove temporary folder
        shutil.rmtree(temp_folder)
        ## Recreate the temp folder for the next method
        os.makedirs(temp_folder, exist_ok=True)
        shutil.copy(file, temp_folder)

        # Now we segment using sct_deepseg lesion_ms_axial_t2
        ## Run the command
        output_file = os.path.join(temp_folder, file.name.replace('_0000.nii.gz', '.nii.gz'))
        assert os.system(f"sct_deepseg lesion_ms_axial_t2 -i {inference_file} -o {output_file}") == 0
        ## Copy the output segmentation to the output folder
        output_t2_ax_path = os.path.join(output_folder_lesion_ms_axial_t2, file.name.replace('_0000.nii.gz', '.nii.gz'))
        shutil.copy(output_file.replace(".nii.gz", "_lesion_seg.nii.gz"), output_t2_ax_path)
        ## Clean up the temporary folder
        shutil.rmtree(temp_folder)
        ## Recreate the temp folder for the next method
        os.makedirs(temp_folder, exist_ok=True)
        shutil.copy(file, temp_folder)

        # Now we segment using sct_deepseg_lesion_ms_mp2rage
        ## Run the command
        output_file = os.path.join(temp_folder, file.name.replace('_0000.nii.gz', '.nii.gz'))
        assert os.system(f"sct_deepseg lesion_ms_mp2rage -i {inference_file} -o {output_file}") == 0
        ## Copy the output segmentation to the output folder
        output_mp2rage_path = os.path.join(output_folder_lesion_ms_mp2rage, file.name.replace('_0000.nii.gz', '.nii.gz'))
        shutil.copy(output_file, output_mp2rage_path)
        ## Clean up the temporary folder
        shutil.rmtree(temp_folder)

    print("Inference completed for all files.")

    return None


if __name__ == "__main__":
    main()