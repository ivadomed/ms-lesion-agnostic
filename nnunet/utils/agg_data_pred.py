"""
This file is used to aggregate the predictions of the folds of the model on the training, test and external test sets.

Input:
    -path-to-data: Path to the data folder

Output:
    None

Example:
    python agg_data_pred.py -path-to-data /path/to/data

Author: Pierre-Louis Benveniste
"""
import os
from pathlib import Path
import nibabel as nib
from tqdm import tqdm
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-path-to-data", required=True, type=str, help="Path to the data folder")
    return parser.parse_args()


def main():
    # Parse arguments
    args = parse_args()
    path_to_data = args.path_to_data

    path_fold0 = os.path.join(path_to_data, 'predicted_imagesTr_fold0')
    path_fold1 = os.path.join(path_to_data, 'predicted_imagesTr_fold1')
    path_fold2 = os.path.join(path_to_data, 'predicted_imagesTr_fold2')
    path_fold3 = os.path.join(path_to_data, 'predicted_imagesTr_fold3')
    path_fold4 = os.path.join(path_to_data, 'predicted_imagesTr_fold4')

    output_folder_avg = os.path.join(path_to_data, 'predicted_imagesTr_avg')
    output_folder_avg_bin05 = os.path.join(path_to_data, 'predicted_imagesTr_avg_bin05')

    # Create the output folders
    os.makedirs(output_folder_avg, exist_ok=True)
    os.makedirs(output_folder_avg_bin05, exist_ok=True)

    # List all files in fold0 with .nii.gz extension with rglob
    files_fold0 = list(Path(path_fold0).rglob('*.nii.gz'))

    for image in tqdm(files_fold0):
        # Load the image
        img_0 = nib.load(str(image))
        
        # Load the same image in the other folds
        img_1 = nib.load(os.path.join(path_fold1, image.name))
        img_2 = nib.load(os.path.join(path_fold2, image.name))
        img_3 = nib.load(os.path.join(path_fold3, image.name))
        img_4 = nib.load(os.path.join(path_fold4, image.name))
        
        # Get the data
        data_0 = img_0.get_fdata()
        data_1 = img_1.get_fdata()
        data_2 = img_2.get_fdata()
        data_3 = img_3.get_fdata()
        data_4 = img_4.get_fdata()

        # Avg the 5 images
        data_avg = (data_0 + data_1 + data_2 + data_3 + data_4) / 5

        # Save the averaged image
        img_avg = nib.Nifti1Image(data_avg, img_0.affine, img_0.header)
        nib.save(img_avg, os.path.join(output_folder_avg, image.name))

        # Binarize the averaged image
        data_avg_bin05 = data_avg >= 0.5
        img_avg_bin05 = nib.Nifti1Image(data_avg_bin05, img_0.affine, img_0.header)
        nib.save(img_avg_bin05, os.path.join(output_folder_avg_bin05, image.name))

    # We do the same for the test images
    path_test0 = os.path.join(path_to_data, 'predicted_imagesTs_fold0')
    path_test1 = os.path.join(path_to_data, 'predicted_imagesTs_fold1')
    path_test2 = os.path.join(path_to_data, 'predicted_imagesTs_fold2')
    path_test3 = os.path.join(path_to_data, 'predicted_imagesTs_fold3')
    path_test4 = os.path.join(path_to_data, 'predicted_imagesTs_fold4')

    output_folder_avg_test = os.path.join(path_to_data, 'predicted_imagesTs_avg')  
    output_folder_avg_bin05_test = os.path.join(path_to_data, 'predicted_imagesTs_avg_bin05')

    # Create the output folders
    os.makedirs(output_folder_avg_test, exist_ok=True)
    os.makedirs(output_folder_avg_bin05_test, exist_ok=True)

    # List all files in fold0 with .nii.gz extension with rglob
    files_test0 = list(Path(path_test0).rglob('*.nii.gz'))

    for image in tqdm(files_test0):
        # Load the image
        img_0 = nib.load(str(image))
        
        # Load the same image in the other folds
        img_1 = nib.load(os.path.join(path_test1, image.name))
        img_2 = nib.load(os.path.join(path_test2, image.name))
        img_3 = nib.load(os.path.join(path_test3, image.name))
        img_4 = nib.load(os.path.join(path_test4, image.name))
        
        # Get the data
        data_0 = img_0.get_fdata()
        data_1 = img_1.get_fdata()
        data_2 = img_2.get_fdata()
        data_3 = img_3.get_fdata()
        data_4 = img_4.get_fdata()

        # Avg the 5 images
        data_avg = (data_0 + data_1 + data_2 + data_3 + data_4) / 5

        # Save the averaged image
        img_avg = nib.Nifti1Image(data_avg, img_0.affine, img_0.header)
        nib.save(img_avg, os.path.join(output_folder_avg_test, image.name))

        # Binarize the averaged image
        data_avg_bin05 = data_avg >= 0.5
        img_avg_bin05 = nib.Nifti1Image(data_avg_bin05, img_0.affine, img_0.header)
        nib.save(img_avg_bin05, os.path.join(output_folder_avg_bin05_test, image.name))

    # Same with the external test sets
    path_extTest0 = os.path.join(path_to_data, 'predicted_imagesExternalTs_fold0')
    path_extTest1 = os.path.join(path_to_data, 'predicted_imagesExternalTs_fold1')
    path_extTest2 = os.path.join(path_to_data, 'predicted_imagesExternalTs_fold2')
    path_extTest3 = os.path.join(path_to_data, 'predicted_imagesExternalTs_fold3')
    path_extTest4 = os.path.join(path_to_data, 'predicted_imagesExternalTs_fold4')

    output_folder_avg_extTest = os.path.join(path_to_data, 'predicted_imagesExternalTs_avg')
    output_folder_avg_bin05_extTest = os.path.join(path_to_data, 'predicted_imagesExternalTs_avg_bin05')

    # Create the output folders
    os.makedirs(output_folder_avg_extTest, exist_ok=True)
    os.makedirs(output_folder_avg_bin05_extTest, exist_ok=True)

    # List all files in fold0 with .nii.gz extension with rglob
    files_extTest0 = list(Path(path_extTest0).rglob('*.nii.gz'))

    for image in tqdm(files_extTest0):
        # Load the image
        img_0 = nib.load(str(image))
        
        # Load the same image in the other folds
        img_1 = nib.load(os.path.join(path_extTest1, image.name))
        img_2 = nib.load(os.path.join(path_extTest2, image.name))
        img_3 = nib.load(os.path.join(path_extTest3, image.name))
        img_4 = nib.load(os.path.join(path_extTest4, image.name))
        
        # Get the data
        data_0 = img_0.get_fdata()
        data_1 = img_1.get_fdata()
        data_2 = img_2.get_fdata()
        data_3 = img_3.get_fdata()
        data_4 = img_4.get_fdata()

        # Avg the 5 images
        data_avg = (data_0 + data_1 + data_2 + data_3 + data_4) / 5

        # Save the averaged image
        img_avg = nib.Nifti1Image(data_avg, img_0.affine, img_0.header)
        nib.save(img_avg, os.path.join(output_folder_avg_extTest, image.name))

        # Binarize the averaged image
        data_avg_bin05 = data_avg >= 0.5
        img_avg_bin05 = nib.Nifti1Image(data_avg_bin05, img_0.affine, img_0.header)
        nib.save(img_avg_bin05, os.path.join(output_folder_avg_bin05_extTest, image.name))

    print('Done!')


if __name__ == '__main__':
    main()