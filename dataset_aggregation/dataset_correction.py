"""
This script is used to correct the dataset by comparying the mask and the image to see if directions, orientationa, size, etc. are correct.

Author: Pierre-Louis Benveniste
"""

import os
import shutil
import pathlib
import nibabel as nib
import tqdm

#-------------
# CANPROCO DATASET
#-------------
# we first get the names of the files where there are differences between the mask and the image

# get all lesion mask
canproco_path = pathlib.Path('/home/GRAMES.POLYMTL.CA/p119007/ms_lesion_agnostic/data/canproco')
lesion_files = list(canproco_path.rglob('*lesion-manual.nii.gz'))

for file in lesion_files:
    # corresponding image 
    relative_path = file.relative_to(canproco_path).parent
    image_path = canproco_path / str(relative_path).replace('derivatives/labels/','') / file.name.replace('_lesion-manual.nii.gz', '.nii.gz')
    
    # check if the image and the label have the same orientation
    image = nib.load(str(image_path))
    mask = nib.load(str(file))

    # print image header
    # print(image.header)
    # print(mask.header)

    # check if the image and the label have the same dimensions
    if image.shape != mask.shape:
        print(f'Image {image_path.name} and mask have different dimensions: {image.shape} and {mask.shape}')
        break
    
    # check if the image and the label have the same orientation
    if (image.affine - mask.affine).any() != 0:
        print(f'Image {image_path.name} and mask have different orientation: {image.affine} and {mask.affine}')
        break 

    # check if the image and the label have the same sform matrix
    if (image.header.get_sform() - mask.header.get_sform()).any() != 0:
        print(f'Image {image_path.name} and mask have different sform matrix: {image.header.get_sform()} and {mask.header.get_sform()}')
        break

    # check if the image and the label have the same qform matrix
    if (image.header.get_qform() - mask.header.get_qform()).any() != 0:
        print(f'Image {image_path.name}  and mask have different qform matrix: {image.header.get_qform()} and {mask.header.get_qform()}')
        break



# #-------------
# # BASEL MP2RAGE DATASET
# #-------------

# # get all lesion mask
# basel_path = pathlib.Path('/home/GRAMES.POLYMTL.CA/p119007/ms_lesion_agnostic/data/basel-mp2rage')

# lesion_files = list(basel_path.rglob('*_lesion-manualNeuroPoly.nii.gz')) + list(basel_path.rglob('*_lesion-manualKatrin.nii.gz')) + list(basel_path.rglob('*_lesion-manualHaris.nii.gz'))

# for file in lesion_files:
#     # corresponding image 
#     relative_path = file.relative_to(basel_path).parent
#     image_path = basel_path / str(relative_path).replace('derivatives/labels/','') / file.name.replace('_lesion-manualNeuroPoly.nii.gz', '.nii.gz').replace('_lesion-manualKatrin.nii.gz','.nii.gz').replace('_lesion-manualHaris.nii.gz','.nii.gz')
    
#     # check if the image and the label have the same orientation
#     image = nib.load(str(image_path))
#     mask = nib.load(str(file))

#     # print image header
#     # print(image.header)
#     # print(mask.header)

#    # check if the image and the label have the same dimensions
#     if image.shape != mask.shape:
#         print(f'Image and mask have different dimensions: {image.shape} and {mask.shape}')
#         break
    
#     # check if the image and the label have the same orientation
#     if not (image.affine - mask.affine).all() == 0:
#         print(f'Image and mask have different orientation: {image.affine} and {mask.affine}')
#         break 

#     # check if the image and the label have the same sform matrix
#     if not (image.header.get_sform() - mask.header.get_sform()).all() == 0:
#         print(f'Image and mask have different sform matrix: {image.header.get_sform()} and {mask.header.get_sform()}')
#         break

#     # check if the image and the label have the same qform matrix
#     if not (image.header.get_qform() - mask.header.get_qform()).all() == 0:
#         print(f'Image and mask have different qform matrix: {image.header.get_qform()} and {mask.header.get_qform()}')
#         break

# #-------------
# # SCT-TESTING-LARGE DATASET
# #-------------
# # Now the path to the sct-testing-large dataset
# sct_testing_path = pathlib.Path('/home/GRAMES.POLYMTL.CA/p119007/ms_lesion_agnostic/data/sct-testing-large')


# lesion_files = list(sct_testing_path.rglob('*_lesion-manual.nii.gz'))

# for file in lesion_files:
#     # corresponding image 
#     relative_path = file.relative_to(sct_testing_path).parent
#     image_path = sct_testing_path / str(relative_path).replace('derivatives/labels/','') / file.name.replace('_lesion-manual.nii.gz', '.nii.gz')
    
#     # check if the image and the label have the same orientation
#     image = nib.load(str(image_path))
#     mask = nib.load(str(file))

#     # print image header
#     # print(image.header)
#     # print(mask.header)

#    # check if the image and the label have the same dimensions
#     if image.shape != mask.shape:
#         print(f'Image and mask have different dimensions: {image.shape} and {mask.shape}')
#         break
    
#     # check if the image and the label have the same orientation
#     if not (image.affine - mask.affine).all() == 0:
#         print(f'Image and mask have different orientation: {image.affine} and {mask.affine}')
#         break 

#     # check if the image and the label have the same sform matrix
#     if not (image.header.get_sform() - mask.header.get_sform()).all() == 0:
#         print(f'Image and mask have different sform matrix: {image.header.get_sform()} and {mask.header.get_sform()}')
#         break

#     # check if the image and the label have the same qform matrix
#     if not (image.header.get_qform() - mask.header.get_qform()).all() == 0:
#         print(f'Image and mask have different qform matrix: {image.header.get_qform()} and {mask.header.get_qform()}')
#         break