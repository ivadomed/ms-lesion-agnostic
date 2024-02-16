"""
This script is used to correct the dataset by comparying the mask and the image to see if directions, orientationa, size, etc. are correct.

Author: Pierre-Louis Benveniste
"""

import os
import shutil
import pathlib
import nibabel as nib
import tqdm

from image import Image, get_dimension, change_orientation

# #-------------
# # CANPROCO DATASET
# #-------------
# # we first get the names of the files where there are differences between the mask and the image

# # get all lesion mask (lesion-manual.nii.gz), disc labels (_labels-disc.nii.gz) and SC seg (_seg-manual.nii.gz)
# canproco_path = pathlib.Path('/home/GRAMES.POLYMTL.CA/p119007/ms_lesion_agnostic/data/canproco')
# mask_files = list(canproco_path.rglob('*lesion-manual.nii.gz')) + list(canproco_path.rglob('*labels-disc.nii.gz')) + list(canproco_path.rglob('*seg-manual.nii.gz'))

# count = 0
# for file in mask_files:
#     # corresponding image 
#     relative_path = file.relative_to(canproco_path).parent
#     image_path = canproco_path / str(relative_path).replace('derivatives/labels/','') / file.name.replace('_lesion-manual.nii.gz', '.nii.gz').replace('_labels-disc.nii.gz','.nii.gz').replace('_seg-manual.nii.gz','.nii.gz')
    
#     # check if the image and the label have the same orientation
#     image = nib.load(str(image_path))
#     mask = nib.load(str(file))

#     change = False

#     # check if the image and the label have the same dimensions
#     if image.shape != mask.shape:
#         print(f'Image {image_path.name} and {file.name} have different dimensions')
#         change = True
    
#     # check if the image and the label have the same orientation
#     elif (image.affine - mask.affine).any() != 0:
#         print(f'Image {image_path.name} and {file.name} have different orientation')
#         change = True

#     # check if the image and the label have the same sform matrix
#     elif (image.header.get_sform() - mask.header.get_sform()).any() != 0:
#         print(f'Image {image_path.name} and {file.name} have different sform matrix')
#         change = True

#     # check if the image and the label have the same qform matrix
#     elif (image.header.get_qform() - mask.header.get_qform()).any() != 0:
#         print(f'Image {image_path.name}  and {file.name} have different qform matrix')
#         change = True

#     if change:
#         count += 1
#         # change the orientation of the image to match the mask
#         image = Image(str(image_path))
#         mask = Image(str(file))
#         mask = change_orientation(mask, image.orientation)
#         mask.save(str(file))
#         # copy the header of the image to the mask
#         os.system(f'sct_image -i {image_path} -copy-header {file} -o {file}')

# print(f'{count} files have been modified')


# #-------------
# # BASEL MP2RAGE DATASET
# #-------------

# # get all lesion mask
# basel_path = pathlib.Path('/home/GRAMES.POLYMTL.CA/p119007/ms_lesion_agnostic/data/basel-mp2rage')
# lesion_files = list(basel_path.rglob('*_lesion-manualNeuroPoly.nii.gz')) + list(basel_path.rglob('*_lesion-manualKatrin.nii.gz')) + list(basel_path.rglob('*_lesion-manualHaris.nii.gz')) + list(basel_path.rglob('*_label-SC_seg.nii.gz'))

# count = 0
# for file in lesion_files:
#     # corresponding image 
#     relative_path = file.relative_to(basel_path).parent
#     image_path = basel_path / str(relative_path).replace('derivatives/labels/','') / file.name.replace('_lesion-manualNeuroPoly.nii.gz', '.nii.gz').replace('_lesion-manualKatrin.nii.gz','.nii.gz').replace('_lesion-manualHaris.nii.gz','.nii.gz').replace('_label-SC_seg.nii.gz','.nii.gz')
    
#     # check if the image and the label have the same orientation
#     image = nib.load(str(image_path))
#     mask = nib.load(str(file))

#     change = False

#     # check if the image and the label have the same dimensions
#     if image.shape != mask.shape:
#         print(f'Image {image_path.name} and {file.name} have different dimensions')
#         change = True
    
#     # check if the image and the label have the same orientation
#     elif (image.affine - mask.affine).any() != 0:
#         print(f'Image {image_path.name} and {file.name} have different orientation')
#         change = True

#     # check if the image and the label have the same sform matrix
#     elif (image.header.get_sform() - mask.header.get_sform()).any() != 0:
#         print(f'Image {image_path.name} and {file.name} have different sform matrix')
#         change = True

#     # check if the image and the label have the same qform matrix
#     elif (image.header.get_qform() - mask.header.get_qform()).any() != 0:
#         print(f'Image {image_path.name}  and {file.name} have different qform matrix')
#         change = True

#     if change:
#         count += 1
#         # change the orientation of the image to match the mask
#         image = Image(str(image_path))
#         mask = Image(str(file))
#         mask = change_orientation(mask, image.orientation)
#         mask.save(str(file))
#         # copy the header of the image to the mask
#         os.system(f'sct_image -i {image_path} -copy-header {file} -o {file}')

# print(f'{count} files have been modified')

# #-------------
# # SCT-TESTING-LARGE DATASET
# #-------------
# # Now the path to the sct-testing-large dataset
# sct_testing_path = pathlib.Path('/home/GRAMES.POLYMTL.CA/p119007/ms_lesion_agnostic/data/sct-testing-large')
# lesion_files = list(sct_testing_path.rglob('*_lesion-manual.nii.gz'))

# count = 0
# for file in lesion_files:
#     # corresponding image 
#     relative_path = file.relative_to(sct_testing_path).parent
#     image_path = sct_testing_path / str(relative_path).replace('derivatives/labels/','') / file.name.replace('_lesion-manual.nii.gz', '.nii.gz')
    
#     # check if the image and the label have the same orientation
#     image = nib.load(str(image_path))
#     mask = nib.load(str(file))

#     change = False

#     # check if the image and the label have the same dimensions
#     if image.shape != mask.shape:
#         print(f'Image {image_path.name} and {file.name} have different dimensions')
#         change = True
    
#     # check if the image and the label have the same orientation
#     elif (image.affine - mask.affine).any() != 0:
#         print(f'Image {image_path.name} and {file.name} have different orientation')
#         change = True

#     # check if the image and the label have the same sform matrix
#     elif (image.header.get_sform() - mask.header.get_sform()).any() != 0:
#         print(f'Image {image_path.name} and {file.name} have different sform matrix')
#         change = True

#     # check if the image and the label have the same qform matrix
#     elif (image.header.get_qform() - mask.header.get_qform()).any() != 0:
#         print(f'Image {image_path.name}  and {file.name} have different qform matrix')
#         change = True

#     if change:
#         count += 1
#         # change the orientation of the image to match the mask
#         image = Image(str(image_path))
#         mask = Image(str(file))
#         mask = change_orientation(mask, image.orientation)
#         mask.save(str(file))
#         # copy the header of the image to the mask
#         os.system(f'sct_image -i {image_path} -copy-header {file} -o {file}')

# print(f'{count} files have been modified')


#-------------
# BAVARIA DATASET
#-------------
# Now the path to the sct-testing-large dataset
bavaria_path = pathlib.Path('/home/GRAMES.POLYMTL.CA/p119007/ms_lesion_agnostic/data/bavaria-quebec-spine-ms')
lesion_files = list(bavaria_path.rglob('*_lesions-manual_T2w.nii.gz'))

count = 0
for file in lesion_files:
    # corresponding image 
    relative_path = file.relative_to(bavaria_path).parent                                                                                              
    image_path = bavaria_path / str(relative_path).replace('derivatives/labels/','') / file.name.replace('_lesions-manual_T2w.nii.gz', '_T2w.nii.gz')
    
    # check if the image and the label have the same orientation
    image = nib.load(str(image_path))
    mask = nib.load(str(file))

    change = False

    # check if the image and the label have the same dimensions
    if image.shape != mask.shape:
        print(f'Image {image_path.name} and {file.name} have different dimensions')
        change = True
    
    # check if the image and the label have the same orientation
    elif (image.affine - mask.affine).any() != 0:
        print(f'Image {image_path.name} and {file.name} have different orientation')
        change = True

    # check if the image and the label have the same sform matrix
    elif (image.header.get_sform() - mask.header.get_sform()).any() != 0:
        print(f'Image {image_path.name} and {file.name} have different sform matrix')
        change = True

    # check if the image and the label have the same qform matrix
    elif (image.header.get_qform() - mask.header.get_qform()).any() != 0:
        print(f'Image {image_path.name}  and {file.name} have different qform matrix')
        change = True

    if change:
        count += 1
        # change the orientation of the image to match the mask
        image = Image(str(image_path))
        mask = Image(str(file))
        mask = change_orientation(mask, image.orientation)
        mask.save(str(file))
        # copy the header of the image to the mask
        os.system(f'sct_image -i {image_path} -copy-header {file} -o {file}')

print(f'{count} files have been modified')