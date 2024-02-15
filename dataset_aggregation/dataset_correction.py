"""
This script is used to correct the dataset by comparying the mask and the image to see if directions, orientationa, size, etc. are correct.

Author: Pierre-Louis Benveniste
"""

import os
import shutil
import pathlib
from image import Image, get_dimension, change_orientation
import tqdm

#-------------
# CANPROCO DATASET
#-------------
# we first get the names of the files where there are differences between the mask and the image

# get all lesion mask
canproco_path = pathlib.Path('/home/GRAMES.POLYMTL.CA/p119007/ms_lesion_agnostic/data/canproco')
lesion_files = list(canproco_path.rglob('*lesion-manual.nii.gz'))

for file in tqdm.tqdm(lesion_files):
    # corresponding image 
    relative_path = file.relative_to(canproco_path).parent
    image = canproco_path / relative_path.replace('derivatives/labels/','') / file.name.replace('_lesion-manual.nii.gz', '.nii.gz')
    print(image)
    if not image.exists():
        print(f'No image for {file}')
        continue