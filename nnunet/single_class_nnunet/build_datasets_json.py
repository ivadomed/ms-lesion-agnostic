"""
This script is used to build the 'datasets.json' file that is used by the web app to display the datasets available for training models. 
The datasets are:
- canproco : PSIR and STIR contrast
- basel-mp2rage : MP2RAGE
- sct-testing-large : T1, T2 and T2*
- bavaria-quebec-spine-ms : T2w
- msseg_challenge_2021 : FLAIR but need to first crop the top of the image : BUT the images include very little of the spinal cord (no lesions in the SC I think)
"""

import os
import pathlib
import shutil
import tqdm
import json
import random as rd

#output folder
output_folder = pathlib.Path('/home/GRAMES.POLYMTL.CA/p119007/ms_lesion_agnostic/data/all_ms_sc_data')

#building the image and mask conversion dictionaries
conversion_dict_training = {}
conversion_dict_testing = {}
inference_images = {}

# training and testing ratio
train_ratio = 0.8

#------------------------- CANPROCO -------------------------
# We want all PSIR and STIR files (not all are annotated)
# It contains:
#   - the images sub-tor098_ses-M0_PSIR.nii.gz
#   - the lesion seg sub-tor098_ses-M0_lesion-manual.nii.gz (NOT ALWAYS)
#   - the vert labels sub-tor049_ses-M0_PSIR_labels-disc.nii.gz (NOT ALWAYS)
#   - the SC seg sub-tor049_ses-M0_PSIR_seg-manual.nii.gz (NOT ALWAYS)

# Let's first aggregate the CanProCo dataset
canproco_path = pathlib.Path('/home/GRAMES.POLYMTL.CA/p119007/ms_lesion_agnostic/data/canproco')

count_canproco_train = 0
count_canproco_test = 0
count_canproco_inf = 0

files = list(canproco_path.rglob('*_PSIR.nii.gz')) + list(canproco_path.rglob('*STIR.nii.gz'))
for file in tqdm.tqdm(files):
    if 'SHA256' not in str(file):
        # we check if the file has a mask in the derivatives folder (derivatives/folder and then same relative path)
        relative_path = file.relative_to(canproco_path).parent
        lesion_mask = canproco_path / 'derivatives' / 'labels' / relative_path / file.name.replace('.nii.gz', '_lesion-manual.nii.gz')
        # if the mask exists we add it either to training or testing
        if lesion_mask.exists():
            # we add it to the conversion dictionary of training or testing depending on the ratio
            if rd.random() < train_ratio:
                conversion_dict_training[str(lesion_mask)] = str(file)
                count_canproco_train +=1
            else:
                conversion_dict_testing[str(lesion_mask)] = str(file)
                count_canproco_test += 1
        else:
            # we add it to the inference images
            inference_images[count_canproco_inf] = str(file)
            count_canproco_inf += 1
                    
print(f'CanProCo: {count_canproco_train} images for training')
print(f'CanProCo: {count_canproco_test} images for testing')
print(f'CanProCo: {count_canproco_inf} images for inference')


#------------------------- BASEL-MP2RAGE -------------------------
# we want all the UNIT1 files (normally all are annotated)
# it contains: 
#   - the images (e.g. sub-C101_UNIT1.nii.gz)
#   - SC masks (e.g. sub-P015_UNIT1_label-SC_seg.nii.gz)
#   - two lesion masks (e.g. sub-P015_UNIT1_lesion-manualNeuroPoly.nii.gz and sub-P015_UNIT1_lesion-manualKatrin.nii.gz)
                
# Now the path to the basel-mp2rage dataset
basel_path = pathlib.Path('/home/GRAMES.POLYMTL.CA/p119007/ms_lesion_agnostic/data/basel-mp2rage')

count_basel_train= 0
count_basel_test = 0
count_basel_inf = 0

# copy all MP2RAGE files in the output folder
files = list(basel_path.rglob('*UNIT1.nii.gz'))
for file in tqdm.tqdm(files):
    if 'SHA256' not in str(file):
        # we check if the file has a mask in the derivatives folder (derivatives/folder and then same relative path)
        relative_path = file.relative_to(basel_path).parent
        lesion_mask1 = basel_path / 'derivatives' / 'labels' / relative_path / file.name.replace('.nii.gz', '_lesion-manualNeuroPoly.nii.gz')
        lesion_mask2 = basel_path / 'derivatives' / 'labels' / relative_path / file.name.replace('.nii.gz', '_lesion-manualKatrin.nii.gz')
        lesion_mask3 = basel_path / 'derivatives' / 'labels' / relative_path / file.name.replace('.nii.gz', '_lesion-manualHaris.nii.gz')
        # if the mask exists we add it either to training or testing
        if lesion_mask1.exists():
            # we add it to the conversion dictionary of training or testing depending on the ratio
            if rd.random() < train_ratio:
                conversion_dict_training[str(lesion_mask1)] = str(file)
                count_basel_train += 1
            else:
                conversion_dict_testing[str(lesion_mask1)] = str(file)
                count_basel_test += 1
            
        if lesion_mask2.exists():
            # we add it to the conversion dictionary of training or testing depending on the ratio
            if rd.random() < train_ratio:
                conversion_dict_training[str(lesion_mask2)] = str(file)
                count_basel_train += 1
            else:
                conversion_dict_testing[str(lesion_mask2)] = str(file)
                count_basel_test += 1
        if lesion_mask3.exists():
            # we add it to the conversion dictionary of training or testing depending on the ratio
            if rd.random() < train_ratio:
                conversion_dict_training[str(lesion_mask3)] = str(file)
                count_basel_train += 1
            else:
                conversion_dict_testing[str(lesion_mask3)] = str(file)
                count_basel_test += 1
        else:
            # we add it to the inference images
            inference_images[count_basel_inf] = str(file)
            count_basel_inf += 1

print(f'Basel: {count_basel_train} images for training')
print(f'Basel: {count_basel_test} images for testing')
print(f'Basel: {count_basel_inf} images for inference')

           
#------------------------- SCT-TESTING-LARGE -------------------------
# we want all the files which have a lesion segmentation
# These files end with "lesion_manual.nii.gz"
# It contains:
#   - the image
#   - the lesion seg (ending with "lesion-manual.nii.gz")

# Now the path to the sct-testing-large dataset
sct_testing_path = pathlib.Path('/home/GRAMES.POLYMTL.CA/p119007/ms_lesion_agnostic/data/sct-testing-large')

count_sct_testing_train = 0
count_sct_testing_test = 0

files = list(sct_testing_path.rglob('*_lesion-manual.nii.gz'))
for file in tqdm.tqdm(files):
    if 'SHA256' not in str(file):
        # we build the corresponding image
        image = str(file).replace('_lesion-manual.nii.gz', '.nii.gz').replace('derivatives/labels/', '')
        # we add it to the conversion dictionary of training or testing depending on the ratio
        if rd.random() < train_ratio:
            conversion_dict_training[str(file)] = str(image)
            count_sct_testing_train+=1
        else:
            conversion_dict_testing[str(file)] = str(image)
            count_sct_testing_test+=1

print(f'SCT-Testing: {count_sct_testing_train} images for training')
print(f'SCT-Testing: {count_sct_testing_test} images for testing')
print(f'SCT-Testing: 0 images for inference')


#------------------------- BAVARIA -------------------------
# we want all the .nii.gz files
# It contains:
#   - the images (e.g. sub-m023917_ses-20130506_acq-sag_T2w.nii.gz )
#   - the lesion seg (e.g. sub-m998939_ses-20110516_acq-ax_lesions-manual_T2w.nii.gz) (NOT ALWAYS) : name modified to sub-m998939_ses-20110516_acq-ax_T2w_lesion-manual.nii.gz
#   - the SC seg (e.g. sub-m998939_ses-20110516_acq-ax_seg-manual_T2w.nii.gz ) (NOT ALWAYS) : name modified to sub-m998939_ses-20110516_acq-ax_T2w_seg-manual.nii.gz
# In brief we moved the T2w in the name and removed the "s" from "lesions-manual"

# Now the path to the bavaria-quebec-spine-ms dataset
bavaria_path = pathlib.Path('/home/GRAMES.POLYMTL.CA/p119007/ms_lesion_agnostic/data/bavaria-quebec-spine-ms')

count_bavaria_train = 0
count_bavaria_test = 0
count_bavaria_inf = 0

# copy all nifti files in the output folder
files = list(bavaria_path.rglob('*T2w.nii.gz'))
for file in tqdm.tqdm(files):
    if 'SHA256' not in str(file):
        # we check if the file has a mask in the derivatives folder (derivatives/folder and then same relative path)
        relative_path = file.relative_to(bavaria_path).parent
        lesion_mask = bavaria_path / 'derivatives' / 'labels' / relative_path / file.name.replace('_T2w.nii.gz', '_lesions-manual_T2w.nii.gz')
        # if the mask exists we add it either to training or testing
        if lesion_mask.exists():
            # we add it to the conversion dictionary of training or testing depending on the ratio
            if rd.random() < train_ratio:
                conversion_dict_training[str(lesion_mask)] = str(file)
                count_bavaria_train += 1
            else:
                conversion_dict_testing[str(lesion_mask)] = str(file)
                count_bavaria_test += 1
        else:
            # we add it to the inference images
            inference_images[count_bavaria_inf] = str(file)
            count_bavaria_inf += 1

# print the number of files copied
print(f'Bavaria: {count_bavaria_train} images for training')
print(f'Bavaria: {count_bavaria_test} images for testing')
print(f'Bavaria: {count_bavaria_inf} images for inference')

print("Total image for training: ", count_canproco_train + count_basel_train + count_sct_testing_train + count_bavaria_train)
print("Total image for testing: ", count_canproco_test + count_basel_test  + count_sct_testing_test + count_bavaria_test)
print("Total image for inference: ", count_canproco_inf + count_basel_inf + count_bavaria_inf)

# save the conversion dictionaries in the output folder
with open(output_folder / 'data_singleclass_nnunet.json', 'w') as f:
    json.dump({'training': conversion_dict_training, 'testing': conversion_dict_testing, 'inference': inference_images}, f, indent=4)
print("Conversion dictionaries saved")
