"""
This script is used to aggregate the different datasets into a single folder.
It was built because each dataset has a specific structure and we want to have a single folder with all the data.
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

#output folder
output_folder = pathlib.Path('/home/GRAMES.POLYMTL.CA/p119007/ms_lesion_agnostic/data/all_ms_sc_data')

"""
#------------------------- CANPROCO -------------------------
# We want all PSIR and STIR files (not all are annotated)
# It contains:
#   - the images sub-tor098_ses-M0_PSIR.nii.gz
#   - the lesion seg sub-tor098_ses-M0_lesion-manual.nii.gz (NOT ALWAYS)
#   - the vert labels sub-tor049_ses-M0_PSIR_labels-disc.nii.gz (NOT ALWAYS)
#   - the SC seg sub-tor049_ses-M0_PSIR_seg-manual.nii.gz (NOT ALWAYS)

# Let's first aggregate the CanProCo dataset
canproco_path = pathlib.Path('/home/GRAMES.POLYMTL.CA/p119007/ms_lesion_agnostic/data/canproco')

# copy all PSIR and STIR files in the output folder
for root, dirs, files in os.walk(canproco_path):
    for file in files:
        if file.endswith('.nii.gz') and 'SHA256' not in str(file):
            if 'PSIR' in file or 'STIR' in file:
                # then we copy the file with the same folder structure
                source_file = pathlib.Path(root) / file
                relative_path = source_file.relative_to(canproco_path)
                destination_file = output_folder / relative_path
                destination_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_file, destination_file)
                # and copy the corresponding json file
                source_json_file = str(source_file).replace('.nii.gz', '.json')
                destination_json_file = str(destination_file).replace('.nii.gz', '.json')
                shutil.copy2(source_json_file, destination_json_file)
                #print(f'Copied {source_file.name}')

#------------------------- BASEL-MP2RAGE -------------------------
# we want all the Nifti files (normally all are annotated)
# it contains: 
#   - the images (e.g. sub-C101_UNIT1.nii.gz)
#   - SC masks (e.g. sub-P015_UNIT1_label-SC_seg.nii.gz)
#   - two lesion masks (e.g. sub-P015_UNIT1_lesion-manualNeuroPoly.nii.gz and sub-P015_UNIT1_lesion-manualKatrin.nii.gz)
                
# Now the path to the basel-mp2rage dataset
basel_path = pathlib.Path('/home/GRAMES.POLYMTL.CA/p119007/ms_lesion_agnostic/data/basel-mp2rage')

# copy all MP2RAGE files in the output folder
for root, dirs, files in os.walk(basel_path):
    for file in files:
        if file.endswith('.nii.gz') and 'SHA256' not in str(file):
            # then we copy the file with the same folder structure
            source_file = pathlib.Path(root) / file
            relative_path = source_file.relative_to(basel_path)
            destination_file = output_folder / relative_path
            destination_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_file, destination_file)
            #print(f'Copied {source_file.name}')

            
#------------------------- SCT-TESTING-LARGE -------------------------
# we want all the files which have a lesion segmentation
# These files end with "lesion_manual.nii.gz"
# It contains:
#   - the image
#   - the lesion seg (ending with "lesion-manual.nii.gz")

# Now the path to the sct-testing-large dataset
sct_testing_path = pathlib.Path('/home/GRAMES.POLYMTL.CA/p119007/ms_lesion_agnostic/data/sct-testing-large')

# list all files with "lesion_manual.nii.gz" and copy them and associated image to the output folder
lesion_files = list(sct_testing_path.rglob('*lesion-manual.nii.gz'))
for lesion_file in lesion_files:
    # then we copy the file with the same folder structure
    source_file = lesion_file
    relative_path = source_file.relative_to(sct_testing_path)
    destination_file = output_folder / relative_path
    destination_file.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_file, destination_file)
    # and corresponding json file if it exists 
    source_json_file = str(source_file).replace('.nii.gz', '.json')
    if os.path.isfile(source_json_file):
        destination_json_file = str(destination_file).replace('.nii.gz', '.json')
        shutil.copy2(source_json_file, destination_json_file)
    # and we copy the corresponding image
    source_image_file = str(source_file).replace('_lesion-manual.nii.gz', '.nii.gz').replace('/derivatives/labels', '')
    relative_image_path = pathlib.Path(source_image_file).relative_to(sct_testing_path)
    destination_image_file = output_folder / relative_image_path
    destination_image_file.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_image_file, destination_image_file)
    # and corresponding json file if it exists
    source_json_file = str(source_image_file).replace('.nii.gz', '.json')
    if os.path.isfile(source_json_file):
        destination_json_file = str(destination_image_file).replace('.nii.gz', '.json')
        shutil.copy2(source_json_file, destination_json_file)
    #print(f'Copied {pathlib.Path(source_image_file).name}')


#------------------------- BAVARIA -------------------------
# we want all the .nii.gz files
# It contains:
#   - the images (e.g. sub-m023917_ses-20130506_acq-sag_T2w.nii.gz )
#   - the lesion seg (e.g. sub-m998939_ses-20110516_acq-ax_lesions-manual_T2w.nii.gz) (NOT ALWAYS) : name modified to sub-m998939_ses-20110516_acq-ax_T2w_lesion-manual.nii.gz
#   - the SC seg (e.g. sub-m998939_ses-20110516_acq-ax_seg-manual_T2w.nii.gz ) (NOT ALWAYS) : name modified to sub-m998939_ses-20110516_acq-ax_T2w_seg-manual.nii.gz
# In brief we moved the T2w in the name and removed the "s" from "lesions-manual"

# Now the path to the bavaria-quebec-spine-ms dataset
bavaria_path = pathlib.Path('/home/GRAMES.POLYMTL.CA/p119007/ms_lesion_agnostic/data/bavaria-quebec-spine-ms')

# copy all nifti files in the output folder
for root, dirs, files in os.walk(bavaria_path):
    for file in files:
        if file.endswith('.nii.gz') and 'SHA256' not in str(file):
            # then we copy the file with the same folder structure
            source_file = pathlib.Path(root) / file
            relative_path = source_file.relative_to(bavaria_path)
            destination_file = output_folder / relative_path
            #print(destination_file)
            # if its a mask we modify the name
            if 'manual' in str(file):
                destination_file = destination_file.parent / destination_file.name.replace('lesions-manual_T2w', 'T2w_lesion-manual')
                destination_file = destination_file.parent / destination_file.name.replace('seg-manual_T2w', 'T2w_seg-manual')
            destination_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_file, destination_file)
            # and copy the corresponding json file
            source_json_file = str(source_file).replace('.nii.gz', '.json')
            destination_json_file = str(destination_file).replace('.nii.gz', '.json')
            shutil.copy2(source_json_file, destination_json_file)
            print(f'Copied {source_file.name}')
"""

#------------------------- MSSEG CHALLENGE 2021 -------------------------
# We want all the .nii.gz files
# It contains:
#   - the image (e.g. sub-015_ses-01_FLAIR.nii.gz)
#   - the lesion segs (there are multiple masks) (e.g. sub-015_ses-02_FLAIR_lesion-manual-rater2.nii.gz) (NOT ALWAYS)

# Now the path to the msseg_challenge_2021 dataset
msseg_path = pathlib.Path('/home/GRAMES.POLYMTL.CA/p119007/ms_lesion_agnostic/data/msseg_challenge_2021')

# copy all nifti files in the output folder
print("ok")
for root, dirs, files in os.walk(msseg_path):
    for file in files:
        if file.endswith('.nii.gz') and 'SHA256' not in str(file):
            print(file)
            # then we copy the file with the same folder structure
            source_file = pathlib.Path(root) / file
            relative_path = source_file.relative_to(msseg_path)
            destination_file = output_folder / relative_path
            destination_file.parent.mkdir(parents=True, exist_ok=True)
            # if it's a segmentation file we skip
            if 'derivatives' in str(file):
                continue
            # we segment the spinal cord and output it in a temp folder in the output folder
            ## create the temp folder in the output folder
            output_folder_temp = output_folder / 'temp'
            output_folder_temp.mkdir(parents=True, exist_ok=True)
            ## build output file name
            seg_file = output_folder_temp / source_file.name.replace('.nii.gz', '_seg.nii.gz')
            ## segment the spinal cord 
            # os.system(f'sct_deepseg_sc -i {source_file} -c t2 -o {seg_file}')

            # shutil.copy2(source_file, destination_file)
            # print(f'Copied {source_file.name}')



