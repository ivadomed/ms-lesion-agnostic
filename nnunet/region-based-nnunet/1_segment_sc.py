"""
In this script we segment the spinal cord of all the files which have a lesion segmentation file but no spinal cord segmentation file.
This script also creates a QC report of the resulting segmentation files for visual inspection and manual correction if necessary.
The segmentations are performed using the Spinal Cord Toolbox (SCT) v6.2: sct_deepseg -task sec_sc_contrast_agnostic

Usage:

python 1_segment_sc.py

Author: Pierre-Louis Benveniste
"""

import os
from pathlib import Path
import tqdm
import yaml



# -----------------------
# CANPROCO
# -----------------------
# First we segment the spinal cord of the Canproco dataset using the previously trained region-based model on canproco (because contrast-agnostic works really bad on PSIR/STIR)

# Canproco dataset
canproco_path = Path('/home/GRAMES.POLYMTL.CA/p119007/ms_lesion_agnostic/data/canproco')
# Exclude liste
exclude_list = Path('/home/GRAMES.POLYMTL.CA/p119007/ms_lesion_agnostic/canproco/exclude.yml')
# Model path 
model_path = Path('/home/GRAMES.POLYMTL.CA/p119007/ms_lesion_agnostic/model_ms_seg_sc-lesion_regionBased/nnUNetTrainer__nnUNetPlans__2d')
# Output folder
canproco_output_folder = Path('/home/GRAMES.POLYMTL.CA/p119007/ms_lesion_agnostic/data/sc_seg/canproco_sc_seg')
# QC folder 
canproco_qc_folder = Path('/home/GRAMES.POLYMTL.CA/p119007/ms_lesion_agnostic/data/sc_seg/canproco_sc_seg_qc')

# import the exclude list
with open(exclude_list, 'r') as file:
    exclude_list = yaml.safe_load(file)

files = list(canproco_path.rglob('*_PSIR.nii.gz')) + list(canproco_path.rglob('*STIR.nii.gz'))

for file in files:
    if 'SHA256' not in str(file) and file.name.replace('_PSIR.nii.gz','').replace('_STIR.nii.gz','') not in exclude_list:
        # we check if the file has a mask in the derivatives folder (derivatives/folder and then same relative path)
        relative_path = file.relative_to(canproco_path).parent
        lesion_mask = canproco_path / 'derivatives' / 'labels' / relative_path / file.name.replace('.nii.gz', '_lesion-manual.nii.gz')
        # if the mask exists we add it either to training or testing
        if lesion_mask.exists():
            sc_seg = lesion_mask.parent / lesion_mask.name.replace('lesion-manual.nii.gz', 'seg-manual.nii.gz')
            if not sc_seg.exists():
                # output file path building
                output_dir = canproco_output_folder / relative_path
                output_dir.mkdir(parents=True, exist_ok=True)
                # inverse the image (multiplied by -1)
                inv_file = output_dir / file.name
                os.system(f'sct_maths -i {str(file)} -mul -1 -o {str(inv_file)}')

                #  #/ file.name.replace('.nii.gz', '_seg-manual.nii.gz')
                print(" Segmentation of the spinal cord for file: ", file.name)
                os.system(f'python /home/GRAMES.POLYMTL.CA/p119007/ms_lesion_agnostic/canproco/packaging/run_inference_single_subject.py --path-image {str(inv_file)} --path-out {str(output_dir)} --path-model {str(model_path)}')
                seg_file = output_dir / file.name.replace('.nii.gz', '_pred.nii.gz')
                # binarize the output
                os.system(f'sct_maths -i {str(seg_file)} -bin 0.2 -o {str(seg_file)}')
                
                # remove inversed file
                os.system(f'rm {str(output_dir / file.name)}')

                # produce the QC report
                
                os.system(f'sct_qc -i {str(file)} -s {seg_file} -d {seg_file} -p sct_deepseg_lesion -plane sagittal -qc {canproco_qc_folder}')
                # os.system(f'sct_deepseg -i {str(file)} -o {output_file} -task seg_sc_contrast_agnostic -thr 0.01')
            


# -----------------------
# BASEL
# -----------------------
# When needed we segment the spinal cord of the Basel dataset
                
# Canproco dataset
basel_path = Path('/home/GRAMES.POLYMTL.CA/p119007/ms_lesion_agnostic/data/basel-mp2rage')

# Output folder
basel_output_folder = Path('/home/GRAMES.POLYMTL.CA/p119007/ms_lesion_agnostic/data/sc_seg/basel_sc_seg')
# QC folder 
basel_qc_folder = Path('/home/GRAMES.POLYMTL.CA/p119007/ms_lesion_agnostic/data/sc_seg/basel_sc_seg_qc')

files = list(basel_path.rglob('*UNIT1.nii.gz'))
count_basel = 0
for file in files:
    if 'SHA256' not in str(file):
        # we check if the file has a mask in the derivatives folder (derivatives/folder and then same relative path)
        relative_path = file.relative_to(basel_path).parent
        lesion_mask1 = basel_path / 'derivatives' / 'labels' / relative_path / file.name.replace('.nii.gz', '_lesion-manualNeuroPoly.nii.gz')
        lesion_mask2 = basel_path / 'derivatives' / 'labels' / relative_path / file.name.replace('.nii.gz', '_lesion-manualKatrin.nii.gz')
        lesion_mask3 = basel_path / 'derivatives' / 'labels' / relative_path / file.name.replace('.nii.gz', '_lesion-manualHaris.nii.gz')
        # if the mask exists we add it either to training or testing
        if lesion_mask1.exists() or lesion_mask2.exists() or lesion_mask3.exists():
            sc_seg = basel_path / 'derivatives' / 'labels' / relative_path / file.name.replace('.nii.gz', '_label-SC_seg.nii.gz')
            if not sc_seg.exists():
                print(f"need to segment {file.name}")
                count_basel += 1
print("Number of files to segment: ", count_basel)
                

# -----------------------
# SCT-TESTING-LARGE
# -----------------------
# When needed we segment the spinal cord of the sct-testing-large dataset
                
# Canproco dataset
sct_testing_path = Path('/home/GRAMES.POLYMTL.CA/p119007/ms_lesion_agnostic/data/sct-testing-large')

# Output folder
sct_testing_output_folder = Path('/home/GRAMES.POLYMTL.CA/p119007/ms_lesion_agnostic/data/sc_seg/sct-testing_sc_seg')
# QC folder 
sct_testing_qc_folder = Path('/home/GRAMES.POLYMTL.CA/p119007/ms_lesion_agnostic/data/sc_seg/sct-testing_sc_seg_qc')

files = list(sct_testing_path.rglob('*_lesion-manual.nii.gz'))
count_sct = 0
for file in files:
    if 'SHA256' not in str(file):
        sc_seg = file.parent / file.name.replace('_lesion-manual.nii.gz', '_seg-manual.nii.gz')
        if not sc_seg.exists():
            # find corresponding image
            image = Path(str(file).replace('_lesion-manual.nii.gz', '.nii.gz').replace('derivatives/labels/', ''))

            # output file path building
            relative_path = image.relative_to(sct_testing_path).parent
            output_dir = sct_testing_output_folder / relative_path
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / file.name.replace('_lesion-manual.nii.gz', '_seg-manual.nii.gz')
            # segmentation of sc using SCT
            print(" Segmentation of the spinal cord for file: ", image.name)
            os.system(f'sct_deepseg -i {str(image)} -o {output_file} -task seg_sc_contrast_agnostic')

            # produce the QC report
            os.system(f'sct_qc -i {str(image)} -s {output_file} -d {output_file} -p sct_deepseg_lesion -plane sagittal -qc {sct_testing_qc_folder}')

            count_sct += 1
print("Number of files to segment: ", count_sct)


# -----------------------
# BAVARIA
# -----------------------
# When needed we segment the spinal cord of the Bavaria dataset
                
# Canproco dataset
bavaria_path = Path('/home/GRAMES.POLYMTL.CA/p119007/ms_lesion_agnostic/data/bavaria-quebec-spine-ms')

# Output folder
bavaria_output_folder = Path('/home/GRAMES.POLYMTL.CA/p119007/ms_lesion_agnostic/data/sc_seg/bavaria_sc_seg')
# QC folder 
bavaria_qc_folder = Path('/home/GRAMES.POLYMTL.CA/p119007/ms_lesion_agnostic/data/sc_seg/bavaria_sc_seg_qc')

files = list(bavaria_path.rglob('*T2w.nii.gz'))
count_bavaria = 0
for file in tqdm.tqdm(files):
    if 'SHA256' not in str(file):
        # we check if the file has a mask in the derivatives folder (derivatives/folder and then same relative path)
        relative_path = file.relative_to(bavaria_path).parent
        lesion_mask = bavaria_path / 'derivatives' / 'labels' / relative_path / file.name.replace('_T2w.nii.gz', '_lesions-manual_T2w.nii.gz')
        # if the mask exists we add it either to training or testing
        if lesion_mask.exists():
            sc_seg = lesion_mask.parent / lesion_mask.name.replace('lesions-manual', 'seg-manual')
            if not sc_seg.exists():
                # output file path building
                output_dir = bavaria_output_folder / relative_path
                output_dir.mkdir(parents=True, exist_ok=True)
                
                print(" Segmentation of the spinal cord for file: ", file.name)
                os.system(f'sct_deepseg -i {str(file)} -o {sc_seg} -task seg_sc_contrast_agnostic')

                # produce the QC report
                os.system(f'sct_qc -i {str(file)} -s {sc_seg} -d {sc_seg} -p sct_deepseg_lesion -plane sagittal -qc {bavaria_qc_folder}')

                count_bavaria += 1
print("Number of files to segment: ", count_bavaria)
                

    