import os
import pathlib
import shutil

#output folder
output_folder = pathlib.Path('/home/GRAMES.POLYMTL.CA/p119007/ms_lesion_agnostic/data/all_ms_sc_data')

# Let's first aggregate the CanProCo dataset
canproco_path = pathlib.Path('/home/GRAMES.POLYMTL.CA/p119007/ms_lesion_agnostic/data/canproco')

# copy all PSIR and STIR files in 
for root, dirs, files in os.walk(canproco_path):
    for file in files:
        if file.endswith('.nii.gz'):
            if 'PSIR' in file or 'STIR' in file:
                shutil.copy(os.path.join(root, file), output_folder)



