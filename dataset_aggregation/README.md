# Dataset aggregation

The files here are used to aggregate the data in MSD style datasets from the following datasets:
- basel-mp2rage: used for training/testing
- bavaria-quebec-spine-ms-unstitched: used for training/testing
- canproco: used for training/testing
- ms-basel-2018: used for external testing
- ms-basel-2020: used for external testing
- ms-karolinska-2020: used for external testing
- ms-nyu: used for training/testing
- nih-ms-mp2rage: used for training/testing
- sct-testing-large: used for training/testing

The scripts use the following exclude files:
- [`exclude.yml`](./exclude.yml): in this repo
- [`exclude.yml`](https://github.com/ivadomed/canproco/blob/main/exclude.yml): in the canproco repo (to exclude files specifically in the canproco repo)

The files in this repo are: 
- `create_msd_data.py`: creates the basic MSD dataset with the above listed datasets
- ~~`create_msd_data_head_cropped.py` : creates the MSD dataset similarly to the code above, but also crops the head.~~ (This file was removed as it wasn't used)
- [`qc_datasets.py`](./qc_datasets.py): used to QC the dataset and check if lesions are located in the spinal cord
