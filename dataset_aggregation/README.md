# Dataset aggregation

The files here are used to aggregate the data in MSD style datasets from the following datasets:
- labeled datasets: 
  - basel-mp2rage
  - bavaria-quebec-spine-ms-unstitched
  - canproco
  - ms-basel-2018
  - ms-basel-2020
  - ms-karolinska-2020
  - ms-nyu
  - nih-ms-mp2rage
  - sct-testing-large
- unlabeled datasets
  - ms-mayo-critical-lesions
  - ms-nmo-beijing
  - umass-ms-ge-hdxt1.5
  - umass-ms-ge-pioneer3
  - umass-ms-siemens-espree1.5
  - umass-ms-ge-excite1.5

The scripts use the following exclude files:
- [`exclude.yml`](./exclude.yml): in this repo
- [`exclude.yml`](https://github.com/ivadomed/canproco/blob/main/exclude.yml): in the canproco repo (to exclude files specifically in the canproco repo)

The files in this repo are: 
- [`create_msd_data.py`](./create_msd_data.py): creates the basic MSD dataset with the above listed datasets
- [`qc_datasets.py`](./qc_datasets.py): used to QC the dataset and check if lesions are located in the spinal cord
- [agregate_unannotated_data.py](./agregate_unannotated_data.py): used to aggregate the unannotated datasets
