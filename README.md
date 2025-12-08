# Segmentation of spinal cord multiple sclerosis lesions

## Robust Spinal Cord MS Lesion Segmentation Across Diverse MRI Protocols and Centers

TODO: when published: add icon similar to this one (https://github.com/sct-pipeline/contrast-agnostic-softseg-spinalcord/blob/bfcb8352aa2cdbcb356e428181fbc3dbd2fa42ef/README.md?plain=1#L3)

<img src="https://github.com/user-attachments/assets/6c86548a-0a28-40e4-9d21-219ac310d867" width="500"/>

Official repository for the segmentation of multiple sclerosis (MS) spinal cord (SC) lesions.

This repo contains all the code for training the SC MS lesion segmentation model. The code for training is based on the nnUNetv2 framework. The segmentation model is available as part of [Spinal Cord Toolbox (SCT)](https://spinalcordtoolbox.com/stable/index.html) via the sct_deepseg functionality.

### Citation Information

If you find this work and/or code useful for your research, please cite our paper:

```
TODO: when published, add citation.
```

### How to use the model

Install the Spinal Cord Toolbox (SCT) [here](https://spinalcordtoolbox.com/stable/user_section/installation.html).

Run the command: 
```console
sct_deepseg lesion_ms
```

More details can be found in the user section [here](https://spinalcordtoolbox.com/stable/user_section/command-line/sct_deepseg.html#sct-deepseg)

### Annotators

Below is a list of certified radiologists who have helped label lesion segmentation.

- Laurent Létourneau-Guillon
- David Araujo
- Lydia Chougar
- Dumitru Fetco
- Masaaki Hori
- Kouhei Kamiya
- Steven Messina
- Charidimos Tsagkas

### Code description

The repository contains all the code for the SC MS lesion segmentation project:
- `compute_canada_scripts`: code used to train the model on compute canada
- `dataset_aggregation`: code used to aggregate all the data
- `dataset_analysis`: code used to analyze the data
- `evaluation`: code used to evaluate the performance of the model
- `nnunet`: code used for training with nnunet
- `post-processing`: code used for post-processing

## Longitudinal tracking of multiple sclerosis lesions in the spinal cord: A validation study

TODO: if accepted add icon here

<img src="https://github.com/user-attachments/assets/cdb12ab9-aa16-42b9-b0f2-8fcfbf2e8062" width="700"/>


Official repository of the longitudinal lesion tracking of MS SC lesions.

This repo contains the code for evaluating 5 strategies to track SC MS lesions across timepoints. More details can be find in the paper.

```
TODO: if accepted add the citation here
```

All the code is contained in the folder [longitudinal_tracking](./longitudinal_tracking). 