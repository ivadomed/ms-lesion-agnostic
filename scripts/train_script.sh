#!/bin/bash
# This script is used for training contrast-agnostic v3.0 and also provides the option to extend the 
# contrast-agnostic spinal cord segmentation model with new datasets. It achieves the following:
# 1. Clones the datasets from NeuroPoly's git-annex server. 
# 2. Creates datalists (i.e. json files with image/label pairs) based on pre-defined or random dataset splits 
# 3. Converts the json files for each dataset into one aggregated dataset in the nnUNet format
# 4. Runs nnUNet preprocessing and training based on the defined configurations (2D/3D).


# Define (full) path to the contrast-agnostic repository
PATH_REPO="/home/GRAMES.POLYMTL.CA/u114716/contrast-agnostic/contrast-agnostic-softseg-spinalcord"


# ====================================
# VARIABLES FOR DATASET CREATION
# ====================================

# Set seed for reproducibility. Note seed=50 was used train the contrast-agnostic model 
# If you're using a different seed, note that you cannot use the predefined random dataset splits (more info below)
SEED=50

# List of datasets to train on
# NOTE 1: the following datasets were used for training the contrast-agnostic v3.0 model
# https://github.com/sct-pipeline/contrast-agnostic-softseg-spinalcord/releases/tag/v3.0
# NOTE 2: training on praxis acute SCI data requires special access to spineimage.ca. Because this is different from
# the usual downloading from git-annex, this script does not support downloading praxis data. To train contrast-agnostic model 
# download the dataset manually and store it in PATH_DATA_BASE (see below)

DATASETS=("data-multi-subject" "basel-mp2rage" "canproco" \
            "lumbar-epfl" "lumbar-vanderbilt" "dcm-brno" "dcm-zurich" "dcm-zurich-lesions" "dcm-zurich-lesions-20231115" \
            "sci-paris" "sci-zurich" "sci-colorado" "sct-testing-large" \
            "site_006" "site_007"
            )
DATASETS=("site_006")

# Path to the folder where the datasets will be downloaded
# PATH_DATA_BASE="/home/GRAMES.POLYMTL.CA/u114716/datasets"
PATH_DATA_BASE="/scratch/naga/contrast_agnostic/datasets"

# Path to the output folder where the dataset in MSD-style format will be saved as json files with image/label pairs
# and other dataset-related statistics. To keep track of the experiments, date is also appended as a prefix or suffix
# Example: 20250211_v31contrastAgnostic
folder_name=$(date +"%Y%m%d")-temp
PATH_OUT_DATALISTS="/scratch/naga/contrast_agnostic/datalists/${folder_name}"

# Path to yml file containing subjects to include. These subjects are curated to be of good quality after visual QC'ing
# Always include this file, when reproducing and also when adding new datasets
PATH_INCLUDE_SUBJECTS=${PATH_REPO}/subjects_to_include.yml


# ====================================
# VARIABLES FOR NNUNET TRAINING
# ====================================

# Path to store the converted dataset (ideally the ${nnUNet_raw} folder once nnUNet is installed)
PATH_NNUNET_RAW="/home/GRAMES.POLYMTL.CA/u114716/nnunet-v2/nnUNet_raw"

# Path to the nnUNet results folder (ideally ${nnUNet_results})
PATH_NNUNET_RESULTS="/home/GRAMES.POLYMTL.CA/u114716/nnunet-v2/nnUNet_results"

# Name and number/id of the dataset to be referenced by nnunet
DATASET_NAME="TempContrastAgnostic"
DATASET_NUMBER=999          # this refers to the `-d` argument when training nnunet models

# Name of the nnUNet trainer variant 
# NOTE: contrast-agnostic v3.0 model used the default trainer defined below
# NNUNET_TRAINER="nnUNetTrainer"
NNUNET_TRAINER="nnUNetTrainer_5epochs"

# Name of the plans file. Recommended to keep the default one below unless you want to train
# other models given in nnunet's model suite
NNUNET_PLANS_FILE="nnUNetPlans"

# Type/Kernel of the model. for 2D training, use "2d"; for 3D training, use "3d_fullres"
# configurations=("2d" "3d_fullres")                        
configurations=("3d_fullres")

# Number of cross-validation folds to run the model on. nnUNet by default allows training on 5 folds
# folds=(0 1 2 3 4)
folds=(0)

# GPU ID to use for training the model
cuda_visible_devices=2


# ====================================
# DOWNLOADING DATASETS AND CREATING DATALISTS
# ====================================

for dataset in ${DATASETS[@]}; do

    if [[ ${dataset} == site_* ]]; then
        echo "-----------------------------------"
        echo "Encountered possibly a PRAXIS dataset, checking if it is already downloaded ..."
        echo "-----------------------------------"
        if [[ ! -d "${PATH_DATA_BASE}/${dataset}" ]]; then
            echo "No dataset found in ${PATH_DATA_BASE}/${dataset}, please download the praxis dataset manually"
            exit
        else
            echo "Dataset found at ${PATH_DATA_BASE}/${dataset}, moving on to datalist creation ..."
        fi
    else
        echo "-----------------------------------"
        echo "Cloning ${dataset} dataset from git-annex ..."
        echo "-----------------------------------"
        python ${PATH_REPO}/nnUnet/01_clone_dataset.py \
            --ofolder ${PATH_DATA_BASE} \
            --dataset ${dataset} 
    
    fi

    echo "-----------------------------------"
    echo "Downloading data from git-annex and creating datalist for ${dataset} ..."
    echo "-----------------------------------"
    
    python ${PATH_REPO}/nnUnet/02_create_msd_data.py \
        --seed ${SEED} \
        --path-data ${PATH_DATA_BASE}/${dataset} \
        --path-out ${PATH_OUT_DATALISTS} \
        --include ${PATH_INCLUDE_SUBJECTS} \
        --use-predefined-splits \
        --path-datasplits ${PATH_REPO}/datasplits

done

echo "-----------------------------------"
echo "Done! Datalists created and stored in ${PATH_OUT_DATALISTS}"
echo "-----------------------------------"


# ====================================
# CREATING DATATSET IN NNUNET FORMAT
# ====================================

# NOTE: Run the command below only when all datalists are created. 
# nnUNet has a numbering system for its datasets that requires all images to be converted once
# rather than greedy conversion (i.e. creating a datalist and immediately converting to nnunet format)
echo "-----------------------------------"
echo "Converting the datalists to nnUNetv2-specific format ..."
echo "-----------------------------------"

# NOTE: When using all the datasets, this commands takes a while (8-10 hours) because of the conversion
# to RPI and ensuring the alignment of images and labels. While this is already using multiprocessing, 
# depending on sct commands is what is takes a long time (e.g. sct_register_multimodal). 
# But, the good part is, once this is done, there is no way nnUNet will throw an error regarding 
# image/label header mismatch. 

python ${PATH_REPO}/nnUnet/03_convert_msd_to_nnunet_reorient.py \
    --input ${PATH_OUT_DATALISTS} \
    --output ${PATH_NNUNET_RAW} \
    --taskname ${DATASET_NAME} \
    --tasknumber ${DATASET_NUMBER} \
    --workers 8

echo "-----------------------------------"
echo "Done! Converted datasets can be found in ${PATH_NNUNET_RAW}/${DATASET_NAME}."
echo "-----------------------------------"


# ====================================
# NNUNET PREPROCESSING
# ====================================

# NOTE: For large datasets, preprocessing takes a lot of time, hence we run a separate loop over the 
# configurations so that once it's done, this part can be commented out and jump to training directly
for configuration in ${configurations[@]}; do

    echo "-----------------------------------"
    echo "Verifying dataset integrity and running preprocessing for ${configuration} configuration ..."
    echo "-----------------------------------"
    nnUNetv2_plan_and_preprocess -d ${DATASET_NUMBER} --verify_dataset_integrity -c ${configuration}

done

echo "-----------------------------------"
echo "Preprocessing completed. Starting training ..."
echo "-----------------------------------"


# ====================================
# NNUNET TRAINING
# ====================================

for configuration in ${configurations[@]}; do

    for fold in ${folds[@]}; do

        echo "-------------------------------------------"
        echo "Training on Fold $fold, Configuration $configuration ..."
        echo "-------------------------------------------"

        # Get the start time
        start=$(date +%s)

        # training
        CUDA_VISIBLE_DEVICES=${cuda_visible_devices} nnUNetv2_train ${DATASET_NUMBER} \
                                $configuration $fold -tr ${NNUNET_TRAINER} -p ${NNUNET_PLANS_FILE}

        echo ""
        echo "-------------------------------------------"
        echo "Training completed for Fold $fold, Configuration $configuration"
        echo "-------------------------------------------"

        # Get the end time
        end=`date +%s`
        runtime=$((end-start))
        echo
        echo "~~~"
        echo "Ran on:      `uname -nsr`"
        echo "Duration:    $(($runtime / 3600))hrs $((($runtime / 60) % 60))min $(($runtime % 60))sec"
        echo "~~~"
    done

done

echo "-------------------------------------------"
echo "Training on all folds completed."
echo "Model can be found in ${PATH_NNUNET_RESULTS}/${DATASET_NAME}"
echo "-------------------------------------------"