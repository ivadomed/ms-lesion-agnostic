#!/bin/bash

# Echo time and hostname into log
echo "Date:     $(date)"
echo "Hostname: $(hostname)"

# activate environment
echo "Activating environment ..."
source /home/p/plb/links/projects/aip-jcohen/plb/final_trainings/.venv_job200/bin/activate

# Define paths used:
PATH_NNUNET_RAW_FOLDER="/home/p/plb/links/projects/aip-jcohen/plb/nnUNet_experiments/nnUNet_raw"
PATH_NNUNET_PREPROCESSED_FOLDER="/home/p/plb/links/projects/aip-jcohen/plb/final_trainings/nnUNet_preprocessed"

# Create the nnUNet_preprocessed
mkdir -p $PATH_NNUNET_PREPROCESSED_FOLDER

# Export nnUNet paths
export nnUNet_raw=${PATH_NNUNET_RAW_FOLDER}
export nnUNet_preprocessed=${PATH_NNUNET_PREPROCESSED_FOLDER}

echo "nnUNet_raw: $nnUNet_raw"
echo "nnUNet_preprocessed: $nnUNet_preprocessed"    

# Define dataset values
dataset_number=902
configurations="3d_fullres"
fold=0
planner="nnUNetPlannerResEncL"
plans="nnUNetResEncUNetL1x1x1_Model2_Plans"

# First we preprocess the nnUNet_raw data

# Then we copy the plans file in the nnUNet_preprocessed folder
echo ""
echo "Copying the plans file in the nnUNet_preprocessed folder"
cp /home/p/plb/links/projects/aip-jcohen/plb/final_trainings/$plans.json $nnUNet_preprocessed/Dataset902_msLesionAgnostic/

# Then we preprocess the data with the new plans file
echo ""
echo "Preprocessing the nnUNet_raw data with the new plans file"
nnUNetv2_preprocess -d $dataset_number -plans_name $plans
