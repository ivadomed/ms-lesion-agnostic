#!/bin/bash
job_name="job9"

# Echo time and hostname into log
echo "Date:     $(date)"
echo "Hostname: $(hostname)"

# activate environment
echo "Activating environment ..."
source /home/p/plb/links/scratch/ms-lesion-agnostic/model_trainings/$job_name/.venv_$job_name/bin/activate        # TODO: update to match the name of your environment

# Definr paths used:
PATH_NNUNET_RAW_FOLDER="/home/p/plb/links/projects/aip-jcohen/plb/nnUNet_experiments/nnUNet_raw"
PATH_MSD_DATA="/home/p/plb/links/projects/aip-jcohen/plb/msd_data/dataset_2025-04-15_seed42.json"
PATH_OUTPUT="/home/p/plb/links/scratch/ms-lesion-agnostic/model_trainings/"$job_name

# Create the nnUNet_preprocessed and nnUNet_results folders
mkdir -p $PATH_OUTPUT/nnUNet_preprocessed
mkdir -p $PATH_OUTPUT/nnUNet_results

# Export nnUNet paths
export nnUNet_raw=${PATH_NNUNET_RAW_FOLDER}
export nnUNet_preprocessed=${PATH_OUTPUT}/nnUNet_preprocessed
export nnUNet_results=${PATH_OUTPUT}/nnUNet_results

echo "nnUNet_raw: $nnUNet_raw"
echo "nnUNet_preprocessed: $nnUNet_preprocessed"    
echo "nnUNet_results: $nnUNet_results"

# Define dataset values
dataset_number=902
configurations="3d_fullres"
fold=0
planner="nnUNetPlannerResEncL"
plans="nnUNetResEncUNetL1x1x1_Model1_Plans"
trainer="nnUNetTrainerDAExt_DiceCELoss_noSmooth_unbalancedSampling_2000epochs"
pretrained_model_path="/home/p/plb/links/scratch/ms-lesion-agnostic/model_trainings/job6/pretrained_model/Dataset617_nativect/MultiTalent_trainer_4000ep__nnUNetResEncUNetL1x1x1_Plans_bs24__3d_fullres/fold_all/checkpoint_final.pth"

# First we preprocess the nnUNet_raw data
## Echo the command to be run
echo ""
echo "Preprocessing the nnUNet_raw data"
echo "nnUNetv2_plan_and_preprocess -d $dataset_number -c $configurations  --verify_dataset_integrity"
## Run the command
nnUNetv2_plan_and_preprocess -d $dataset_number -c $configurations --verify_dataset_integrity

# Then we copy the plans file in the nnUNet_preprocessed folder
echo ""
echo "Copying the plans file in the nnUNet_preprocessed folder"
cp /home/p/plb/links/scratch/ms-lesion-agnostic/model_trainings/$job_name/$plans.json $nnUNet_preprocessed/Dataset902_msLesionAgnostic/

# Then we preprocess the data with the new plans file
echo ""
echo "Preprocessing the nnUNet_raw data with the new plans file"
nnUNetv2_preprocess -d $dataset_number -plans_name $plans 

# We add the probabilities
echo ""
echo "Adding the probabilities to the nnUNet_preprocessed data"
python /home/p/plb/links/scratch/ms-lesion-agnostic/model_trainings/$job_name/nnUNet/add_contrast_probability_to_preprocessed_dataset.py -c $PATH_NNUNET_RAW_FOLDER/Dataset902_msLesionAgnostic/conversion_dict.json -d $nnUNet_preprocessed/Dataset902_msLesionAgnostic/dataset.json

# Model training:
echo ""
echo "Training the model"
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train  $dataset_number  $configurations 0 -p $plans -tr $trainer -pretrained_weights $pretrained_model_path