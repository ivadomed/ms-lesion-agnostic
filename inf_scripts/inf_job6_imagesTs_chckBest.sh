#!/bin/bash

# Echo time and hostname into log
echo "Date:     $(date)"
echo "Hostname: $(hostname)"

# activate environment
echo "Activating environment ..."
source /home/p/plb/links/scratch/ms-lesion-agnostic/model_trainings/job11x12x13x14x15/.venv_job11x12x13x14x15/bin/activate

# Define paths used:
PATH_INPUT_DATASET="/home/p/plb/links/projects/aip-jcohen/plb/nnUNet_experiments/nnUNet_raw/Dataset902_msLesionAgnostic"
PATH_JOB="/home/p/plb/links/projects/aip-jcohen/plb/results/job6"

# Export nnUNet path result
export nnUNet_results=${PATH_JOB}/nnUNet_results

echo "nnUNet_results: $nnUNet_results"

# Define dataset values
dataset_number=902
configurations="3d_fullres"
fold=0
planner="nnUNetPlannerResEncL"
plans="nnUNetResEncUNetL1x1x1_Model1_Plans"
trainer="nnUNetTrainerDiceCELoss_noSmooth_2000epochs"

model_checkpoint="checkpoint_best.pth"
image_folder="imagesTs"
output_folder="imagesTs_pred_chckBest_fold0"

# First we preprocess the nnUNet_raw data
echo "Running inference on imagesTs with chck_best"

CUDA_VISIBLE_DEVICES=2 nnUNetv2_predict -i $PATH_INPUT_DATASET/$image_folder -o $PATH_JOB/$output_folder \
    -d 902 -p $plans -tr $trainer -c $configurations -f $fold -chk $model_checkpoint --continue_prediction
