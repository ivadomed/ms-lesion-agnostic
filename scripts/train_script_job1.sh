#!/bin/bash
#SBATCH --account=aip-jcohen
#SBATCH --job-name=job1     # set a more descriptive job-name 
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --time=1-00:00:00   # DD-HH:MM:SS
#SBATCH --output=/home/p/plb/links/scratch/ms-lesion-agnostic/model_trainings/job1/%x_%A_v2.out
#SBATCH --error=/home/p/plb/links/scratch/ms-lesion-agnostic/model_trainings/job1/%x_%A_v2.err
#SBATCH --mail-user=pierrelouis.benveniste03@gmail.com     # whenever the job starts/fails/completes, an email will be sent 
#SBATCH --mail-type=ALL

job_name="job1"

# Echo time and hostname into log
echo "Date:     $(date)"
echo "Hostname: $(hostname)"

# load the required modules
echo "Loading modules ..."
module load python/3.10.13 cuda/12.2    # TODO: might differ depending on the python and cuda version you have

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
plans="nnUNetResEncUNetLPlans"
trainer="nnUNetTrainerDiceCELoss_2000epochs"
model_checkpoint="checkpoint_final.pth"

# First we preprocess the nnUNet_raw data
## Echo the command to be run
echo ""
echo "Preprocessing the nnUNet_raw data"
echo "nnUNetv2_plan_and_preprocess -d $dataset_number -c $configurations -pl $planner --verify_dataset_integrity"
## Run the command
# nnUNetv2_plan_and_preprocess -d $dataset_number -c $configurations -pl $planner --verify_dataset_integrity

# Model training:
echo ""
echo "Training the model"
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train  $dataset_number  $configurations  0 -p $plans -tr $trainer &
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train  $dataset_number  $configurations  1 -p $plans -tr $trainer &
CUDA_VISIBLE_DEVICES=2 nnUNetv2_train  $dataset_number  $configurations  2 -p $plans -tr $trainer & 
CUDA_VISIBLE_DEVICES=3 nnUNetv2_train  $dataset_number  $configurations  3 -p $plans -tr $trainer &