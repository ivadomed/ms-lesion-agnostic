#!/bin/bash
#SBATCH --account=aip-jcohen
#SBATCH --job-name=preprocModel1     # set a more descriptive job-name 
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --time=1-00:00:00   # DD-HH:MM:SS
#SBATCH --output=/home/p/plb/links/projects/aip-jcohen/plb/final_trainings/%x_%A_v2.out
#SBATCH --error=/home/p/plb/links/projects/aip-jcohen/plb/final_trainings/%x_%A_v2.err
#SBATCH --mail-user=pierrelouis.benveniste03@gmail.com     # whenever the job starts/fails/completes, an email will be sent 
#SBATCH --mail-type=ALL


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
plans="nnUNetResEncUNetL1x1x1_Model1_Plans"

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
cp /home/p/plb/links/projects/aip-jcohen/plb/final_trainings/$plans.json $nnUNet_preprocessed/Dataset902_msLesionAgnostic/

# Then we preprocess the data with the new plans file
echo ""
echo "Preprocessing the nnUNet_raw data with the new plans file"
nnUNetv2_preprocess -d $dataset_number -plans_name $plans

# We add the probabilities
python /home/p/plb/links/projects/aip-jcohen/plb/final_trainings/nnUNet/add_contrast_probability_to_preprocessed_dataset.py -c $PATH_NNUNET_RAW_FOLDER/Dataset902_msLesionAgnostic/conversion_dict.json -d $nnUNet_preprocessed/Dataset902_msLesionAgnostic/dataset.json