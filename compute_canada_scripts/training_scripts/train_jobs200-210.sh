#!/bin/bash
#SBATCH --account=aip-jcohen
#SBATCH --job-name=job200to210     # set a more descriptive job-name 
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=300G
#SBATCH --time=1-00:00:00   # DD-HH:MM:SS
#SBATCH --output=/home/p/plb/links/scratch/ms-lesion-agnostic/final_trainings/%x_%A_v2.out
#SBATCH --error=/home/p/plb/links/scratch/ms-lesion-agnostic/final_trainings/%x_%A_v2.err
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
PATH_NNUNET_RESULTS_FOLDER="/home/p/plb/links/scratch/ms-lesion-agnostic/final_trainings/nnUNet_results"

# Export nnUNet paths
export nnUNet_raw=${PATH_NNUNET_RAW_FOLDER}
export nnUNet_preprocessed=${PATH_NNUNET_PREPROCESSED_FOLDER}
export nnUNet_results=${PATH_NNUNET_RESULTS_FOLDER}

echo "nnUNet_raw: $nnUNet_raw"
echo "nnUNet_preprocessed: $nnUNet_preprocessed"    
echo "nnUNet_results: $nnUNet_results"

# Common variables
dataset_number=902
configurations="3d_fullres"

#Define variables for model 1
plans_model1="nnUNetResEncUNetL1x1x1_Model1_Plans"
trainer_job200="nnUNetTrainerDiceCELoss_noSmooth_4000epochs"
trainer_job210="nnUNetTrainerDiceCELoss_noSmooth_unbalancedSampling_4000epochs"
pretrained_model1="/home/p/plb/links/projects/aip-jcohen/plb/final_trainings/pretrained_models/Dataset617_nativect/MultiTalent_trainer_4000ep__nnUNetResEncUNetL1x1x1_Plans_bs24__3d_fullres/fold_all/checkpoint_final.pth"

# Launch jobs
parallel --verbose --jobs 8 ::: \
  "(ts=\$(date '+%Y-%m-%d-%H-%M-%S');  CUDA_VISIBLE_DEVICES=0 nnUNetv2_train $dataset_number  $configurations 0 -p $plans_model1 -tr $trainer_job200 -pretrained_weights $pretrained_model1 2>&1 | tee /home/p/plb/links/scratch/ms-lesion-agnostic/final_trainings/logfile_job200_fold0_\$ts.txt)" \
  "(ts=\$(date '+%Y-%m-%d-%H-%M-%S');  CUDA_VISIBLE_DEVICES=0 nnUNetv2_train $dataset_number  $configurations 1 -p $plans_model1 -tr $trainer_job200 -pretrained_weights $pretrained_model1 2>&1 | tee /home/p/plb/links/scratch/ms-lesion-agnostic/final_trainings/logfile_job200_fold1_\$ts.txt)" \
  "(ts=\$(date '+%Y-%m-%d-%H-%M-%S');  CUDA_VISIBLE_DEVICES=1 nnUNetv2_train $dataset_number  $configurations 2 -p $plans_model1 -tr $trainer_job200 -pretrained_weights $pretrained_model1 2>&1 | tee /home/p/plb/links/scratch/ms-lesion-agnostic/final_trainings/logfile_job200_fold2_\$ts.txt)" \
  "(ts=\$(date '+%Y-%m-%d-%H-%M-%S');  CUDA_VISIBLE_DEVICES=1 nnUNetv2_train $dataset_number  $configurations 3 -p $plans_model1 -tr $trainer_job200 -pretrained_weights $pretrained_model1 2>&1 | tee /home/p/plb/links/scratch/ms-lesion-agnostic/final_trainings/logfile_job200_fold3_\$ts.txt)" \
  "(ts=\$(date '+%Y-%m-%d-%H-%M-%S');  CUDA_VISIBLE_DEVICES=2 nnUNetv2_train $dataset_number  $configurations 4 -p $plans_model1 -tr $trainer_job200 -pretrained_weights $pretrained_model1 2>&1 | tee /home/p/plb/links/scratch/ms-lesion-agnostic/final_trainings/logfile_job200_fold4_\$ts.txt)" \
  "(ts=\$(date '+%Y-%m-%d-%H-%M-%S');  CUDA_VISIBLE_DEVICES=2 nnUNetv2_train $dataset_number  $configurations 0 -p $plans_model1 -tr $trainer_job210 -pretrained_weights $pretrained_model1 2>&1 | tee /home/p/plb/links/scratch/ms-lesion-agnostic/final_trainings/logfile_job210_fold0_\$ts.txt)" \
  "(ts=\$(date '+%Y-%m-%d-%H-%M-%S');  CUDA_VISIBLE_DEVICES=3 nnUNetv2_train $dataset_number  $configurations 1 -p $plans_model1 -tr $trainer_job210 -pretrained_weights $pretrained_model1 2>&1 | tee /home/p/plb/links/scratch/ms-lesion-agnostic/final_trainings/logfile_job210_fold1_\$ts.txt)" \
  "(ts=\$(date '+%Y-%m-%d-%H-%M-%S');  CUDA_VISIBLE_DEVICES=3 nnUNetv2_train $dataset_number  $configurations 2 -p $plans_model1 -tr $trainer_job210 -pretrained_weights $pretrained_model1 2>&1 | tee /home/p/plb/links/scratch/ms-lesion-agnostic/final_trainings/logfile_job210_fold2_\$ts.txt)" \