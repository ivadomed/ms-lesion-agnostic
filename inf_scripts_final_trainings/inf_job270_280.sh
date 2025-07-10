#!/bin/bash
#SBATCH --account=aip-jcohen
#SBATCH --job-name=inf_job270to280
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

PATH_IMAGESTR="/home/p/plb/links/projects/aip-jcohen/plb/nnUNet_experiments/nnUNet_raw/Dataset902_msLesionAgnostic/imagesTr"
PATH_IMAGESTS="/home/p/plb/links/projects/aip-jcohen/plb/nnUNet_experiments/nnUNet_raw/Dataset902_msLesionAgnostic/imagesTs"
PATH_OUT="/home/p/plb/links/scratch/ms-lesion-agnostic/final_trainings/predictions"

echo "nnUNet_results: $nnUNet_results"

# Make directories for outputs
mkdir -p $PATH_OUT/job270
mkdir -p $PATH_OUT/job280

# Common variables
dataset_number=902
configurations="3d_fullres"
model_checkpoint="checkpoint_best.pth"

# Define 
model_checkpoint="checkpoint_best.pth"
plans_model_multistem="nnUNetResEncUNetL1x1x1_Multistem_Plans"
trainer_job270="nnUNetTrainerDiceCELoss_noSmooth_4000epochs_stem351_5"
trainer_job280="nnUNetTrainerDiceCELoss_noSmooth_unbalancedSampling_4000epochs_stem351_5"

# First we preprocess the nnUNet_raw data
echo "Running inference on imagesTr with chck_best"

# Launch jobs
parallel --verbose --jobs 20 ::: \
    "(ts=\$(date '+%Y-%m-%d-%H-%M-%S'); CUDA_VISIBLE_DEVICES=0 nnUNetv2_predict -i $PATH_IMAGESTR -o $PATH_OUT/job270/imagesTr_fold0 \
    -d 902 -p $plans_model_multistem -tr $trainer_job270 -c $configurations -f 0 -chk $model_checkpoint  --continue_prediction 2>&1 | tee \
    $PATH_OUT/job270/logfile_inf_job270_imagesTr_fold0_\$ts.txt)" \
    "(ts=\$(date '+%Y-%m-%d-%H-%M-%S'); CUDA_VISIBLE_DEVICES=0 nnUNetv2_predict -i $PATH_IMAGESTR -o $PATH_OUT/job270/imagesTr_fold1 \
    -d 902 -p $plans_model_multistem -tr $trainer_job270 -c $configurations -f 1 -chk $model_checkpoint  --continue_prediction 2>&1 | tee \
    $PATH_OUT/job270/logfile_inf_job270_imagesTr_fold1_\$ts.txt)" \
    "(ts=\$(date '+%Y-%m-%d-%H-%M-%S'); CUDA_VISIBLE_DEVICES=1 nnUNetv2_predict -i $PATH_IMAGESTR -o $PATH_OUT/job270/imagesTr_fold2 \
    -d 902 -p $plans_model_multistem -tr $trainer_job270 -c $configurations -f 2 -chk $model_checkpoint  --continue_prediction 2>&1 | tee \
    $PATH_OUT/job270/logfile_inf_job270_imagesTr_fold2_\$ts.txt)" \
    "(ts=\$(date '+%Y-%m-%d-%H-%M-%S'); CUDA_VISIBLE_DEVICES=1 nnUNetv2_predict -i $PATH_IMAGESTR -o $PATH_OUT/job270/imagesTr_fold3 \
    -d 902 -p $plans_model_multistem -tr $trainer_job270 -c $configurations -f 3 -chk $model_checkpoint  --continue_prediction 2>&1 | tee \
    $PATH_OUT/job270/logfile_inf_job270_imagesTr_fold3_\$ts.txt)" \
    "(ts=\$(date '+%Y-%m-%d-%H-%M-%S'); CUDA_VISIBLE_DEVICES=2 nnUNetv2_predict -i $PATH_IMAGESTR -o $PATH_OUT/job270/imagesTr_fold4 \
    -d 902 -p $plans_model_multistem -tr $trainer_job270 -c $configurations -f 4 -chk $model_checkpoint  --continue_prediction 2>&1 | tee \
    $PATH_OUT/job270/logfile_inf_job270_imagesTr_fold4_\$ts.txt)" \

    "(ts=\$(date '+%Y-%m-%d-%H-%M-%S'); CUDA_VISIBLE_DEVICES=0 nnUNetv2_predict -i $PATH_IMAGESTS -o $PATH_OUT/job270/imagesTs_fold0 \
    -d 902 -p $plans_model_multistem -tr $trainer_job270 -c $configurations -f 0 -chk $model_checkpoint  --continue_prediction 2>&1 | tee \
    $PATH_OUT/job270/logfile_inf_job270_imagesTs_fold0_\$ts.txt)" \
    "(ts=\$(date '+%Y-%m-%d-%H-%M-%S'); CUDA_VISIBLE_DEVICES=0 nnUNetv2_predict -i $PATH_IMAGESTS -o $PATH_OUT/job270/imagesTs_fold1 \
    -d 902 -p $plans_model_multistem -tr $trainer_job270 -c $configurations -f 1 -chk $model_checkpoint  --continue_prediction 2>&1 | tee \
    $PATH_OUT/job270/logfile_inf_job270_imagesTs_fold1_\$ts.txt)" \
    "(ts=\$(date '+%Y-%m-%d-%H-%M-%S'); CUDA_VISIBLE_DEVICES=1 nnUNetv2_predict -i $PATH_IMAGESTS -o $PATH_OUT/job270/imagesTs_fold2 \
    -d 902 -p $plans_model_multistem -tr $trainer_job270 -c $configurations -f 2 -chk $model_checkpoint  --continue_prediction 2>&1 | tee \
    $PATH_OUT/job270/logfile_inf_job270_imagesTs_fold2_\$ts.txt)" \
    "(ts=\$(date '+%Y-%m-%d-%H-%M-%S'); CUDA_VISIBLE_DEVICES=1 nnUNetv2_predict -i $PATH_IMAGESTS -o $PATH_OUT/job270/imagesTs_fold3 \
    -d 902 -p $plans_model_multistem -tr $trainer_job270 -c $configurations -f 3 -chk $model_checkpoint  --continue_prediction 2>&1 | tee \
    $PATH_OUT/job270/logfile_inf_job270_imagesTs_fold3_\$ts.txt)" \
    "(ts=\$(date '+%Y-%m-%d-%H-%M-%S'); CUDA_VISIBLE_DEVICES=2 nnUNetv2_predict -i $PATH_IMAGESTS -o $PATH_OUT/job270/imagesTs_fold4 \
    -d 902 -p $plans_model_multistem -tr $trainer_job270 -c $configurations -f 4 -chk $model_checkpoint  --continue_prediction 2>&1 | tee \
    $PATH_OUT/job270/logfile_inf_job270_imagesTs_fold4_\$ts.txt)" \

    "(ts=\$(date '+%Y-%m-%d-%H-%M-%S'); CUDA_VISIBLE_DEVICES=2 nnUNetv2_predict -i $PATH_IMAGESTR -o $PATH_OUT/job280/imagesTr_fold0 \
    -d 902 -p $plans_model_multistem -tr $trainer_job280 -c $configurations -f 0 -chk $model_checkpoint  --continue_prediction 2>&1 | tee \
    $PATH_OUT/job280/logfile_inf_job280_imagesTr_fold0_\$ts.txt)" \
    "(ts=\$(date '+%Y-%m-%d-%H-%M-%S'); CUDA_VISIBLE_DEVICES=3 nnUNetv2_predict -i $PATH_IMAGESTR -o $PATH_OUT/job280/imagesTr_fold1 \
    -d 902 -p $plans_model_multistem -tr $trainer_job280 -c $configurations -f 1 -chk $model_checkpoint  --continue_prediction 2>&1 | tee \
    $PATH_OUT/job280/logfile_inf_job280_imagesTr_fold1_\$ts.txt)" \
    "(ts=\$(date '+%Y-%m-%d-%H-%M-%S'); CUDA_VISIBLE_DEVICES=3 nnUNetv2_predict -i $PATH_IMAGESTR -o $PATH_OUT/job280/imagesTr_fold2 \
    -d 902 -p $plans_model_multistem -tr $trainer_job280 -c $configurations -f 2 -chk $model_checkpoint  --continue_prediction 2>&1 | tee \
    $PATH_OUT/job280/logfile_inf_job280_imagesTr_fold2_\$ts.txt)" \
    "(ts=\$(date '+%Y-%m-%d-%H-%M-%S'); CUDA_VISIBLE_DEVICES=0 nnUNetv2_predict -i $PATH_IMAGESTR -o $PATH_OUT/job280/imagesTr_fold3 \
    -d 902 -p $plans_model_multistem -tr $trainer_job280 -c $configurations -f 3 -chk $model_checkpoint  --continue_prediction 2>&1 | tee \
    $PATH_OUT/job280/logfile_inf_job280_imagesTr_fold3_\$ts.txt)" \
    "(ts=\$(date '+%Y-%m-%d-%H-%M-%S'); CUDA_VISIBLE_DEVICES=1 nnUNetv2_predict -i $PATH_IMAGESTR -o $PATH_OUT/job280/imagesTr_fold4 \
    -d 902 -p $plans_model_multistem -tr $trainer_job280 -c $configurations -f 4 -chk $model_checkpoint  --continue_prediction 2>&1 | tee \
    $PATH_OUT/job280/logfile_inf_job280_imagesTr_fold4_\$ts.txt)" \

    "(ts=\$(date '+%Y-%m-%d-%H-%M-%S'); CUDA_VISIBLE_DEVICES=2 nnUNetv2_predict -i $PATH_IMAGESTS -o $PATH_OUT/job280/imagesTs_fold0 \
    -d 902 -p $plans_model_multistem -tr $trainer_job280 -c $configurations -f 0 -chk $model_checkpoint  --continue_prediction 2>&1 | tee \
    $PATH_OUT/job280/logfile_inf_job280_imagesTs_fold0_\$ts.txt)" \
    "(ts=\$(date '+%Y-%m-%d-%H-%M-%S'); CUDA_VISIBLE_DEVICES=3 nnUNetv2_predict -i $PATH_IMAGESTS -o $PATH_OUT/job280/imagesTs_fold1 \
    -d 902 -p $plans_model_multistem -tr $trainer_job280 -c $configurations -f 1 -chk $model_checkpoint  --continue_prediction 2>&1 | tee \
    $PATH_OUT/job280/logfile_inf_job280_imagesTs_fold1_\$ts.txt)" \
    "(ts=\$(date '+%Y-%m-%d-%H-%M-%S'); CUDA_VISIBLE_DEVICES=3 nnUNetv2_predict -i $PATH_IMAGESTS -o $PATH_OUT/job280/imagesTs_fold2 \
    -d 902 -p $plans_model_multistem -tr $trainer_job280 -c $configurations -f 2 -chk $model_checkpoint  --continue_prediction 2>&1 | tee \
    $PATH_OUT/job280/logfile_inf_job280_imagesTs_fold2_\$ts.txt)" \
    "(ts=\$(date '+%Y-%m-%d-%H-%M-%S'); CUDA_VISIBLE_DEVICES=0 nnUNetv2_predict -i $PATH_IMAGESTS -o $PATH_OUT/job280/imagesTs_fold3 \
    -d 902 -p $plans_model_multistem -tr $trainer_job280 -c $configurations -f 3 -chk $model_checkpoint  --continue_prediction 2>&1 | tee \
    $PATH_OUT/job280/logfile_inf_job280_imagesTs_fold3_\$ts.txt)" \
    "(ts=\$(date '+%Y-%m-%d-%H-%M-%S'); CUDA_VISIBLE_DEVICES=1 nnUNetv2_predict -i $PATH_IMAGESTS -o $PATH_OUT/job280/imagesTs_fold4 \
    -d 902 -p $plans_model_multistem -tr $trainer_job280 -c $configurations -f 4 -chk $model_checkpoint  --continue_prediction 2>&1 | tee \
    $PATH_OUT/job280/logfile_inf_job280_imagesTs_fold4_\$ts.txt)" \

