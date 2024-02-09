#!/bin/bash
#
# Run nnUNet training and testing on ms lesion aggregated dataset
#
# Usage:
#     cd ~/code/model-seg-dcm
#     ./training/01_run_training_dcm-zurich-lesions.sh
#
# Author: Jan Valosek, Naga Karthik, Pierre-Louis Benveniste
#

# Uncomment for full verbose
set -x

# Immediately exit if error
set -e -o pipefail

# Exit if user presses CTRL+C (Linux) or CMD+C (OSX)
trap "echo Caught Keyboard Interrupt within script. Exiting now.; exit" INT


# define arguments for nnUNet
dataset_num="101"
dataset_name="Dataset${dataset_num}_singleClassNnunetMsLesion"
#nnunet_trainer="nnUNetTrainerDiceCELoss_noSmooth"
nnunet_trainer="nnUNetTrainer"
#nnunet_trainer="nnUNetTrainer_2000epochs"       # default: nnUNetTrainer
configuration="3d_fullres"                      # for 2D training, use "2d"
cuda_visible_devices=2
folds=(1)
#folds=(3)
#sites=(dcm-zurich-lesions dcm-zurich-lesions-20231115)
#region_based="--region-based"
#region_based=""


echo "-------------------------------------------------------"
echo "Running preprocessing and verifying dataset integrity"
echo "-------------------------------------------------------"
nnUNetv2_plan_and_preprocess -d ${dataset_num} -c ${configuration}


for fold in ${folds[@]}; do
    echo "-------------------------------------------"
    echo "Training on Fold $fold"
    echo "-------------------------------------------"

    # training
    CUDA_VISIBLE_DEVICES=${cuda_visible_devices} nnUNetv2_train ${dataset_num} ${configuration} ${fold} -tr ${nnunet_trainer}

    echo ""
    echo "-------------------------------------------"
    echo "Training completed, Testing on Fold $fold"
    echo "-------------------------------------------"

    
    CUDA_VISIBLE_DEVICES=${cuda_visible_devices} nnUNetv2_predict -i ${nnUNet_raw}/${dataset_name}/imagesTs -tr ${nnunet_trainer} -o ${nnUNet_results}/${dataset_name}/${nnunet_trainer}__nnUNetPlans__${configuration}/fold_${fold}/test -d ${dataset_num} -f ${fold} -c ${configuration} # -step_size 0.9 --disable_tta

    # echo "-------------------------------------------------------"
    # echo "Running ANIMA evaluation on Test set for ${site} "
    # echo "-------------------------------------------------------"

    # python training/02_compute_anima_metrics.py --pred-folder ${nnUNet_results}/${dataset_name}/${nnunet_trainer}__nnUNetPlans__${configuration}/fold_${fold}/test_${site} --gt-folder ${nnUNet_raw}/${dataset_name}/labelsTs_${site} --dataset-name ${site} ${region_based}

    done

done