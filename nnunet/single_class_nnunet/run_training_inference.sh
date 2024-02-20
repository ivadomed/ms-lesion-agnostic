#!/bin/bash
#
# Run nnUNet training and testing on ms lesion aggregated dataset
#
# Usage:
#     ./nnunet/single_class_nnunet/run_training_inference.sh
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
configuration="3d_fullres"                      # for 2D training, use "2d"
cuda_visible_devices=3
folds=(1)


echo "-------------------------------------------------------"
echo "Building the dataset"
echo "-------------------------------------------------------"
python ~/ms_lesion_agnostic/ms-lesion-agnostic/nnunet/single_class_nnunet/build_dataset.py 

echo "-------------------------------------------------------"
echo "Converting the dataset to nnUNet format"
echo "-------------------------------------------------------"
python ~/ms_lesion_agnostic/ms-lesion-agnostic/nnunet/single_class_nnunet/convert_BIDS_to_nnunet.py --path-data-json ~/ms_lesion_agnostic/data/all_ms_sc_data/data_singleclass_nnunet.json --path-out ~/ms_lesion_agnostic/data/nnUNet_raw/ --taskname singleClassNnunetMsLesion --tasknumber 101

echo "-------------------------------------------------------"
echo "Running preprocessing"
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
    
    CUDA_VISIBLE_DEVICES=${cuda_visible_devices} nnUNetv2_predict -i ${nnUNet_raw}/${dataset_name}/imagesTs/canproco -tr ${nnunet_trainer} -o ${nnUNet_results}/${dataset_name}/${nnunet_trainer}__nnUNetPlans__${configuration}/fold_${fold}/test_canproco -d ${dataset_num} -f ${fold} -c ${configuration} 
    CUDA_VISIBLE_DEVICES=${cuda_visible_devices} nnUNetv2_predict -i ${nnUNet_raw}/${dataset_name}/imagesTs/basel -tr ${nnunet_trainer} -o ${nnUNet_results}/${dataset_name}/${nnunet_trainer}__nnUNetPlans__${configuration}/fold_${fold}/test_basel -d ${dataset_num} -f ${fold} -c ${configuration} 
    CUDA_VISIBLE_DEVICES=${cuda_visible_devices} nnUNetv2_predict -i ${nnUNet_raw}/${dataset_name}/imagesTs/sct-testing -tr ${nnunet_trainer} -o ${nnUNet_results}/${dataset_name}/${nnunet_trainer}__nnUNetPlans__${configuration}/fold_${fold}/sct-testing -d ${dataset_num} -f ${fold} -c ${configuration} 
    CUDA_VISIBLE_DEVICES=${cuda_visible_devices} nnUNetv2_predict -i ${nnUNet_raw}/${dataset_name}/imagesTs/bavaria -tr ${nnunet_trainer} -o ${nnUNet_results}/${dataset_name}/${nnunet_trainer}__nnUNetPlans__${configuration}/fold_${fold}/bavaria -d ${dataset_num} -f ${fold} -c ${configuration}

    echo "-------------------------------------------------------"
    echo "Running ANIMA evaluation on Test set"
    echo "-------------------------------------------------------"
    python ~/ms_lesion_agnostic/ms-lesion-agnostic/nnunet/single_class_nnunet/evaluate_lesion_seg_prediction.py --pred-folder ~/ms_lesion_agnostic/data/predictions/Dataset101_singleClassNnunetMsLesion/nnUNetTrainer__3d_fullres/fold_${fold}/test_canproco --mask-gt ~/ms_lesion_agnostic/data/nnUNet_raw/Dataset101_singleClassNnunetMsLesion/labelsTs/canproco --animaPath ~/anima/Anima-Binaries-4.2 --output-folder ~/ms_lesion_agnostic/data/predictions/Dataset101_singleClassNnunetMsLesion/nnUNetTrainer__3d_fullres/fold_1/test_canproco_metrics
    python ~/ms_lesion_agnostic/ms-lesion-agnostic/nnunet/single_class_nnunet/evaluate_lesion_seg_prediction.py --pred-folder ~/ms_lesion_agnostic/data/predictions/Dataset101_singleClassNnunetMsLesion/nnUNetTrainer__3d_fullres/fold_${fold}/test_basel --mask-gt ~/ms_lesion_agnostic/data/nnUNet_raw/Dataset101_singleClassNnunetMsLesion/labelsTs/basel --animaPath ~/anima/Anima-Binaries-4.2 --output-folder ~/ms_lesion_agnostic/data/predictions/Dataset101_singleClassNnunetMsLesion/nnUNetTrainer__3d_fullres/fold_1/test_basel_metrics
    python ~/ms_lesion_agnostic/ms-lesion-agnostic/nnunet/single_class_nnunet/evaluate_lesion_seg_prediction.py --pred-folder ~/ms_lesion_agnostic/data/predictions/Dataset101_singleClassNnunetMsLesion/nnUNetTrainer__3d_fullres/fold_${fold}/sct-testing --mask-gt ~/ms_lesion_agnostic/data/nnUNet_raw/Dataset101_singleClassNnunetMsLesion/labelsTs/sct-testing --animaPath ~/anima/Anima-Binaries-4.2 --output-folder ~/ms_lesion_agnostic/data/predictions/Dataset101_singleClassNnunetMsLesion/nnUNetTrainer__3d_fullres/fold_1/test_sct-testing_metrics
    python ~/ms_lesion_agnostic/ms-lesion-agnostic/nnunet/single_class_nnunet/evaluate_lesion_seg_prediction.py --pred-folder ~/ms_lesion_agnostic/data/predictions/Dataset101_singleClassNnunetMsLesion/nnUNetTrainer__3d_fullres/fold_${fold}/bavaria --mask-gt ~/ms_lesion_agnostic/data/nnUNet_raw/Dataset101_singleClassNnunetMsLesion/labelsTs/bavaria --animaPath ~/anima/Anima-Binaries-4.2 --output-folder ~/ms_lesion_agnostic/data/predictions/Dataset101_singleClassNnunetMsLesion/nnUNetTrainer__3d_fullres/fold_1/test_bavaria_metrics



    # python training/02_compute_anima_metrics.py --pred-folder ${nnUNet_results}/${dataset_name}/${nnunet_trainer}__nnUNetPlans__${configuration}/fold_${fold}/test_${site} --gt-folder ${nnUNet_raw}/${dataset_name}/labelsTs_${site} --dataset-name ${site} ${region_based}

    done

done