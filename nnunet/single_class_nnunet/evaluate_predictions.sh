#!/bin/bash

# This filewas created to test the evaluation of performances. 

echo "-------------------------------------------------------"
echo "Running ANIMA evaluation on Test set"
echo "-------------------------------------------------------"

python ~/ms_lesion_agnostic/ms-lesion-agnostic/nnunet/single_class_nnunet/evaluate_lesion_seg_prediction.py --pred-folder ~/ms_lesion_agnostic/data/predictions/Dataset101_singleClassNnunetMsLesion/nnUNetTrainer__3d_fullres/fold_1/test --mask-gt ~/ms_lesion_agnostic/data/nnUNet_raw/Dataset101_singleClassNnunetMsLesion/labelsTs --animaPath ~/anima/Anima-Binaries-4.2 --output-folder ~/ms_lesion_agnostic/data/predictions/Dataset101_singleClassNnunetMsLesion/nnUNetTrainer__3d_fullres/fold_1/test_anima_metrics
