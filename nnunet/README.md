# nnUNet model training and evaluation

In this folder we have the scripts needed to convert the data to the nnUNet format and train the segmentation model.
For the model in release https://github.com/ivadomed/ms-lesion-agnostic/releases/tag/r20250909, the model was trained using the following steps:

1. Convert the dataset to the nnUNet format:
```console
python nnunet/convert_msd_to_nnunet.py --input MSD_PATH -o OUTPUT_PATH --tasknumber 902
```
Or with re-orientation (recommended):
```console
python nnunet/convert_msd_to_nnunet_reorient.py --input MSD_PATH -o OUTPUT_PATH --tasknumber 902
```

2. Install my own fork of nnunet
I was previously working on this fork https://github.com/plbenveniste/nnUNet/tree/plb/multistem, but now the recommended strategy is to use the Neuropoly fork: https://github.com/spinalcordtoolbox/nnUNet-neuropoly

3. Preprocess the data:
```console
export nnUNet_raw="PATH_RAW"
export nnUNet_results="PATH_RESULTS"
export nnUNet_preprocessed="PATH_PREPROCESSED"
nnUNetv2_plan_and_preprocess -d 902 -c 3d_fullres --verify_dataset_integrity
```
In this project, I was using a specific planer: [nnUNetResEncUNetL1x1x1_Model2_Plans.json](https://github.com/ivadomed/ms-lesion-agnostic/releases/download/r20250909/nnUNetResEncUNetL1x1x1_Model2_Plans.json)
```console
nnnUNetv2_preprocess -d 902 -plans_name nnUNetResEncUNetL1x1x1_Model2_Plans
```

4. Train the model (it was trained on TamIA) on 5 folds.
```console
CUDA_VISIBLE_DEVICES=2 nnUNetv2_train 902  3d_fullres 1 -p nnUNetResEncUNetL1x1x1_Model2_Plans -tr nnUNetTrainerDiceCELoss_noSmooth_4000epochs_fromScratch -pretrained_weights path/to/pretrained_models/Dataset617_nativect/MultiTalent_trainer_4000ep__nnUNetResEncUNetL1x1x1_Plans_znorm_bs24__3d_fullres/fold_all/checkpoint_final.pth
```
The pretrained models can be downloaded here: https://zenodo.org/records/13753413

## Trainer class

The `trainer_class.py` contains the nnUNet trainer used for training the model associated to the release https://github.com/ivadomed/ms-lesion-agnostic/releases/tag/r20250909
It is also stored in the model_fold0.zip so that SCT can use it when performing inference with this specific model. 