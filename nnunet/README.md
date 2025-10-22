# nnUNet model training and evaluation

In this folder we have the scripts needed to convert the data to the nnUNet format, train the models, evaluate them and plot the performances. 
To do so: 

1. Create the virtual environment and install the required libraries
```console
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r nnunet/requirements.txt
pip install --upgrade git+https://github.com/FabianIsensee/hiddenlayer.git
```

2. Convert the dataset to the nnUNet format:
```console
python nnunet/convert_msd_to_nnunet.py --input MSD_PATH -o OUTPUT_PATH --tasknumber XXX
```
Or with re-orientation (recommended):
```console
python nnunet/convert_msd_to_nnunet_reorient.py --input MSD_PATH -o OUTPUT_PATH --tasknumber XXX
```

3. Preprocess the data:
```console
export nnUNet_raw="PATH_RAW"
export nnUNet_results="PATH_RESULTS"
export nnUNet_preprocessed="PATH_PREPROCESSED"
nnUNetv2_plan_and_preprocess -d XXX --verify_dataset_integrity -c 3d_fullres -pl -pl nnUNetPlannerResEncL
```
The flag `-pl nnUNetPlannerResEncL` can be removed if you do not wish to use the ResEnc planer. 

4. Train the model (we used the nnUNetTrainerDiceCELoss_noSmooth_2000epochs trainer for the training)
```console
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train XXX 3d_fullres FOLD -p nnUNetResEncUNetLPlans -tr-tr nnUNetTrainerDiceCELoss_noSmooth_2000epochs
```
Again, the flag `-pl nnUNetPlannerResEncL` can be removed.

5. Perform predictions
```console
CUDA_VISIBLE_DEVICES=1 nnUNetv2_predict -i PATH_imagesTs -o OUTPUT_PATH -d XXX -c 3d_fullres -f FOLD -chk checkpoint_best.pth -p nnUNetResEncUNetLPlans -tr nnUNetTrainerDiceCELoss_noSmooth_2000epochs
```

6. Evaluate predictions and plot the results:
```console
python evaluation/evaluate_predictions.py -pred-folder PRED_PATH -label-folder PATH_labelsTs  -image-folder PATH_imagesTs -conversion-dict PATH_conversion_dict.json -output-folder PATH_OUTPUT
python evaluation/plot_performance.py --pred-dir-path PRED_PATH  --data-json-path MSD_PATH --split test
```

## Trainer class

The `trainer_class.py` contains the nnUNet trainer used for training the model associated to the release https://github.com/ivadomed/ms-lesion-agnostic/releases/tag/r20250909
It is also stored in the model_fold0.zip so that SCT can use it when performing inference with this specific model. 