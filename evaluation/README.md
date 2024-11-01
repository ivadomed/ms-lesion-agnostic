# Evaluation of SCT methods

The files in the folder were used to evaluate the SCT methods on the test set and the external test sets. 
The test set is described in the MSD dataset while the external test sets are the `ms-basel-2018` and `ms-basel-2020` dataset. 

The files can be used the following way:

1. Create a virtual environment and install the required libraries

```console
conda create -n venv_eval python=3.9
conda activate venv_eval
pip install -r evaluation/requirements
```


2. Evaluate the models on the test set:

```console
python evaluation/test_sct_deepseg_lesion.py --msd-data-path MSD_PATH --output-path OUTPUT_PATH
python evaluation/test_sct_deepseg_psir_stir.py --msd-data-path MSD_PATH --output-path OUTPUT_PATH
python evaluation/test_sct_deepseg_mp2rage.py --msd-data-path MSD_PATH --output-path OUTPUT_PATH
```

3. Plot the results for each result (to be done individually for each model result):

```console
python evaluation/plot_performance.py --pred-dir-path RESULT_PATH --data-json-path MSD_DATA_PATH  --split test
```

4. Evaluate the models on the external test sets:

```console
python evaluation/test_sct_lesion_external_dataset.py --input-folder DATA_PATH --output-path OUTPUT_PATH
python evaluation/test_sct_psir-stir_external_dataset.py --input-folder DATA_PATH--output-path OUTPUT_PATH
python evaluation/test_sct_mp2rage_external_dataset.py --input-folder DATA_PATH --output-path OUTPUT_PATH
```
