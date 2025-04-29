# Creation of the virtual environment

console```
python3 -m venv .venv_job1
source .venv_job1/bin/activate
```

# Installing libraries required

```console
pip install torch torchvision torchaudio
```

# In my fork of nnUNet
```console
pip install -e .
```

# Make the script executable
```console
chmod u+x bash_script.sh
```
And run it
```console
bash_script.sh
``` 

# On compute canada
We can test scripts before launching them, using salloc.
```console
salloc --time=1:0:0 --mem-per-cpu=3G --ntasks=1 --account=aip-jcohen
```
# Other:
```console
export nnUNet_raw="/home/p/plb/links/projects/aip-jcohen/plb/nnUNet_experiments/nnUNet_raw"
export nnUNet_preprocessed="/home/p/plb/links/scratch/ms-lesion-agnostic/model_trainings/job1/nnUNet_preprocessed"
export nnUNet_results="/home/p/plb/links/scratch/ms-lesion-agnostic/model_trainings/job1/nnUNet_results"
```
```console
scp train_script.sh  plb@tamia.alliancecan.ca:~/links/scratch/ms-lesion-agnostic/model_trainings/job1/
```
