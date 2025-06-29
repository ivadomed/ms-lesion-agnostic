# Creation of the virtual environment

```console
python3 -m venv .venv_job1
source .venv_job1/bin/activate
```

# Installing libraries required

```console
pip3 install torch torchvision torchaudio
```

# In my fork of nnUNet
```console
pip install -e .
```

I had the following error: 
```console
torch._dynamo.exc.BackendCompilerFailed: backend='inductor' raised:
RuntimeError: Cannot find a working triton installation. Either the package is not installed or it is too old. More information on installing Triton can be found at https://github.com/openai/triton

Set TORCH_LOGS="+dynamo" and TORCHDYNAMO_VERBOSE=1 for more information

You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

Exception in thread Thread-2 (results_loop):
Traceback (most recent call last):
  File "/cvmfs/soft.computecanada.ca/gentoo/2023/x86-64-v3/usr/lib/python3.11/threading.py", line 1038, in _bootstrap_inner
Exception in thread Thread-1 (results_loop):
Traceback (most recent call last):
  File "/cvmfs/soft.computecanada.ca/gentoo/2023/x86-64-v3/usr/lib/python3.11/threading.py", line 1038, in _bootstrap_inner
``` 
I solved it by running `pip install triton`.

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
salloc --time=1:0:0 --mem-per-cpu=40G --ntasks=1 --account=aip-jcohen  --gpus-per-node=h100:4
```

# Other:
Here are some usefull commands:
```console
export nnUNet_raw="/home/p/plb/links/projects/aip-jcohen/plb/nnUNet_experiments/nnUNet_raw"
export nnUNet_preprocessed="/home/p/plb/links/scratch/ms-lesion-agnostic/model_trainings/job1/nnUNet_preprocessed"
export nnUNet_results="/home/p/plb/links/scratch/ms-lesion-agnostic/model_trainings/job1/nnUNet_results"
```
```console
scp train_script.sh  plb@tamia.alliancecan.ca:~/links/scratch/ms-lesion-agnostic/model_trainings/job1/
```
To do everything at once: 
```console
python3 -m venv .venv_job1 && source .venv_job1/bin/activate && pip3 install torch torchvision torchaudio && cd nnUNet && pip install -e . && pip install triton
```
To inspect GPU or CPU usage
```console
srun --jobid 123456 --pty watch -n 5 nvidia-smi
srun --jobid 123456 --pty htop -u plb
```
To display storage use on scratch and project:
```console
diskusage_report
```