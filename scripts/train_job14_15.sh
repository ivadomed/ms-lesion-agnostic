#!/bin/bash
#SBATCH --account=aip-jcohen
#SBATCH --job-name=job14x15     # set a more descriptive job-name 
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=24
#SBATCH --mem=80G
#SBATCH --time=1-00:00:00   # DD-HH:MM:SS
#SBATCH --output=/home/p/plb/links/scratch/ms-lesion-agnostic/model_trainings/%x_%A_v2.out
#SBATCH --error=/home/p/plb/links/scratch/ms-lesion-agnostic/model_trainings/%x_%A_v2.err
#SBATCH --mail-user=pierrelouis.benveniste03@gmail.com     # whenever the job starts/fails/completes, an email will be sent 
#SBATCH --mail-type=ALL

# Launch job14
bash /home/p/plb/links/scratch/ms-lesion-agnostic/model_trainings/job14/train_script_job14.sh  | tee /home/p/plb/links/scratch/ms-lesion-agnostic/model_trainings/job14/logfile_$(date '+%Y-%m-%d-%H-%M-%S').txt &

# Launch job15
bash /home/p/plb/links/scratch/ms-lesion-agnostic/model_trainings/job15/train_script_job15.sh  | tee /home/p/plb/links/scratch/ms-lesion-agnostic/model_trainings/job15/logfile_$(date '+%Y-%m-%d-%H-%M-%S').txt &

wait