#!/bin/bash
#SBATCH --account=aip-jcohen
#SBATCH --job-name=job6x7x8x9     # set a more descriptive job-name 
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=80G
#SBATCH --time=1-00:00:00   # DD-HH:MM:SS
#SBATCH --output=/home/p/plb/links/scratch/ms-lesion-agnostic/model_trainings/%x_%A_v2.out
#SBATCH --error=/home/p/plb/links/scratch/ms-lesion-agnostic/model_trainings/%x_%A_v2.err
#SBATCH --mail-user=pierrelouis.benveniste03@gmail.com     # whenever the job starts/fails/completes, an email will be sent 
#SBATCH --mail-type=ALL

# Launch job6
bash /home/p/plb/links/scratch/ms-lesion-agnostic/model_trainings/job6/train_script_job6.sh

# Launch job7
bash /home/p/plb/links/scratch/ms-lesion-agnostic/model_trainings/job7/train_script_job7.sh

# Launch job8
bash /home/p/plb/links/scratch/ms-lesion-agnostic/model_trainings/job8/train_script_job8.sh

# Launch job9
bash /home/p/plb/links/scratch/ms-lesion-agnostic/model_trainings/job9/train_script_job9.sh