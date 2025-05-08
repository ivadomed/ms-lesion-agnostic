#!/bin/bash
#SBATCH --account=aip-jcohen
#SBATCH --job-name=job3x4x5     # set a more descriptive job-name 
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=36
#SBATCH --mem=80G
#SBATCH --time=1-00:00:00   # DD-HH:MM:SS
#SBATCH --output=/home/p/plb/links/scratch/ms-lesion-agnostic/model_trainings/%x_%A_v2.out
#SBATCH --error=/home/p/plb/links/scratch/ms-lesion-agnostic/model_trainings/%x_%A_v2.err
#SBATCH --mail-user=pierrelouis.benveniste03@gmail.com     # whenever the job starts/fails/completes, an email will be sent 
#SBATCH --mail-type=ALL

# Launch job3
bash /home/p/plb/links/scratch/ms-lesion-agnostic/model_trainings/job3/continue_train_job3.sh

# Launch job4
bash /home/p/plb/links/scratch/ms-lesion-agnostic/model_trainings/job4/continue_train_job4.sh

# Launch job5
bash /home/p/plb/links/scratch/ms-lesion-agnostic/model_trainings/job5/continue_train_job5.sh