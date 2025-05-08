#!/bin/bash
#SBATCH --account=aip-jcohen
#SBATCH --job-name=job10x11x12x13     # set a more descriptive job-name 
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=80G
#SBATCH --time=1-00:00:00   # DD-HH:MM:SS
#SBATCH --output=/home/p/plb/links/scratch/ms-lesion-agnostic/model_trainings/%x_%A_v2.out
#SBATCH --error=/home/p/plb/links/scratch/ms-lesion-agnostic/model_trainings/%x_%A_v2.err
#SBATCH --mail-user=pierrelouis.benveniste03@gmail.com     # whenever the job starts/fails/completes, an email will be sent 
#SBATCH --mail-type=ALL

# Launch job10
bash /home/p/plb/links/scratch/ms-lesion-agnostic/model_trainings/job10/train_script_job10.sh  | tee /home/p/plb/links/scratch/ms-lesion-agnostic/model_trainings/job10/logfile_$(date '+%Y-%m-%d-%H-%M-%S').txt &

# Launch job11
bash /home/p/plb/links/scratch/ms-lesion-agnostic/model_trainings/job11/train_script_job11.sh  | tee /home/p/plb/links/scratch/ms-lesion-agnostic/model_trainings/job11/logfile_$(date '+%Y-%m-%d-%H-%M-%S').txt &

# Launch job12
bash /home/p/plb/links/scratch/ms-lesion-agnostic/model_trainings/job12/train_script_job12.sh  | tee /home/p/plb/links/scratch/ms-lesion-agnostic/model_trainings/job12/logfile_$(date '+%Y-%m-%d-%H-%M-%S').txt &

# Launch job13
bash /home/p/plb/links/scratch/ms-lesion-agnostic/model_trainings/job13/train_script_job13.sh  | tee /home/p/plb/links/scratch/ms-lesion-agnostic/model_trainings/job13/logfile_$(date '+%Y-%m-%d-%H-%M-%S').txt &

wait