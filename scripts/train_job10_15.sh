#!/bin/bash
#SBATCH --account=aip-jcohen
#SBATCH --job-name=job10x15     # set a more descriptive job-name 
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=24
#SBATCH --mem=80G
#SBATCH --time=1-00:00:00   # DD-HH:MM:SS
#SBATCH --output=/home/p/plb/links/scratch/ms-lesion-agnostic/model_trainings/%x_%A_v2.out
#SBATCH --error=/home/p/plb/links/scratch/ms-lesion-agnostic/model_trainings/%x_%A_v2.err
#SBATCH --mail-user=pierrelouis.benveniste03@gmail.com     # whenever the job starts/fails/completes, an email will be sent 
#SBATCH --mail-type=ALL

# Launch jobs
parallel --verbose --jobs 2 ::: \
  "(ts=\$(date '+%Y-%m-%d-%H-%M-%S'); bash /home/p/plb/links/scratch/ms-lesion-agnostic/model_trainings/scripts/train_script_job10.sh 2>&1 | tee /home/p/plb/links/scratch/ms-lesion-agnostic/model_trainings/job6x7x8x9x10/logfile_job10_\$ts.txt)" \
  "(ts=\$(date '+%Y-%m-%d-%H-%M-%S'); bash /home/p/plb/links/scratch/ms-lesion-agnostic/model_trainings/scripts/train_script_job15.sh 2>&1 | tee /home/p/plb/links/scratch/ms-lesion-agnostic/model_trainings/job11x12x13x14/logfile_job15_\$ts.txt)"
