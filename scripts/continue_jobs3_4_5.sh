#!/bin/bash
#SBATCH --account=aip-jcohen
#SBATCH --job-name=job3x4x5     # set a more descriptive job-name 
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=36
#SBATCH --mem=120G
#SBATCH --time=1-00:00:00   # DD-HH:MM:SS
#SBATCH --output=/home/p/plb/links/scratch/ms-lesion-agnostic/model_trainings/%x_%A_v2.out
#SBATCH --error=/home/p/plb/links/scratch/ms-lesion-agnostic/model_trainings/%x_%A_v2.err
#SBATCH --mail-user=pierrelouis.benveniste03@gmail.com     # whenever the job starts/fails/completes, an email will be sent 
#SBATCH --mail-type=ALL

# Launch with parallel 
parallel --jobs 3 ::: \
  "(ts=\$(date '+%Y-%m-%d-%H-%M-%S'); ./job3/continue_train_job3.sh 2>&1 | tee ./job3/logfile_\$ts.txt)" \
  "(ts=\$(date '+%Y-%m-%d-%H-%M-%S'); ./job4/train_script_job4.sh 2>&1 | tee ./job4/logfile_\$ts.txt)" \
  "(ts=\$(date '+%Y-%m-%d-%H-%M-%S'); ./job5/train_script_job5.sh 2>&1 | tee ./job5/logfile_\$ts.txt)"