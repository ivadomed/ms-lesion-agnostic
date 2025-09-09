#!/bin/bash
#SBATCH --account=aip-jcohen
#SBATCH --job-name=job3x4x5     # set a more descriptive job-name 
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=120G
#SBATCH --time=1-00:00:00   # DD-HH:MM:SS
#SBATCH --output=/home/p/plb/links/scratch/ms-lesion-agnostic/model_trainings/job3/%x_%A_v2.out
#SBATCH --error=/home/p/plb/links/scratch/ms-lesion-agnostic/model_trainings/job3/%x_%A_v2.err
#SBATCH --mail-user=pierrelouis.benveniste03@gmail.com     # whenever the job starts/fails/completes, an email will be sent 
#SBATCH --mail-type=ALL

# Launch with parallel 
parallel --jobs 3 ::: \
  "(ts=\$(date '+%Y-%m-%d-%H-%M-%S'); /home/p/plb/links/scratch/ms-lesion-agnostic/model_trainings/scripts/continue_train_job3.sh 2>&1 | tee /home/p/plb/links/scratch/ms-lesion-agnostic/model_trainings/job3/logfile_job3_\$ts.txt)" \
  "(ts=\$(date '+%Y-%m-%d-%H-%M-%S'); /home/p/plb/links/scratch/ms-lesion-agnostic/model_trainings/scripts/continue_train_job4.sh 2>&1 | tee /home/p/plb/links/scratch/ms-lesion-agnostic/model_trainings/job3/logfile_job4_\$ts.txt)" \
  "(ts=\$(date '+%Y-%m-%d-%H-%M-%S'); /home/p/plb/links/scratch/ms-lesion-agnostic/model_trainings/scripts/continue_train_job5.sh 2>&1 | tee /home/p/plb/links/scratch/ms-lesion-agnostic/model_trainings/job3/logfile_job5_\$ts.txt)"