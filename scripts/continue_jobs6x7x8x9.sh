#!/bin/bash
#SBATCH --account=aip-jcohen
#SBATCH --job-name=job6x7x8x9     # set a more descriptive job-name 
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=160G
#SBATCH --time=1-00:00:00   # DD-HH:MM:SS
#SBATCH --output=/home/p/plb/links/scratch/ms-lesion-agnostic/model_trainings/job6x7x8x9x10/%x_%A_v2.out
#SBATCH --error=/home/p/plb/links/scratch/ms-lesion-agnostic/model_trainings/job6x7x8x9x10/%x_%A_v2.err
#SBATCH --mail-user=pierrelouis.benveniste03@gmail.com     # whenever the job starts/fails/completes, an email will be sent 
#SBATCH --mail-type=ALL

# Launch jobs
parallel --verbose --jobs 4 ::: \
  "(ts=\$(date '+%Y-%m-%d-%H-%M-%S'); bash /home/p/plb/links/scratch/ms-lesion-agnostic/model_trainings/scripts/continue_train_job6.sh 2>&1 | tee /home/p/plb/links/scratch/ms-lesion-agnostic/model_trainings/job6x7x8x9x10/logfile_job6_\$ts.txt)" \
  "(ts=\$(date '+%Y-%m-%d-%H-%M-%S'); bash /home/p/plb/links/scratch/ms-lesion-agnostic/model_trainings/scripts/continue_train_job7.sh 2>&1 | tee /home/p/plb/links/scratch/ms-lesion-agnostic/model_trainings/job6x7x8x9x10/logfile_job7_\$ts.txt)" \
  "(ts=\$(date '+%Y-%m-%d-%H-%M-%S'); bash /home/p/plb/links/scratch/ms-lesion-agnostic/model_trainings/scripts/continue_train_job8.sh 2>&1 | tee /home/p/plb/links/scratch/ms-lesion-agnostic/model_trainings/job6x7x8x9x10/logfile_job8_\$ts.txt)" \
  "(ts=\$(date '+%Y-%m-%d-%H-%M-%S'); bash /home/p/plb/links/scratch/ms-lesion-agnostic/model_trainings/scripts/continue_train_job9.sh 2>&1 | tee /home/p/plb/links/scratch/ms-lesion-agnostic/model_trainings/job6x7x8x9x10/logfile_job9_\$ts.txt)"
