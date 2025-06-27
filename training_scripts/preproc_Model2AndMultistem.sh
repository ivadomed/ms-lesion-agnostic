#!/bin/bash
#SBATCH --account=aip-jcohen
#SBATCH --job-name=preprocModel2Multistem     # set a more descriptive job-name 
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=200G
#SBATCH --time=1-00:00:00   # DD-HH:MM:SS
#SBATCH --output=/home/p/plb/links/projects/aip-jcohen/plb/final_trainings/%x_%A_v2.out
#SBATCH --error=/home/p/plb/links/projects/aip-jcohen/plb/final_trainings/%x_%A_v2.err
#SBATCH --mail-user=pierrelouis.benveniste03@gmail.com     # whenever the job starts/fails/completes, an email will be sent 
#SBATCH --mail-type=ALL

# Launch jobs
parallel --verbose --jobs 2 ::: \
  "(ts=\$(date '+%Y-%m-%d-%H-%M-%S'); bash /home/p/plb/links/projects/aip-jcohen/plb/final_trainings/training_scripts/preproc_jobMultistem.sh 2>&1 | tee /home/p/plb/links/projects/aip-jcohen/plb/final_trainings/logfile_preprocModel2_\$ts.txt)" \
  "(ts=\$(date '+%Y-%m-%d-%H-%M-%S'); bash /home/p/plb/links/projects/aip-jcohen/plb/final_trainings/training_scripts/preproc_jobModel2.sh 2>&1 | tee /home/p/plb/links/projects/aip-jcohen/plb/final_trainings/logfile_preprocMultistem_\$ts.txt)" \