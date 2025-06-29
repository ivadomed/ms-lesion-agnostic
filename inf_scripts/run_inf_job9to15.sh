#!/bin/bash
#SBATCH --account=aip-jcohen
#SBATCH --job-name=run_inf9-15     # set a more descriptive job-name 
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=300G
#SBATCH --time=1-00:00:00   # DD-HH:MM:SS
#SBATCH --output=/home/p/plb/links/projects/aip-jcohen/plb/inf_logs/%x_%A_v2.out
#SBATCH --error=/home/p/plb/links/projects/aip-jcohen/plb/inf_logs/%x_%A_v2.err
#SBATCH --mail-user=pierrelouis.benveniste03@gmail.com     # whenever the job starts/fails/completes, an email will be sent 
#SBATCH --mail-type=ALL

# Launch jobs
parallel --verbose --jobs 28 ::: \
  "(ts=\$(date '+%Y-%m-%d-%H-%M-%S'); bash /home/p/plb/links/projects/aip-jcohen/plb/inf_scripts/inf_job9_imagesTr_chckBest.sh 2>&1 | tee /home/p/plb/links/projects/aip-jcohen/plb/inf_logs/logfile_inf_job9_imagesTr_chckBest_\$ts.txt)" \
  "(ts=\$(date '+%Y-%m-%d-%H-%M-%S'); bash /home/p/plb/links/projects/aip-jcohen/plb/inf_scripts/inf_job9_imagesTr_chckFinal.sh 2>&1 | tee /home/p/plb/links/projects/aip-jcohen/plb/inf_logs/logfile_inf_job9_imagesTr_chckFinal_\$ts.txt)" \
  "(ts=\$(date '+%Y-%m-%d-%H-%M-%S'); bash /home/p/plb/links/projects/aip-jcohen/plb/inf_scripts/inf_job9_imagesTs_chckBest.sh 2>&1 | tee /home/p/plb/links/projects/aip-jcohen/plb/inf_logs/logfile_inf_job9_imagesTs_chckBest_\$ts.txt)" \
  "(ts=\$(date '+%Y-%m-%d-%H-%M-%S'); bash /home/p/plb/links/projects/aip-jcohen/plb/inf_scripts/inf_job9_imagesTs_chckFinal.sh 2>&1 | tee /home/p/plb/links/projects/aip-jcohen/plb/inf_logs/logfile_inf_job9_imagesTs_chckFinal_\$ts.txt)" \
  "(ts=\$(date '+%Y-%m-%d-%H-%M-%S'); bash /home/p/plb/links/projects/aip-jcohen/plb/inf_scripts/inf_job10_imagesTr_chckBest.sh 2>&1 | tee /home/p/plb/links/projects/aip-jcohen/plb/inf_logs/logfile_inf_job10_imagesTr_chckBest_\$ts.txt)" \
  "(ts=\$(date '+%Y-%m-%d-%H-%M-%S'); bash /home/p/plb/links/projects/aip-jcohen/plb/inf_scripts/inf_job10_imagesTr_chckFinal.sh 2>&1 | tee /home/p/plb/links/projects/aip-jcohen/plb/inf_logs/logfile_inf_job10_imagesTr_chckFinal_\$ts.txt)" \
  "(ts=\$(date '+%Y-%m-%d-%H-%M-%S'); bash /home/p/plb/links/projects/aip-jcohen/plb/inf_scripts/inf_job10_imagesTs_chckBest.sh 2>&1 | tee /home/p/plb/links/projects/aip-jcohen/plb/inf_logs/logfile_inf_job10_imagesTs_chckBest_\$ts.txt)" \
  "(ts=\$(date '+%Y-%m-%d-%H-%M-%S'); bash /home/p/plb/links/projects/aip-jcohen/plb/inf_scripts/inf_job10_imagesTs_chckFinal.sh 2>&1 | tee /home/p/plb/links/projects/aip-jcohen/plb/inf_logs/logfile_inf_job10_imagesTs_chckFinal_\$ts.txt)" \
  "(ts=\$(date '+%Y-%m-%d-%H-%M-%S'); bash /home/p/plb/links/projects/aip-jcohen/plb/inf_scripts/inf_job11_imagesTr_chckBest.sh 2>&1 | tee /home/p/plb/links/projects/aip-jcohen/plb/inf_logs/logfile_inf_job11_imagesTr_chckBest_\$ts.txt)" \
  "(ts=\$(date '+%Y-%m-%d-%H-%M-%S'); bash /home/p/plb/links/projects/aip-jcohen/plb/inf_scripts/inf_job11_imagesTr_chckFinal.sh 2>&1 | tee /home/p/plb/links/projects/aip-jcohen/plb/inf_logs/logfile_inf_job11_imagesTr_chckFinal_\$ts.txt)" \
  "(ts=\$(date '+%Y-%m-%d-%H-%M-%S'); bash /home/p/plb/links/projects/aip-jcohen/plb/inf_scripts/inf_job11_imagesTs_chckBest.sh 2>&1 | tee /home/p/plb/links/projects/aip-jcohen/plb/inf_logs/logfile_inf_job11_imagesTs_chckBest_\$ts.txt)" \
  "(ts=\$(date '+%Y-%m-%d-%H-%M-%S'); bash /home/p/plb/links/projects/aip-jcohen/plb/inf_scripts/inf_job11_imagesTs_chckFinal.sh 2>&1 | tee /home/p/plb/links/projects/aip-jcohen/plb/inf_logs/logfile_inf_job11_imagesTs_chckFinal_\$ts.txt)" \
  "(ts=\$(date '+%Y-%m-%d-%H-%M-%S'); bash /home/p/plb/links/projects/aip-jcohen/plb/inf_scripts/inf_job12_imagesTr_chckBest.sh 2>&1 | tee /home/p/plb/links/projects/aip-jcohen/plb/inf_logs/logfile_inf_job12_imagesTr_chckBest_\$ts.txt)" \
  "(ts=\$(date '+%Y-%m-%d-%H-%M-%S'); bash /home/p/plb/links/projects/aip-jcohen/plb/inf_scripts/inf_job12_imagesTr_chckFinal.sh 2>&1 | tee /home/p/plb/links/projects/aip-jcohen/plb/inf_logs/logfile_inf_job12_imagesTr_chckFinal_\$ts.txt)" \
  "(ts=\$(date '+%Y-%m-%d-%H-%M-%S'); bash /home/p/plb/links/projects/aip-jcohen/plb/inf_scripts/inf_job12_imagesTs_chckBest.sh 2>&1 | tee /home/p/plb/links/projects/aip-jcohen/plb/inf_logs/logfile_inf_job12_imagesTs_chckBest_\$ts.txt)" \
  "(ts=\$(date '+%Y-%m-%d-%H-%M-%S'); bash /home/p/plb/links/projects/aip-jcohen/plb/inf_scripts/inf_job12_imagesTs_chckFinal.sh 2>&1 | tee /home/p/plb/links/projects/aip-jcohen/plb/inf_logs/logfile_inf_job12_imagesTs_chckFinal_\$ts.txt)" \
  "(ts=\$(date '+%Y-%m-%d-%H-%M-%S'); bash /home/p/plb/links/projects/aip-jcohen/plb/inf_scripts/inf_job13_imagesTr_chckBest.sh 2>&1 | tee /home/p/plb/links/projects/aip-jcohen/plb/inf_logs/logfile_inf_job13_imagesTr_chckBest_\$ts.txt)" \
  "(ts=\$(date '+%Y-%m-%d-%H-%M-%S'); bash /home/p/plb/links/projects/aip-jcohen/plb/inf_scripts/inf_job13_imagesTr_chckFinal.sh 2>&1 | tee /home/p/plb/links/projects/aip-jcohen/plb/inf_logs/logfile_inf_job13_imagesTr_chckFinal_\$ts.txt)" \
  "(ts=\$(date '+%Y-%m-%d-%H-%M-%S'); bash /home/p/plb/links/projects/aip-jcohen/plb/inf_scripts/inf_job13_imagesTs_chckBest.sh 2>&1 | tee /home/p/plb/links/projects/aip-jcohen/plb/inf_logs/logfile_inf_job13_imagesTs_chckBest_\$ts.txt)" \
  "(ts=\$(date '+%Y-%m-%d-%H-%M-%S'); bash /home/p/plb/links/projects/aip-jcohen/plb/inf_scripts/inf_job13_imagesTs_chckFinal.sh 2>&1 | tee /home/p/plb/links/projects/aip-jcohen/plb/inf_logs/logfile_inf_job13_imagesTs_chckFinal_\$ts.txt)" \
  "(ts=\$(date '+%Y-%m-%d-%H-%M-%S'); bash /home/p/plb/links/projects/aip-jcohen/plb/inf_scripts/inf_job14_imagesTr_chckBest.sh 2>&1 | tee /home/p/plb/links/projects/aip-jcohen/plb/inf_logs/logfile_inf_job14_imagesTr_chckBest_\$ts.txt)" \
  "(ts=\$(date '+%Y-%m-%d-%H-%M-%S'); bash /home/p/plb/links/projects/aip-jcohen/plb/inf_scripts/inf_job14_imagesTr_chckFinal.sh 2>&1 | tee /home/p/plb/links/projects/aip-jcohen/plb/inf_logs/logfile_inf_job14_imagesTr_chckFinal_\$ts.txt)" \
  "(ts=\$(date '+%Y-%m-%d-%H-%M-%S'); bash /home/p/plb/links/projects/aip-jcohen/plb/inf_scripts/inf_job14_imagesTs_chckBest.sh 2>&1 | tee /home/p/plb/links/projects/aip-jcohen/plb/inf_logs/logfile_inf_job14_imagesTs_chckBest_\$ts.txt)" \
  "(ts=\$(date '+%Y-%m-%d-%H-%M-%S'); bash /home/p/plb/links/projects/aip-jcohen/plb/inf_scripts/inf_job14_imagesTs_chckFinal.sh 2>&1 | tee /home/p/plb/links/projects/aip-jcohen/plb/inf_logs/logfile_inf_job14_imagesTs_chckFinal_\$ts.txt)" \
  "(ts=\$(date '+%Y-%m-%d-%H-%M-%S'); bash /home/p/plb/links/projects/aip-jcohen/plb/inf_scripts/inf_job15_imagesTr_chckBest.sh 2>&1 | tee /home/p/plb/links/projects/aip-jcohen/plb/inf_logs/logfile_inf_job15_imagesTr_chckBest_\$ts.txt)" \
  "(ts=\$(date '+%Y-%m-%d-%H-%M-%S'); bash /home/p/plb/links/projects/aip-jcohen/plb/inf_scripts/inf_job15_imagesTr_chckFinal.sh 2>&1 | tee /home/p/plb/links/projects/aip-jcohen/plb/inf_logs/logfile_inf_job15_imagesTr_chckFinal_\$ts.txt)" \
  "(ts=\$(date '+%Y-%m-%d-%H-%M-%S'); bash /home/p/plb/links/projects/aip-jcohen/plb/inf_scripts/inf_job15_imagesTs_chckBest.sh 2>&1 | tee /home/p/plb/links/projects/aip-jcohen/plb/inf_logs/logfile_inf_job15_imagesTs_chckBest_\$ts.txt)" \
  "(ts=\$(date '+%Y-%m-%d-%H-%M-%S'); bash /home/p/plb/links/projects/aip-jcohen/plb/inf_scripts/inf_job15_imagesTs_chckFinal.sh 2>&1 | tee /home/p/plb/links/projects/aip-jcohen/plb/inf_logs/logfile_inf_job15_imagesTs_chckFinal_\$ts.txt)" \


