#!/bin/bash
#SBATCH --account=aip-jcohen
#SBATCH --job-name=run_inf1to8     # set a more descriptive job-name 
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
parallel --verbose --jobs 32 ::: \
  "(ts=\$(date '+%Y-%m-%d-%H-%M-%S'); bash /home/p/plb/links/projects/aip-jcohen/plb/inf_scripts/inf_job1_imagesTr_chckBest.sh 2>&1 | tee /home/p/plb/links/projects/aip-jcohen/plb/inf_logs/logfile_inf_job1_imagesTr_chckBest_\$ts.txt)" \
  "(ts=\$(date '+%Y-%m-%d-%H-%M-%S'); bash /home/p/plb/links/projects/aip-jcohen/plb/inf_scripts/inf_job1_imagesTr_chckFinal.sh 2>&1 | tee /home/p/plb/links/projects/aip-jcohen/plb/inf_logs/logfile_inf_job1_imagesTr_chckFinal_\$ts.txt)" \
  "(ts=\$(date '+%Y-%m-%d-%H-%M-%S'); bash /home/p/plb/links/projects/aip-jcohen/plb/inf_scripts/inf_job1_imagesTs_chckBest.sh 2>&1 | tee /home/p/plb/links/projects/aip-jcohen/plb/inf_logs/logfile_inf_job1_imagesTs_chckBest_\$ts.txt)" \
  "(ts=\$(date '+%Y-%m-%d-%H-%M-%S'); bash /home/p/plb/links/projects/aip-jcohen/plb/inf_scripts/inf_job1_imagesTs_chckFinal.sh 2>&1 | tee /home/p/plb/links/projects/aip-jcohen/plb/inf_logs/logfile_inf_job1_imagesTs_chckFinal_\$ts.txt)" \
  "(ts=\$(date '+%Y-%m-%d-%H-%M-%S'); bash /home/p/plb/links/projects/aip-jcohen/plb/inf_scripts/inf_job2_imagesTr_chckBest.sh 2>&1 | tee /home/p/plb/links/projects/aip-jcohen/plb/inf_logs/logfile_inf_job2_imagesTr_chckBest_\$ts.txt)" \
  "(ts=\$(date '+%Y-%m-%d-%H-%M-%S'); bash /home/p/plb/links/projects/aip-jcohen/plb/inf_scripts/inf_job2_imagesTr_chckFinal.sh 2>&1 | tee /home/p/plb/links/projects/aip-jcohen/plb/inf_logs/logfile_inf_job2_imagesTr_chckFinal_\$ts.txt)" \
  "(ts=\$(date '+%Y-%m-%d-%H-%M-%S'); bash /home/p/plb/links/projects/aip-jcohen/plb/inf_scripts/inf_job2_imagesTs_chckBest.sh 2>&1 | tee /home/p/plb/links/projects/aip-jcohen/plb/inf_logs/logfile_inf_job2_imagesTs_chckBest_\$ts.txt)" \
  "(ts=\$(date '+%Y-%m-%d-%H-%M-%S'); bash /home/p/plb/links/projects/aip-jcohen/plb/inf_scripts/inf_job2_imagesTs_chckFinal.sh 2>&1 | tee /home/p/plb/links/projects/aip-jcohen/plb/inf_logs/logfile_inf_job2_imagesTs_chckFinal_\$ts.txt)" \
  "(ts=\$(date '+%Y-%m-%d-%H-%M-%S'); bash /home/p/plb/links/projects/aip-jcohen/plb/inf_scripts/inf_job3_imagesTr_chckBest.sh 2>&1 | tee /home/p/plb/links/projects/aip-jcohen/plb/inf_logs/logfile_inf_job3_imagesTr_chckBest_\$ts.txt)" \
  "(ts=\$(date '+%Y-%m-%d-%H-%M-%S'); bash /home/p/plb/links/projects/aip-jcohen/plb/inf_scripts/inf_job3_imagesTr_chckFinal.sh 2>&1 | tee /home/p/plb/links/projects/aip-jcohen/plb/inf_logs/logfile_inf_job3_imagesTr_chckFinal_\$ts.txt)" \
  "(ts=\$(date '+%Y-%m-%d-%H-%M-%S'); bash /home/p/plb/links/projects/aip-jcohen/plb/inf_scripts/inf_job3_imagesTs_chckBest.sh 2>&1 | tee /home/p/plb/links/projects/aip-jcohen/plb/inf_logs/logfile_inf_job3_imagesTs_chckBest_\$ts.txt)" \
  "(ts=\$(date '+%Y-%m-%d-%H-%M-%S'); bash /home/p/plb/links/projects/aip-jcohen/plb/inf_scripts/inf_job3_imagesTs_chckFinal.sh 2>&1 | tee /home/p/plb/links/projects/aip-jcohen/plb/inf_logs/logfile_inf_job3_imagesTs_chckFinal_\$ts.txt)" \
  "(ts=\$(date '+%Y-%m-%d-%H-%M-%S'); bash /home/p/plb/links/projects/aip-jcohen/plb/inf_scripts/inf_job4_imagesTr_chckBest.sh 2>&1 | tee /home/p/plb/links/projects/aip-jcohen/plb/inf_logs/logfile_inf_job4_imagesTr_chckBest_\$ts.txt)" \
  "(ts=\$(date '+%Y-%m-%d-%H-%M-%S'); bash /home/p/plb/links/projects/aip-jcohen/plb/inf_scripts/inf_job4_imagesTr_chckFinal.sh 2>&1 | tee /home/p/plb/links/projects/aip-jcohen/plb/inf_logs/logfile_inf_job4_imagesTr_chckFinal_\$ts.txt)" \
  "(ts=\$(date '+%Y-%m-%d-%H-%M-%S'); bash /home/p/plb/links/projects/aip-jcohen/plb/inf_scripts/inf_job4_imagesTs_chckBest.sh 2>&1 | tee /home/p/plb/links/projects/aip-jcohen/plb/inf_logs/logfile_inf_job4_imagesTs_chckBest_\$ts.txt)" \
  "(ts=\$(date '+%Y-%m-%d-%H-%M-%S'); bash /home/p/plb/links/projects/aip-jcohen/plb/inf_scripts/inf_job4_imagesTs_chckFinal.sh 2>&1 | tee /home/p/plb/links/projects/aip-jcohen/plb/inf_logs/logfile_inf_job4_imagesTs_chckFinal_\$ts.txt)" \
  "(ts=\$(date '+%Y-%m-%d-%H-%M-%S'); bash /home/p/plb/links/projects/aip-jcohen/plb/inf_scripts/inf_job5_imagesTr_chckBest.sh 2>&1 | tee /home/p/plb/links/projects/aip-jcohen/plb/inf_logs/logfile_inf_job5_imagesTr_chckBest_\$ts.txt)" \
  "(ts=\$(date '+%Y-%m-%d-%H-%M-%S'); bash /home/p/plb/links/projects/aip-jcohen/plb/inf_scripts/inf_job5_imagesTr_chckFinal.sh 2>&1 | tee /home/p/plb/links/projects/aip-jcohen/plb/inf_logs/logfile_inf_job5_imagesTr_chckFinal_\$ts.txt)" \
  "(ts=\$(date '+%Y-%m-%d-%H-%M-%S'); bash /home/p/plb/links/projects/aip-jcohen/plb/inf_scripts/inf_job5_imagesTs_chckBest.sh 2>&1 | tee /home/p/plb/links/projects/aip-jcohen/plb/inf_logs/logfile_inf_job5_imagesTs_chckBest_\$ts.txt)" \
  "(ts=\$(date '+%Y-%m-%d-%H-%M-%S'); bash /home/p/plb/links/projects/aip-jcohen/plb/inf_scripts/inf_job5_imagesTs_chckFinal.sh 2>&1 | tee /home/p/plb/links/projects/aip-jcohen/plb/inf_logs/logfile_inf_job5_imagesTs_chckFinal_\$ts.txt)" \
  "(ts=\$(date '+%Y-%m-%d-%H-%M-%S'); bash /home/p/plb/links/projects/aip-jcohen/plb/inf_scripts/inf_job6_imagesTr_chckBest.sh 2>&1 | tee /home/p/plb/links/projects/aip-jcohen/plb/inf_logs/logfile_inf_job6_imagesTr_chckBest_\$ts.txt)" \
  "(ts=\$(date '+%Y-%m-%d-%H-%M-%S'); bash /home/p/plb/links/projects/aip-jcohen/plb/inf_scripts/inf_job6_imagesTr_chckFinal.sh 2>&1 | tee /home/p/plb/links/projects/aip-jcohen/plb/inf_logs/logfile_inf_job6_imagesTr_chckFinal_\$ts.txt)" \
  "(ts=\$(date '+%Y-%m-%d-%H-%M-%S'); bash /home/p/plb/links/projects/aip-jcohen/plb/inf_scripts/inf_job6_imagesTs_chckBest.sh 2>&1 | tee /home/p/plb/links/projects/aip-jcohen/plb/inf_logs/logfile_inf_job6_imagesTs_chckBest_\$ts.txt)" \
  "(ts=\$(date '+%Y-%m-%d-%H-%M-%S'); bash /home/p/plb/links/projects/aip-jcohen/plb/inf_scripts/inf_job6_imagesTs_chckFinal.sh 2>&1 | tee /home/p/plb/links/projects/aip-jcohen/plb/inf_logs/logfile_inf_job6_imagesTs_chckFinal_\$ts.txt)" \
  "(ts=\$(date '+%Y-%m-%d-%H-%M-%S'); bash /home/p/plb/links/projects/aip-jcohen/plb/inf_scripts/inf_job7_imagesTr_chckBest.sh 2>&1 | tee /home/p/plb/links/projects/aip-jcohen/plb/inf_logs/logfile_inf_job7_imagesTr_chckBest_\$ts.txt)" \
  "(ts=\$(date '+%Y-%m-%d-%H-%M-%S'); bash /home/p/plb/links/projects/aip-jcohen/plb/inf_scripts/inf_job7_imagesTr_chckFinal.sh 2>&1 | tee /home/p/plb/links/projects/aip-jcohen/plb/inf_logs/logfile_inf_job7_imagesTr_chckFinal_\$ts.txt)" \
  "(ts=\$(date '+%Y-%m-%d-%H-%M-%S'); bash /home/p/plb/links/projects/aip-jcohen/plb/inf_scripts/inf_job7_imagesTs_chckBest.sh 2>&1 | tee /home/p/plb/links/projects/aip-jcohen/plb/inf_logs/logfile_inf_job7_imagesTs_chckBest_\$ts.txt)" \
  "(ts=\$(date '+%Y-%m-%d-%H-%M-%S'); bash /home/p/plb/links/projects/aip-jcohen/plb/inf_scripts/inf_job7_imagesTs_chckFinal.sh 2>&1 | tee /home/p/plb/links/projects/aip-jcohen/plb/inf_logs/logfile_inf_job7_imagesTs_chckFinal_\$ts.txt)" \
  "(ts=\$(date '+%Y-%m-%d-%H-%M-%S'); bash /home/p/plb/links/projects/aip-jcohen/plb/inf_scripts/inf_job8_imagesTr_chckBest.sh 2>&1 | tee /home/p/plb/links/projects/aip-jcohen/plb/inf_logs/logfile_inf_job8_imagesTr_chckBest_\$ts.txt)" \
  "(ts=\$(date '+%Y-%m-%d-%H-%M-%S'); bash /home/p/plb/links/projects/aip-jcohen/plb/inf_scripts/inf_job8_imagesTr_chckFinal.sh 2>&1 | tee /home/p/plb/links/projects/aip-jcohen/plb/inf_logs/logfile_inf_job8_imagesTr_chckFinal_\$ts.txt)" \
  "(ts=\$(date '+%Y-%m-%d-%H-%M-%S'); bash /home/p/plb/links/projects/aip-jcohen/plb/inf_scripts/inf_job8_imagesTs_chckBest.sh 2>&1 | tee /home/p/plb/links/projects/aip-jcohen/plb/inf_logs/logfile_inf_job8_imagesTs_chckBest_\$ts.txt)" \
  "(ts=\$(date '+%Y-%m-%d-%H-%M-%S'); bash /home/p/plb/links/projects/aip-jcohen/plb/inf_scripts/inf_job8_imagesTs_chckFinal.sh 2>&1 | tee /home/p/plb/links/projects/aip-jcohen/plb/inf_logs/logfile_inf_job8_imagesTs_chckFinal_\$ts.txt)" \


