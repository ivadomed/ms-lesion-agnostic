#!/bin/bash
#SBATCH --account=aip-jcohen
#SBATCH --job-name=eval1to15     # set a more descriptive job-name 
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=300G
#SBATCH --time=1-00:00:00   # DD-HH:MM:SS
#SBATCH --output=/home/p/plb/links/projects/aip-jcohen/plb/eval_logs/%x_%A_v2.out
#SBATCH --error=/home/p/plb/links/projects/aip-jcohen/plb/eval_logs/%x_%A_v2.err
#SBATCH --mail-user=pierrelouis.benveniste03@gmail.com     # whenever the job starts/fails/completes, an email will be sent 
#SBATCH --mail-type=ALL

echo "Activating environment ..."
source /home/p/plb/links/scratch/ms-lesion-agnostic/model_trainings/job11x12x13x14x15/.venv_job11x12x13x14x15/bin/activate

# Launch jobs
parallel --verbose --jobs 32 ::: \
  "(ts=\$(date '+%Y-%m-%d-%H-%M-%S'); python /home/p/plb/links/projects/aip-jcohen/plb/evaluation/ms-lesion-agnostic/nnunet/evaluate_predictions.py -pred-folder /home/p/plb/links/projects/aip-jcohen/plb/results/job1/imagesTr_pred_chckFinal_fold0 -label-folder /home/p/plb/links/projects/aip-jcohen/plb/nnUNet_experiments/nnUNet_raw/Dataset902_msLesionAgnostic/labelsTr -image-folder /home/p/plb/links/projects/aip-jcohen/plb/nnUNet_experiments/nnUNet_raw/Dataset902_msLesionAgnostic/imagesTr -conversion-dict /home/p/plb/links/projects/aip-jcohen/plb/nnUNet_experiments/nnUNet_raw/Dataset902_msLesionAgnostic/conversion_dict.json -output-folder /home/p/plb/links/projects/aip-jcohen/plb/results/job1/imagesTr_pred_chckFinal_fold0_results)" \
