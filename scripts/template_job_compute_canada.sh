#!/bin/bash
#SBATCH --account=def-jcohen
#SBATCH --job-name=job1     # set a more descriptive job-name 
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --time=20-00:00:00   # DD-HH:MM:SS
#SBATCH --output=/home/<your-user-name>/outs/%x_%A_v2.out #  %x is the job name, %A is the job array's master job allocation number.
#SBATCH --error=/home/<your-user-name>/errs/%x_%A_v2.err
#SBATCH --mail-user=<your-email-id>     # whenever the job starts/fails/completes, an email will be sent 
#SBATCH --mail-type=ALL

# Echo time and hostname into log
echo "Date:     $(date)"
echo "Hostname: $(hostname)"

# load the required modules
echo "Loading modules ..."
module load python/3.10.13 cuda/12.2    # TODO: might differ depending on the python and cuda version you have

# activate environment
echo "Activating environment ..."
source /home/$(whoami)/envs/venv_nnunet/bin/activate        # TODO: update to match the name of your environment

# Run the model
bash <path/to/run_nnunet_compute_canada/script>