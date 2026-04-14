#!/bin/bash
#SBATCH --job-name=training_scheduler_sbi_lisa    # nom du job
#SBATCH --mail-user=tobiasliaudat@gmail.com
#SBATCH --mail-type=ALL
##SBATCH --partition=gpu_p2          # de-commente pour la partition gpu_p2
#SBATCH --ntasks=1                   # nombre total de tache MPI (= nombre total de GPU)
#SBATCH --ntasks-per-node=1          # nombre de tache MPI par noeud (= nombre de GPU par noeud)
#SBATCH --gres=gpu:1                 # nombre de GPU par n?~Sud (max 8 avec gpu_p2)
#SBATCH --cpus-per-task=10           # nombre de coeurs CPU par tache (un quart du noeud ici)
#SBATCH -C v100-32g 
# /!\ Attention, "multithread" fait reference a l'hyperthreading dans la terminologie Slurm
#SBATCH --hint=nomultithread         # hyperthreading desactive
#SBATCH --time=20:00:00              # temps d'execution maximum demande (HH:MM:SS)
#SBATCH --output=R-%x_%A_%a.out  # nom du fichier de sortie
#SBATCH --error=R-%x_%A_%a.err   # nom du fichier d'erreur (ici commun avec la sortie)
#SBATCH -A ney@v100                   # specify the project
##SBATCH --qos=qos_gpu-dev            # using the dev queue, as this is only for profiling
##SBATCH --qos=qos_gpu-t4              # Long queue
##SBATCH --array=0-6


# Load the environment
. /lustre/fswork/projects/rech/ney/ulx23va/projects/LISA/repos/submission-scripts/jean_zay/env_configs/lisajax_sbi.sh

# echo des commandes lancees
set -x

cd /lustre/fswork/projects/rech/ney/ulx23va/projects/LISA/repos/lisajax_sbi/scripts/training/

export TMPDIR=$JOBSCRATCH

export CONFIG_FILE=/lustre/fswork/projects/rech/ney/ulx23va/projects/LISA/repos/submission-scripts/jean_zay/raw/lisajax_sbi/configs/gpu_training_500k_sim_1800_samples_tc_range_1000_fixed_masses_noise_off_sky_uniform_230226_scheduler.yaml

echo "Precompute summaries ..."
srun python -u training_large_dataset.py -c $CONFIG_FILE --precompute-summaries

echo "Train NPE for SBI..."
srun python -u training_large_dataset.py -c $CONFIG_FILE

# Compute the coverage
echo "Run validation metrics..."
srun python -u validate_posterior.py -c $CONFIG_FILE
