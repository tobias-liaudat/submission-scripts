#!/bin/bash
#SBATCH --job-name=nestdiff_NS_job    # nom du job
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
#SBATCH --time=100:00:00              # temps d'execution maximum demande (HH:MM:SS)
#SBATCH --output=R-%x_%A_%a.out  # nom du fichier de sortie
#SBATCH --error=R-%x_%A_%a.err   # nom du fichier d'erreur (ici commun avec la sortie)
#SBATCH -A ynx@gpu                   # specify the project
##SBATCH --qos=qos_gpu-dev            # using the dev queue, as this is only for profiling
#SBATCH --qos=qos_gpu-t4              # Long queue
#SBATCH --array=0-2

cd $WORK/projects/submission-scripts/jean_zay/env_configs/

. diffnest.sh

# echo des commandes lancees
set -x

opt[0]="--config $WORK/projects/submission-scripts/jean_zay/raw/diffnest/configs/nest_diff_run1_samples_2e3_1e4_warm_coeff_2.yml"
opt[1]="--config $WORK/projects/submission-scripts/jean_zay/raw/diffnest/configs/nest_diff_run1_samples_2e3_1e4_warm_coeff_5.yml"
opt[2]="--config $WORK/projects/submission-scripts/jean_zay/raw/diffnest/configs/nest_diff_run1_samples_2e3_1e4_warm_coeff_10.yml"

cd $WORK/projects/projects-ns/proxnest_diff/nestdiff

srun python -u ./run_nested_sampling.py \
    --slurm True \
    ${opt[$SLURM_ARRAY_TASK_ID]}
