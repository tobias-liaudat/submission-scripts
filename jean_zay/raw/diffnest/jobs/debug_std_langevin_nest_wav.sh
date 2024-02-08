#!/bin/bash
#SBATCH --job-name=debug_job    # nom du job
#SBATCH --mail-user=tobiasliaudat@gmail.com
#SBATCH --mail-type=ALL
##SBATCH --partition=gpu_p2          # de-commente pour la partition gpu_p2
#SBATCH --ntasks=1                   # nombre total de tache MPI (= nombre total de GPU)
#SBATCH --ntasks-per-node=1          # nombre de tache MPI par noeud (= nombre de GPU par noeud)
#SBATCH --gres=gpu:1                 # nombre de GPU par n?~Sud (max 8 avec gpu_p2)
#SBATCH --cpus-per-task=10           # nombre de coeurs CPU par tache (un quart du noeud ici)
##SBATCH -C v100-32g 
# /!\ Attention, "multithread" fait reference a l'hyperthreading dans la terminologie Slurm
#SBATCH --hint=nomultithread         # hyperthreading desactive
#SBATCH --time=00:20:00              # temps d'execution maximum demande (HH:MM:SS)
#SBATCH --output=%x_%A_%a.out  # nom du fichier de sortie
#SBATCH --error=%x_%A_%a.err   # nom du fichier d'erreur (ici commun avec la sortie)
#SBATCH -A ynx@v100                   # specify the project
#SBATCH --qos=qos_gpu-dev            # using the dev queue, as this is only for profiling
##SBATCH --array=0-4

cd $WORK/projects/submission-scripts/jean_zay/env_configs/

. diffnest.sh

# echo des commandes lancees
set -x

cd $WORK/projects/projects-ns/proxnest_diff/ProxNest

srun python -u ./run_nested_sampling.py \
    --slurm True \
    --config $WORK/projects/submission-scripts/jean_zay/raw/diffnest/configs/debug_std_langevin_nest_wav.yml
