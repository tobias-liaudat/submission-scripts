#!/bin/bash
#SBATCH --job-name=PR_gaussian_benchmark    # nom du job
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
#SBATCH --output=PR_gaussian_benchmark-%x_%A_%a.out  # nom du fichier de sortie
#SBATCH --error=PR_gaussian_benchmark-%x_%A_%a.err   # nom du fichier d'erreur (ici commun avec la sortie)
#SBATCH -A rbn@v100                   # specify the project
##SBATCH --qos=qos_gpu-dev            # using the dev queue, as this is only for profiling
##SBATCH --array=0-4

cd /lustre/fswork/projects/rech/rbn/ulx23va/projects/nested_sampling/repos/submission-scripts/jean_zay/env_configs/

. diffnest.sh

# echo des commandes lancees
set -x

cd /lustre/fswork/projects/rech/rbn/ulx23va/projects/nested_sampling/repos/PR_proxnest_diff/proxnest_diff/examples/experiments/

srun python -u ./gaussian_benchmark.py -s 