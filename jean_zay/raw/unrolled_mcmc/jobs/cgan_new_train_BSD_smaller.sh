#!/bin/bash
#SBATCH --job-name=cgan_new_train_GaussDeblur_BSD_smaller    # nom du job
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
#SBATCH -A rbn@v100                   # specify the project
##SBATCH --qos=qos_gpu-dev            # using the dev queue, as this is only for profiling
##SBATCH --qos=qos_gpu-t4              # Long queue
##SBATCH --array=0-6

# echo des commandes lancees

cd /lustre/fswork/projects/rech/rbn/ulx23va/projects/unrolled_cGAN/repos/unrolled_cgan/

. /lustre/fswork/projects/rech/rbn/ulx23va/projects/unrolled_cGAN/repos/submission-scripts/jean_zay/env_configs/unrolled_mcmc.sh

set -x

srun python -u unrolled_cgan.py \
    --config /lustre/fswork/projects/rech/rbn/ulx23va/projects/unrolled_cGAN/repos/submission-scripts/jean_zay/raw/unrolled_mcmc/configs/args_new_cGAN_BSD_fast.yml

