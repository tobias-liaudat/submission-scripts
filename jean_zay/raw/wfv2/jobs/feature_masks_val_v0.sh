#!/bin/bash
#SBATCH --job-name=std_feature_mask_v0_    # nom du job
##SBATCH --partition=gpu_p2          # de-commente pour la partition gpu_p2
#SBATCH --ntasks=1                   # nombre total de tache MPI (= nombre total de GPU)
#SBATCH --ntasks-per-node=1          # nombre de tache MPI par noeud (= nombre de GPU par noeud)
#SBATCH --gres=gpu:1                 # nombre de GPU par n?~Sud (max 8 avec gpu_p2)
#SBATCH --cpus-per-task=10           # nombre de coeurs CPU par tache (un quart du noeud ici)
#SBATCH -C v100-32g 
# /!\ Attention, "multithread" fait reference a l'hyperthreading dans la terminologie Slurm
#SBATCH --hint=nomultithread         # hyperthreading desactive
#SBATCH --time=20:00:00              # temps d'execution maximum demande (HH:MM:SS)
#SBATCH --output=std_feature_mask_v0_%j.out  # nom du fichier de sortie
#SBATCH --error=std_feature_mask_v0_%j.err   # nom du fichier d'erreur (ici commun avec la sortie)
#SBATCH --mail-use=tobiasliaudat@gmail.com
#SBATCH --mail-type=ALL
#SBATCH -A ynx@v100                   # specify the project
##SBATCH --qos=qos_gpu-dev            # using the dev queue, as this is only for profiling
##SBATCH --array=0-3


# Load env
cd /lustre/fswork/projects/rech/ynx/ulx23va/wfv2/repos/feature_mask/submission-scripts/jean_zay/env_configs/
. wfv2.sh

sleep 60

# echo des commandes lancees
set -x

wavediff -c /lustre/fswork/projects/rech/ynx/ulx23va/wfv2/repos/feature_mask/submission-scripts/jean_zay/raw/wfv2/configs/feature-masks/configs.yaml \
         -r /lustre/fswork/projects/rech/ynx/ulx23va/wfv2/repos/feature_mask/wf-psf \
         -o /lustre/fswork/projects/rech/ynx/ulx23va/wfv2/repos/feature_mask/outputs

