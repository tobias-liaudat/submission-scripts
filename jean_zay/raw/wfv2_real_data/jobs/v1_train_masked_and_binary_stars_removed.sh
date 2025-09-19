#!/bin/bash
#SBATCH --job-name=v1_train_masked_and_binary_stars_removed_    # nom du job
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
#SBATCH --mail-use=tobiasliaudat@gmail.com
#SBATCH --mail-type=ALL
#SBATCH -A uem@v100                   # specify the project
##SBATCH --qos=qos_gpu-dev            # using the dev queue, as this is only for profiling
##SBATCH --array=0-3


# Load env
cd /lustre/fswork/projects/rech/rbn/ulx23va/projects/wf_projects/real_data_processing_conference/submission-scripts/jean_zay/env_configs/
. wfv2.sh

sleep 10

# echo des commandes lancees
set -x

wavediff -c /lustre/fswork/projects/rech/rbn/ulx23va/projects/wf_projects/real_data_processing_conference/submission-scripts/jean_zay/raw/wfv2_real_data/configs/v1_train_masked_and_binary_stars_removed/configs.yaml \
         -r /lustre/fswork/projects/rech/rbn/ulx23va/projects/wf_projects/real_data_processing_conference/wf-psf \
         -o /lustre/fswork/projects/rech/rbn/ulx23va/projects/wf_projects/real_data_processing_conference/outputs/dataset_obs2378_masked_and_binary_stars_removed/v1

