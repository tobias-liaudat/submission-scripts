#!/bin/bash
#SBATCH --job-name=PR_val_wfv2_test_v3_    # nom du job
##SBATCH --partition=gpu_p2          # de-commente pour la partition gpu_p2
#SBATCH --ntasks=1                   # nombre total de tache MPI (= nombre total de GPU)
#SBATCH --ntasks-per-node=1          # nombre de tache MPI par noeud (= nombre de GPU par noeud)
#SBATCH --gres=gpu:1                 # nombre de GPU par n?~Sud (max 8 avec gpu_p2)
#SBATCH --cpus-per-task=10           # nombre de coeurs CPU par tache (un quart du noeud ici)
#SBATCH -C v100-32g 
# /!\ Attention, "multithread" fait reference a l'hyperthreading dans la terminologie Slurm
#SBATCH --hint=nomultithread         # hyperthreading desactive
#SBATCH --time=20:00:00              # temps d'execution maximum demande (HH:MM:SS)
#SBATCH --output=PR_val_wfv2_test_v3_%j.out  # nom du fichier de sortie
#SBATCH --error=PR_val_wfv2_test_v3_%j.err   # nom du fichier d'erreur (ici commun avec la sortie)
#SBATCH --mail-use=tobiasliaudat@gmail.com
#SBATCH --mail-type=ALL
#SBATCH -A ynx@v100                   # specify the project
##SBATCH --qos=qos_gpu-dev            # using the dev queue, as this is only for profiling
##SBATCH --array=0-3

# echo des commandes lancees
set -x

# Load env
cd $WORK/projects/submission-scripts/jean_zay/env_configs/
. wfv2.sh

sleep 240

wavediff -c /gpfswork/rech/ynx/ulx23va/projects/submission-scripts/jean_zay/raw/wfv2/configs/phase-retrieval-PR/PR_configs_v3.yaml \
         -r /gpfswork/rech/ynx/ulx23va/wfv2/repos/phase-retrieval-PR/wf-psf \
         -o /gpfsscratch/rech/ynx/ulx23va/wfv2/phase-retrieval-PR/output_v2/PR_validation

