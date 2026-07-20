#!/bin/bash
#SBATCH --job-name=rrt_large_img64_artremoval_4gpu
#SBATCH --mail-user=tobiasliaudat@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --ntasks=4                   # 1 tache MPI par GPU (DDP: doit = runtime.devices)
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4                 # un noeud H100 complet (4 GPU/noeud)
#SBATCH --cpus-per-task=24           # coeurs CPU par tache (un quart de noeud H100: 96/4)
#SBATCH -C h100                      # partition H100 (gpu_p6)
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00              # single UNet at 64^2 converges well within one link
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --output=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/submission-scripts/jean_zay/raw/radio_reconstruction_tools/large/jobs/logs/R-%x_%A_%a.out
#SBATCH --error=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/submission-scripts/jean_zay/raw/radio_reconstruction_tools/large/jobs/logs/R-%x_%A_%a.err
#SBATCH -A ney@h100                  # allocation H100 (radio reconstruction, for now)

# ===========================================================================
# Non-unrolled artifact-removal (UNet) BASELINE, img64, 4-GPU DDP on H100.
# Single feed-forward network (deepinv ArtifactRemoval) -> no greedy schedule,
# so this is a plain single job (not self-chaining). It reuses the img64
# unrolled_psf pinned condition (artifacts_dir in the config), so the comparison
# is on identical coverages / images.
# PREREQUISITE: the shared artifacts dir must already exist (built by the img64
# large-run builder job / submit_large_runs.sh).
# Validate + compare afterwards with scripts/validation/validate_unrolled.py
# (it reads model_type=artifact_removal from the run's resolved_config).
# Launch:  cd .../large/jobs/logs && sbatch ../large_run_img64_artifact_removal_unet_finufft_4gpus_h100.sh
# ===========================================================================

set -uo pipefail

CODE_REPO=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/radio_reconstruction_tools
SUBMISSION_REPO=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/submission-scripts
CONFIG=$SUBMISSION_REPO/jean_zay/raw/radio_reconstruction_tools/large/configs/large_run_img64_artifact_removal_unet_finufft_4gpus.yaml
OUTPUT_ROOT=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/outputs/trained_models
RUN_NAME=large_run_img64_artifact_removal_unet_finufft_4gpus_$(date +%Y%m%d_%H%M%S)

# --- Environment (H100: arch/h100 pre-module before pytorch) ---------------
. $SUBMISSION_REPO/jean_zay/env_configs/radio_reconstruction_tools_h100.sh
export TMPDIR=$JOBSCRATCH
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_DUPLICATE_LIB_OK=TRUE
export WANDB_MODE=offline

set -x
cd $CODE_REPO
srun python -u scripts/train_unrolled_cluster.py \
    --config "$CONFIG" \
    --device cuda \
    --log-dir "$OUTPUT_ROOT" \
    --run-name "$RUN_NAME"
