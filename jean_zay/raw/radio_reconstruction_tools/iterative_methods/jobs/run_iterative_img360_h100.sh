#!/bin/bash
#SBATCH --job-name=rrt_iterative_img360
#SBATCH --mail-user=tobiasliaudat@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24           # coeurs CPU par tache (un quart d'un noeud H100: 96/4)
#SBATCH -C h100                      # partition H100 (gpu_p6)
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --qos=qos_gpu_h100-dev       # dev queue H100 (<= 2h, scheduling rapide)
#SBATCH --output=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/submission-scripts/jean_zay/raw/radio_reconstruction_tools/iterative_methods/jobs/logs/R-%x_%A_%a.out
#SBATCH --error=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/submission-scripts/jean_zay/raw/radio_reconstruction_tools/iterative_methods/jobs/logs/R-%x_%A_%a.err
#SBATCH -A ney@h100                  # allocation H100 (radio reconstruction, for now)
#SBATCH --array=0-2                   # one task per prior (wavPrior / DRUNet / DnCNN_lipschitz)

# Training-free ITERATIVE reconstruction baseline on the img360 held-out TEST
# split, for comparison with the trained img360 unrolled / artifact-removal
# models. An array job runs the three priors; each reconstructs with a PGD
# solver on the SAME pinned condition (same test images + UV coverages + Briggs
# operator) as the img360 unrolled_psf run, and writes an iterative_summary.json
# + reconstruction_panel.png (same format as validate_unrolled.py).
# PREREQUISITE: pretrained weights cached on a login node — DRUNet via
# scripts/prefetch_drunet.py, DnCNN via scripts/prefetch_dncnn.py (compute nodes
# are offline); wavPrior needs none.
# NOTE: iterative reconstruction is per-image and slow at 360^2 (NUFFT x
# iterations x n_eval). --n-eval is capped at 100 to fit the 2h dev queue; raise
# it (and/or switch to qos_gpu_h100-t3, 20h) for a fuller test-set estimate.
# Outputs land under <log_dir>/iterative_eval/<config_stem>__<prior>_PGD/.

set -x

CODE_REPO=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/radio_reconstruction_tools
SUBMISSION_REPO=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/submission-scripts
# Condition taken from the img360 unrolled_psf training config — identical to
# what the trained img360 models used.
CONFIG=$SUBMISSION_REPO/jean_zay/raw/radio_reconstruction_tools/large/configs/large_run_img360_unrolled_n15_prior_DRUNet_PSF_finufft_4gpus.yaml

# One prior per array task.
PRIORS=(wavPrior DRUNet DnCNN_lipschitz)
PRIOR=${PRIORS[$SLURM_ARRAY_TASK_ID]}

# --- Environment (H100: arch/h100 pre-module before pytorch) ---------------
. $SUBMISSION_REPO/jean_zay/env_configs/radio_reconstruction_tools_h100.sh
export TMPDIR=$JOBSCRATCH
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_DUPLICATE_LIB_OK=TRUE

cd $CODE_REPO

# --isnr matches the trained-model validations; --max-iter omitted -> each
# prior's tuned default. --n-eval 100 keeps a task within the 2h dev queue.
srun python -u scripts/iterative_methods/run_iterative.py \
    --config "$CONFIG" \
    --prior "$PRIOR" \
    --algo PGD \
    --isnr 20 \
    --n-eval 100 \
    --n-panel 5 \
    --device cuda
