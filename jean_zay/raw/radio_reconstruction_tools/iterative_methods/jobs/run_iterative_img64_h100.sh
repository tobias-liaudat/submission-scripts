#!/bin/bash
#SBATCH --job-name=rrt_iterative_img64
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

# Training-free ITERATIVE reconstruction baseline on the img64 held-out TEST
# split, for comparison with the trained unrolled / artifact-removal models. An
# array job runs the three priors; each reconstructs with a PGD solver on the
# SAME pinned condition (same test images + UV coverages + Briggs operator) as
# the img64 unrolled_psf run, and writes an iterative_summary.json +
# reconstruction_panel.png (same format as validate_unrolled.py).
# PREREQUISITE: the pretrained DRUNet weights must be cached (run
# scripts/prefetch_drunet.py once on a login node); wavPrior/DnCNN need no
# prefetch (DnCNN downloads its own lipschitz weights on first use — do that on
# a login node too if the compute node is offline).
# Outputs land under <log_dir>/iterative_eval/<config_stem>__<prior>_PGD/.

set -x

CODE_REPO=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/radio_reconstruction_tools
SUBMISSION_REPO=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/submission-scripts
# Condition (test split / UV bank / operator / weights) taken from the img64
# unrolled_psf training config — identical to what the trained models used.
CONFIG=$SUBMISSION_REPO/jean_zay/raw/radio_reconstruction_tools/large/configs/large_run_img64_unrolled_n15_prior_DRUNet_PSF_finufft_4gpus.yaml

# One prior per array task.
PRIORS=(wavPrior DRUNet DnCNN_lipschitz)
PRIOR=${PRIORS[$SLURM_ARRAY_TASK_ID]}

# Set the maximum iteration number per model
MAX_ITERATIONS=(150 150 500)
MAX_ITERATION=${MAX_ITERATIONS[$SLURM_ARRAY_TASK_ID]}

# --- Environment (H100: arch/h100 pre-module before pytorch) ---------------
. $SUBMISSION_REPO/jean_zay/env_configs/radio_reconstruction_tools_h100.sh
export TMPDIR=$JOBSCRATCH
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_DUPLICATE_LIB_OK=TRUE

cd $CODE_REPO

# --isnr / --n-eval match the trained-model validations (comparable numbers);
# --max-iter omitted -> each prior's tuned default (wavPrior 100, DRUNet/DnCNN 50).
srun python -u scripts/iterative_methods/run_iterative.py \
    --config "$CONFIG" \
    --prior "$PRIOR" \
    --algo PGD \
    --isnr 20 \
    --max-iter "$MAX_ITERATION" \
    --n-eval 500 \
    --n-panel 5 \
    --device cuda
