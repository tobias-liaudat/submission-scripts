#!/bin/bash
#SBATCH --job-name=rrt_compare_methods_fast
#SBATCH --mail-user=tobiasliaudat@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8            # un huitieme d'un noeud A100 (64/8)
#SBATCH -C a100                      # partition A100 (gpu_p5)
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --qos=qos_gpu_a100-dev       # dev queue A100 (<= 2h, scheduling rapide)
#SBATCH --output=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/submission-scripts/jean_zay/raw/radio_reconstruction_tools/comparison/jobs/logs/R-%x_%A_%a.out
#SBATCH --error=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/submission-scripts/jean_zay/raw/radio_reconstruction_tools/comparison/jobs/logs/R-%x_%A_%a.err
#SBATCH -A ney@a100                  # allocation A100 (radio reconstruction, for now)

# QUICK cross-method comparison sized for the 2h dev queue: same methods and same
# outputs as compare_methods_a100.sh, but the (slow, per-image) iterative
# baselines only score --fast-n-eval test images instead of the full 500. The
# trained models still score all 500 (a few minutes per run), so their rows stay
# directly comparable with the recorded validation_summary.json numbers; the
# iterative rows are noisier and carry their smaller N in the table.
#
# Use this to sanity-check the pipeline / get a first look; use
# compare_methods_a100.sh for the reference table.
#
# Writes to a SEPARATE output dir so it never races with (or overwrites) the full
# comparison. The two runs also keep separate per-method caches.
#
# PREREQUISITE (compute nodes are offline): prefetch the plug-and-play weights on
# a login node — python scripts/prefetch_drunet.py and scripts/prefetch_dncnn.py.

set -x

CODE_REPO=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/radio_reconstruction_tools
SUBMISSION_REPO=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/submission-scripts
OUTPUT_ROOT=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/outputs
OUT_DIR=$OUTPUT_ROOT/comparison_fast

# --- Environment (A100: arch/a100 pre-module before pytorch) ---------------
. $SUBMISSION_REPO/jean_zay/env_configs/radio_reconstruction_tools_a100.sh
export TMPDIR=$JOBSCRATCH
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_DUPLICATE_LIB_OK=TRUE

cd $CODE_REPO

srun python -u scripts/comparison/compare_methods.py \
    --models-root "$OUTPUT_ROOT/trained_models" \
    --output-dir "$OUT_DIR" \
    --img-sizes 64 360 \
    --isnr 20 \
    --n-eval 500 \
    --fast-iterative \
    --fast-n-eval 50 \
    --n-panel 3 \
    --iter-algo PGD \
    --iter-priors 64=wavPrior,DnCNN_lipschitz,DRUNet \
    --iter-priors 360=wavPrior \
    --iter-max-iter wavPrior=150 \
    --iter-max-iter DRUNet=150 \
    --iter-max-iter DnCNN_lipschitz=500 \
    --device cuda
