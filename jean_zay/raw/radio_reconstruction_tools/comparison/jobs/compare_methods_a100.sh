#!/bin/bash
#SBATCH --job-name=rrt_compare_methods
#SBATCH --mail-user=tobiasliaudat@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8            # un huitieme d'un noeud A100 (64/8)
#SBATCH -C a100                      # partition A100 (gpu_p5)
#SBATCH --hint=nomultithread
#SBATCH --time=18:00:00
#SBATCH --qos=qos_gpu_a100-t3        # queue longue (<= 20h): l'evaluation iterative domine
#SBATCH --output=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/submission-scripts/jean_zay/raw/radio_reconstruction_tools/comparison/jobs/logs/R-%x_%A_%a.out
#SBATCH --error=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/submission-scripts/jean_zay/raw/radio_reconstruction_tools/comparison/jobs/logs/R-%x_%A_%a.err
#SBATCH -A ney@a100                  # allocation A100 (radio reconstruction, for now)

# FULL cross-method comparison on the held-out TEST split, both image sizes, all
# 500 test images per method. Evaluates every trained run under OUTPUT_ROOT
# (artifact_removal, unrolled/tkbn, unrolled_psf/finufft — the older duplicate of
# a repeated config is dropped automatically by --dedupe latest) plus the
# training-free iterative baselines: PGD+wavPrior / DnCNN_lipschitz / DRUNet at
# 64, PGD+wavPrior at 360.
#
# Writes comparison_report.md (one results table per image size), one
# comparison_panel_img<size>.png over the first three TEST images (ground truth,
# u-v coverage, dirty image, one reconstruction per method with its PSNR) and
# comparison_summary.json to $OUT_DIR.
#
# Budget (A100): the trained nets are cheap (~1.5 min per run for 500 images on
# H100, cf. the validation jobs); the per-image iterative solvers dominate —
# roughly 3-4 h for the three img64 priors and ~1-1.5 h for img360 wavPrior on
# H100, so allow ~1.5-2x that here. Results are cached per method under
# $OUT_DIR/cache, so relaunching after a timeout resumes instead of recomputing
# (pass --refresh to force a recompute).
#
# PREREQUISITE: compute nodes are offline, so the pretrained plug-and-play
# weights must already be in the torch-hub cache — run, once, on a login node:
#   python scripts/prefetch_drunet.py    # DRUNet (also used as the unrolled prox)
#   python scripts/prefetch_dncnn.py     # DnCNN "download_lipschitz"
# A prior whose weights are missing is reported as failed and the rest of the
# table is still produced.

set -x

CODE_REPO=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/radio_reconstruction_tools
SUBMISSION_REPO=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/submission-scripts
OUTPUT_ROOT=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/outputs
OUT_DIR=$OUTPUT_ROOT/comparison

# --- Environment (A100: arch/a100 pre-module before pytorch) ---------------
. $SUBMISSION_REPO/jean_zay/env_configs/radio_reconstruction_tools_a100.sh
export TMPDIR=$JOBSCRATCH
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_DUPLICATE_LIB_OK=TRUE

cd $CODE_REPO

# --isnr 20 / --n-eval 500 match the recorded per-run validations, so the trained
# rows reproduce their validation_summary.json numbers.
# --iter-max-iter: the values tuned with convergence_diagnostic.py and already
# used by iterative_methods/jobs/run_iterative_img64_h100.sh.
srun python -u scripts/comparison/compare_methods.py \
    --models-root "$OUTPUT_ROOT/trained_models" \
    --output-dir "$OUT_DIR" \
    --img-sizes 64 360 \
    --isnr 20 \
    --n-eval 500 \
    --n-panel 3 \
    --iter-algo PGD \
    --iter-priors 64=wavPrior,DnCNN_lipschitz,DRUNet \
    --iter-priors 360=wavPrior \
    --iter-max-iter wavPrior=150 \
    --iter-max-iter DRUNet=150 \
    --iter-max-iter DnCNN_lipschitz=500 \
    --device cuda
