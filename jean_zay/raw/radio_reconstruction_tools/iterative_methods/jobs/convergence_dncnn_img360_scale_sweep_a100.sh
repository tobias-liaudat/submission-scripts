#!/bin/bash
#SBATCH --job-name=rrt_conv_dncnn_img360_scalesweep
#SBATCH --mail-user=tobiasliaudat@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8            # un huitieme d'un noeud A100 (64/8)
#SBATCH -C a100                      # partition A100 (gpu_p5)
#SBATCH --hint=nomultithread
#SBATCH --time=01:30:00              # small: a few images x a short scale sweep
#SBATCH --qos=qos_gpu_a100-dev       # dev queue A100 (<= 2h, scheduling rapide)
#SBATCH --output=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/submission-scripts/jean_zay/raw/radio_reconstruction_tools/iterative_methods/jobs/logs/R-%x_%A_%a.out
#SBATCH --error=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/submission-scripts/jean_zay/raw/radio_reconstruction_tools/iterative_methods/jobs/logs/R-%x_%A_%a.err
#SBATCH -A ney@a100                  # allocation A100 (radio reconstruction, for now)

# DnCNN img360 DENOISER-SCALE sweep ---------------------------------------------
# Diagnosis (confirmed from the deepinv source): DnCNN(download_lipschitz) IGNORES
# its sigma/g_param argument -- it is a FIXED-strength denoiser trained at
# sigma=2/255 on [0,1] images. That fixed strength is ~right for img64 but
# UNDER-regularises the far more ill-posed img360 problem (same visibility count,
# ~32x more pixels), leaving dirty-beam sidelobe artifacts that NO lambda removes
# (lambda only rebalances data vs prior; the prior itself is capped). wavPrior is
# fine at both sizes because its threshold auto-scales (~0.016 -> ~0.32).
#
# The lever for a fixed-strength denoiser is INPUT SCALING: apply s*D(x/s), which
# makes DnCNN remove ~s x more noise relative to the signal while keeping the
# operating range on [0,1]. This job sweeps s and turns on --log-denoiser-input
# to ALSO print the denoiser-input min/max/mean/std (to directly check the
# normalisation hypothesis: is the iterate actually in the trained [0,1] range?).
# lambda is fixed at 6 (constant, opnorm-independent -> data step 0.3*6=1.8, near
# the PGD stability limit at both sizes; see the earlier lambda analysis).
# PREREQUISITE: pretrained DnCNN lipschitz weights cached on a login node
# (scripts/prefetch_dncnn.py).

set -x

CODE_REPO=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/radio_reconstruction_tools
SUBMISSION_REPO=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/submission-scripts
CONFIG_DIR=$SUBMISSION_REPO/jean_zay/raw/radio_reconstruction_tools/large/configs
CONFIG_360=$CONFIG_DIR/large_run_img360_unrolled_n15_prior_DRUNet_PSF_finufft_4gpus.yaml
OUT_DIR=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/outputs/iterative_convergence

# --- Environment (A100) ----------------------------------------------------
. $SUBMISSION_REPO/jean_zay/env_configs/radio_reconstruction_tools_a100.sh
export TMPDIR=$JOBSCRATCH
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_DUPLICATE_LIB_OK=TRUE

mkdir -p "$OUT_DIR"
cd $CODE_REPO

# stepsize = 0.3 / opnorm, opnorm ~ 9.43 at img360 -> 0.032 (fixed).
# lambda = 6 (constant). Sweep the denoiser input scale s.
#   s = 1  -> current (fixed weak) strength; expect the artifacts we already see.
#   s = 2,3,5 -> progressively stronger effective regularisation.
# --log-denoiser-input prints the input scale on the first calls of each run.
STEPSIZE=0.032
LAM=6.0
for SCALE in 1.0 2.0 3.0 5.0; do
    srun python -u scripts/iterative_methods/convergence_diagnostic.py \
        --config "$CONFIG_360" \
        --prior DnCNN_lipschitz --algo PGD \
        --n-images 3 --max-iter 500 --isnr 20 \
        --stepsize "$STEPSIZE" --lam "$LAM" \
        --denoiser-scale "$SCALE" --log-denoiser-input \
        --output "$OUT_DIR/convergence_img360_PGD_DnCNN_scale${SCALE}.png" \
        --device cuda
done

echo "[conv] done. Compare $OUT_DIR/convergence_img360_PGD_DnCNN_scale*.png (curve + panel)."
echo "[conv] check the printed [denoiser-input] lines: if min/max are far outside [0,1]"
echo "[conv] the problem is a scale/normalisation mismatch; if they are ~[0,1] then it is"
echo "[conv] pure under-regularisation and the best s>1 is the fix."
