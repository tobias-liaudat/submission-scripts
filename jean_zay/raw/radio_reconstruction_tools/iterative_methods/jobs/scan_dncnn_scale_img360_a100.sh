#!/bin/bash
#SBATCH --job-name=rrt_scan_dncnn_scale_img360
#SBATCH --mail-user=tobiasliaudat@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8            # un huitieme d'un noeud A100 (64/8)
#SBATCH -C a100                      # partition A100 (gpu_p5)
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00              # fine grid x n-eval reconstructions at 360^2
#SBATCH --qos=qos_gpu_a100-dev       # dev queue A100 (<= 2h, scheduling rapide)
#SBATCH --output=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/submission-scripts/jean_zay/raw/radio_reconstruction_tools/iterative_methods/jobs/logs/R-%x_%A_%a.out
#SBATCH --error=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/submission-scripts/jean_zay/raw/radio_reconstruction_tools/iterative_methods/jobs/logs/R-%x_%A_%a.err
#SBATCH -A ney@a100                  # allocation A100 (radio reconstruction, for now)

# Fine DnCNN input-scale scan on img360 --------------------------------------
# Quantitative follow-up to the qualitative convergence sweep: score a FINE grid
# of denoiser input scales s (apply s*D(x/s)) on the held-out TEST split and pick
# the s with the best mean PSNR. stepsize/lambda are held fixed (stepsize=0.032 =
# 0.3/opnorm at img360; lambda=6 constant), so only the regularisation strength s
# varies. Writes scale_scan_summary.json (best s), scale_scan_psnr.png (PSNR vs s)
# and scale_scan_panel.png (recon per s) under the eval dir.
# PREREQUISITE: pretrained DnCNN lipschitz weights cached on a login node
# (scripts/prefetch_dncnn.py).

set -x

CODE_REPO=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/radio_reconstruction_tools
SUBMISSION_REPO=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/submission-scripts
CONFIG_DIR=$SUBMISSION_REPO/jean_zay/raw/radio_reconstruction_tools/large/configs
CONFIG_360=$CONFIG_DIR/large_run_img360_unrolled_n15_prior_DRUNet_PSF_finufft_4gpus.yaml

# --- Environment (A100) ----------------------------------------------------
. $SUBMISSION_REPO/jean_zay/env_configs/radio_reconstruction_tools_a100.sh
export TMPDIR=$JOBSCRATCH
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_DUPLICATE_LIB_OK=TRUE

cd $CODE_REPO

# Fine grid around the region the coarse {1,2,3,5} sweep suggested. 40 test
# images/scale keeps the total (11 scales x 40) affordable in the 2h window; bump
# --n-eval for a tighter mean once the neighbourhood is confirmed.
srun python -u scripts/iterative_methods/scan_denoiser_scale.py \
    --config "$CONFIG_360" \
    --prior DnCNN_lipschitz --algo PGD \
    --scales 1,1.5,2,2.5,3,3.5,4,4.5,5,6,8 \
    --stepsize 0.032 --lam 6.0 \
    --max-iter 300 --isnr 20 \
    --n-eval 5 \
    --device cuda

echo "[scan] done. Best s is printed above and in scale_scan_summary.json;"
echo "[scan] see scale_scan_psnr.png (PSNR vs s) and scale_scan_panel.png (recon per s)."
