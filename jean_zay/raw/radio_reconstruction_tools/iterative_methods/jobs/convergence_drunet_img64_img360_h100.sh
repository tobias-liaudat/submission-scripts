#!/bin/bash
#SBATCH --job-name=rrt_conv_diag_drunet
#SBATCH --mail-user=tobiasliaudat@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24           # coeurs CPU par tache (un quart d'un noeud H100: 96/4)
#SBATCH -C h100                      # partition H100 (gpu_p6)
#SBATCH --hint=nomultithread
#SBATCH --time=01:30:00              # small: a few images x a short iteration sweep
#SBATCH --qos=qos_gpu_h100-dev       # dev queue H100 (<= 2h, scheduling rapide)
#SBATCH --output=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/submission-scripts/jean_zay/raw/radio_reconstruction_tools/iterative_methods/jobs/logs/R-%x_%A_%a.out
#SBATCH --error=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/submission-scripts/jean_zay/raw/radio_reconstruction_tools/iterative_methods/jobs/logs/R-%x_%A_%a.err
#SBATCH -A ney@h100                  # allocation H100 (radio reconstruction, for now)

# Small convergence diagnostic for the DRUNet PnP iterative prior, on BOTH the
# img64 and img360 conditions. Runs the solver once per image with early-stop
# off and a generous ceiling, records per-iteration PSNR + residual, and prints
# an approximate --max-iter (quality plateau / residual convergence) + a plot.
# Use the printed suggestion to set --max-iter for run_iterative.py.
# PREREQUISITE: pretrained DRUNet cached on a login node (scripts/prefetch_drunet.py).

set -x

CODE_REPO=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/radio_reconstruction_tools
SUBMISSION_REPO=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/submission-scripts
CONFIG_DIR=$SUBMISSION_REPO/jean_zay/raw/radio_reconstruction_tools/large/configs
CONFIG_64=$CONFIG_DIR/large_run_img64_unrolled_n15_prior_DRUNet_PSF_finufft_4gpus.yaml
CONFIG_360=$CONFIG_DIR/large_run_img360_unrolled_n15_prior_DRUNet_PSF_finufft_4gpus.yaml
OUT_DIR=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/outputs/iterative_convergence

# --- Environment (H100: arch/h100 pre-module before pytorch) ---------------
. $SUBMISSION_REPO/jean_zay/env_configs/radio_reconstruction_tools_h100.sh
export TMPDIR=$JOBSCRATCH
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_DUPLICATE_LIB_OK=TRUE

mkdir -p "$OUT_DIR"
cd $CODE_REPO

# img64 (fast): a longer sweep is cheap here.
srun python -u scripts/iterative_methods/convergence_diagnostic.py \
    --config "$CONFIG_64" \
    --prior DRUNet --algo PGD \
    --n-images 3 --max-iter 400 --isnr 20 \
    --output "$OUT_DIR/convergence_img64_PGD_DRUNet.png" \
    --device cuda

srun python -u scripts/iterative_methods/convergence_diagnostic.py \
    --config "$CONFIG_64" \
    --prior DnCNN_lipschitz --algo PGD \
    --n-images 3 --max-iter 400 --isnr 20 \
    --output "$OUT_DIR/convergence_img64_PGD_DnCNN_lipschitz.png" \
    --device cuda

srun python -u scripts/iterative_methods/convergence_diagnostic.py \
    --config "$CONFIG_64" \
    --prior wavPrior --algo PGD \
    --n-images 3 --max-iter 400 --isnr 20 \
    --output "$OUT_DIR/convergence_img64_PGD_wavPrior.png" \
    --device cuda   


# img360 (heavier per iteration): a slightly shorter ceiling keeps it quick.
srun python -u scripts/iterative_methods/convergence_diagnostic.py \
    --config "$CONFIG_360" \
    --prior DRUNet --algo PGD \
    --n-images 3 --max-iter 400 --isnr 20 \
    --output "$OUT_DIR/convergence_img360_PGD_DRUNet.png" \
    --device cuda

srun python -u scripts/iterative_methods/convergence_diagnostic.py \
    --config "$CONFIG_360" \
    --prior DnCNN_lipschitz --algo PGD \
    --n-images 3 --max-iter 400 --isnr 20 \
    --output "$OUT_DIR/convergence_img360_PGD_DnCNN_lipschitz.png" \
    --device cuda

srun python -u scripts/iterative_methods/convergence_diagnostic.py \
    --config "$CONFIG_360" \
    --prior wavPrior --algo PGD \
    --n-images 3 --max-iter 400 --isnr 20 \
    --output "$OUT_DIR/convergence_img360_PGD_wavPrior.png" \
    --device cuda    

echo "[conv] done. Plots + suggested max_iter in $OUT_DIR (see also the job stdout above)."
