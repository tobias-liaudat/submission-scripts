#!/bin/bash
#SBATCH --job-name=rrt_validate_large_img64_psf_finufft
#SBATCH --mail-user=tobiasliaudat@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24           # coeurs CPU par tache (un quart d'un noeud H100: 96/4)
#SBATCH -C h100                      # partition H100 (gpu_p6)
#SBATCH --hint=nomultithread
#SBATCH --time=01:30:00              # validation is quick (single GPU, ~few min)
#SBATCH --qos=qos_gpu_h100-dev       # dev queue H100 (<= 2h, scheduling rapide)
#SBATCH --output=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/submission-scripts/jean_zay/raw/radio_reconstruction_tools/validation/jobs/logs/R-%x_%A_%a.out
#SBATCH --error=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/submission-scripts/jean_zay/raw/radio_reconstruction_tools/validation/jobs/logs/R-%x_%A_%a.err
#SBATCH -A ney@h100                  # allocation H100 (radio reconstruction, for now)

# Validate the finished img64 PSF/finufft large run: plots the training metrics,
# computes reconstruction PSNR on the held-out TEST split, and renders the
# qualitative panel. Outputs land in <RUN_DIR>/validation/. Point RUN_DIR at a
# different finished run to validate it (the script reads its resolved_config).

set -x

CODE_REPO=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/radio_reconstruction_tools
SUBMISSION_REPO=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/submission-scripts
RUN_DIR=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/outputs/trained_models/large_run_img64_unrolled_n15_prior_DRUNet_PSF_finufft_4gpus_20260717_190728_20260717_190854

# --- Environment (H100: arch/h100 pre-module before pytorch) ---------------
. $SUBMISSION_REPO/jean_zay/env_configs/radio_reconstruction_tools_h100.sh
export TMPDIR=$JOBSCRATCH
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_DUPLICATE_LIB_OK=TRUE

cd $CODE_REPO

# Evaluate the full 500-image test split at a fixed 30 dB input SNR.
srun python -u scripts/validation/validate_unrolled.py \
    --run-dir "$RUN_DIR" \
    --n-eval 500 \
    --n-panel 5 \
    --isnr 20 \
    --device cuda
