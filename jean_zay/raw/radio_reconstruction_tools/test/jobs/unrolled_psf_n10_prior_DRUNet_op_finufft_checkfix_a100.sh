#!/bin/bash
#SBATCH --job-name=rrt_checkfix_unrolled_psf_DRUNet_finufft_a100
#SBATCH --mail-user=tobiasliaudat@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8            # un huitieme d'un noeud A100 (64/8)
#SBATCH -C a100                      # partition A100 (gpu_p5)
#SBATCH --hint=nomultithread
#SBATCH --time=01:00:00              # tiny train + validation
#SBATCH --qos=qos_gpu_a100-dev       # dev queue A100 (<= 2h, scheduling rapide)
#SBATCH --output=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/submission-scripts/jean_zay/raw/radio_reconstruction_tools/test/jobs/logs/R-%x_%A_%a.out
#SBATCH --error=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/submission-scripts/jean_zay/raw/radio_reconstruction_tools/test/jobs/logs/R-%x_%A_%a.err
#SBATCH -A ney@a100                  # allocation A100 (radio reconstruction, for now)

# Checkerboard-fix end-to-end verification: trains the tiny PSF/DRUNet model
# (now loading the PRETRAINED denoiser) THEN validates it and renders the
# reconstruction panel in one job. PREREQUISITE: run scripts/prefetch_drunet.py
# once on a login node so the pretrained DRUNet is cached (compute nodes are
# offline). Expect the training log to print `[DRUNet] loading denoiser
# weights: 'download' (frozen)` and the panel to be checkerboard-free.

set -x

CODE_REPO=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/radio_reconstruction_tools
SUBMISSION_REPO=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/submission-scripts
CONFIG=$SUBMISSION_REPO/jean_zay/raw/radio_reconstruction_tools/test/configs/unrolled_psf_n10_prior_DRUNet_op_finufft_checkfix.yaml
OUTPUT_ROOT=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/outputs/trained_models
RUN_NAME=unrolled_psf_n10_prior_DRUNet_op_finufft_checkfix_a100

# --- Environment (A100: arch/a100 pre-module before pytorch) ---------------
. $SUBMISSION_REPO/jean_zay/env_configs/radio_reconstruction_tools_a100.sh
export TMPDIR=$JOBSCRATCH
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_DUPLICATE_LIB_OK=TRUE
export WANDB_MODE=offline

mkdir -p "$OUTPUT_ROOT"
cd $CODE_REPO

# 1. Train (sequential mints $OUTPUT_ROOT/${RUN_NAME}_<timestamp>/).
srun python -u scripts/train_unrolled_cluster.py \
    --config $CONFIG \
    --device cuda \
    --log-dir $OUTPUT_ROOT \
    --run-name $RUN_NAME

# 2. Resolve the just-created run dir (newest one holding a resolved_config.yaml).
RUN_DIR=""
for d in $(ls -dt "$OUTPUT_ROOT/${RUN_NAME}"_* 2>/dev/null); do
    if [ -f "$d/resolved_config.yaml" ]; then RUN_DIR="$d"; break; fi
done
if [ -z "$RUN_DIR" ]; then
    echo "[checkfix] ERROR: could not find the trained run dir to validate."
    exit 1
fi
echo "[checkfix] validating RUN_DIR=$RUN_DIR"

# 3. Validate + render the reconstruction panel (checkerboard should be gone).
srun python -u scripts/validation/validate_unrolled.py \
    --run-dir "$RUN_DIR" \
    --n-eval 8 \
    --n-panel 5 \
    --isnr 20 \
    --device cuda

echo "[checkfix] done. Inspect $RUN_DIR/validation/reconstruction_panel.png"
