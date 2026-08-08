#!/bin/bash
#SBATCH --job-name=eqb_campaign360_ar
#SBATCH --mail-user=tobiasliaudat@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -C a100
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --qos=qos_gpu_a100-t3
#SBATCH --output=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/submission-scripts/jean_zay/raw/equivariant_bootstrap/jobs/logs/R-%x_%A_%a.out
#SBATCH --error=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/submission-scripts/jean_zay/raw/equivariant_bootstrap/jobs/logs/R-%x_%A_%a.err
#SBATCH -A rbn@a100

# ===========================================================================
# 360^2 confirmation, artifact-removal only. Ladder arms, no kappa sweep.
#
# Separate from the unrolled job because the two differ by ~22x per forward
# pass at this size (1.27 s against 28.1 s, measured on CPU) -- sharing one
# script would size the wall-time for the wrong reconstructor.
#
#   sbatch campaign_360_ar.sh
#
# UNVERIFIED: -C a100 / qos_gpu_a100-t3 / -A rbn@a100 were written without
# cluster access. Run `idrproj` to confirm the rbn a100 allocation exists; if
# it does not, switch to -C v100-32g / qos_gpu-t3 / -A rbn@v100 and expect it
# to be slower.
# ===========================================================================

export GPU_ARCH=a100
SUBMISSION_REPO=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/submission-scripts
JOB_SCRIPT=$SUBMISSION_REPO/jean_zay/raw/equivariant_bootstrap/jobs/campaign_360_ar.sh
. $SUBMISSION_REPO/jean_zay/raw/equivariant_bootstrap/jobs/_common.sh

CONFIG=$CAMPAIGN_DIR/configs/campaign_360.yaml
OUT_DIR=$OUTPUT_ROOT/campaign360
STATE_DIR=$OUT_DIR/.state_ar

chain_or_stop "$STATE_DIR" "$JOB_SCRIPT"

set -x
srun python -u scripts/uq_campaign.py \
    --config "$CONFIG" \
    --out-dir "$OUT_DIR" \
    --device cuda \
    --skip-existing \
    --rows $(rows_for_family "$CONFIG" artifact_removal)
set +x

mark_complete "$STATE_DIR"
