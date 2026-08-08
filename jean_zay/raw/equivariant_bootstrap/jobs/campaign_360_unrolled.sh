#!/bin/bash
#SBATCH --job-name=eqb_campaign360_unr
#SBATCH --mail-user=tobiasliaudat@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -C a100
#SBATCH --hint=nomultithread
#SBATCH --time=100:00:00
#SBATCH --qos=qos_gpu_a100-t4
#SBATCH --output=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/submission-scripts/jean_zay/raw/equivariant_bootstrap/jobs/logs/R-%x_%A_%a.out
#SBATCH --error=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/submission-scripts/jean_zay/raw/equivariant_bootstrap/jobs/logs/R-%x_%A_%a.err
#SBATCH -A rbn@a100

# ===========================================================================
# 360^2 confirmation, the PSF-unrolled network. The expensive job, and the one
# most likely to blow its wall-time -- which is why it takes the long QoS, why
# it self-chains, and why every finished row is skipped on resume.
#
#   sbatch campaign_360_unrolled.sh
#
# Read the projected hours per arm-seed from prepare_cluster.sh before trusting
# the wall-time above; it was set from laptop CPU measurements.
#
# UNVERIFIED: -C a100 / qos_gpu_a100-t4 / -A rbn@a100, as for the AR job.
# ===========================================================================

export GPU_ARCH=a100
SUBMISSION_REPO=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/submission-scripts
JOB_SCRIPT=$SUBMISSION_REPO/jean_zay/raw/equivariant_bootstrap/jobs/campaign_360_unrolled.sh
. $SUBMISSION_REPO/jean_zay/raw/equivariant_bootstrap/jobs/_common.sh

CONFIG=$CAMPAIGN_DIR/configs/campaign_360.yaml
OUT_DIR=$OUTPUT_ROOT/campaign360
STATE_DIR=$OUT_DIR/.state_unrolled

chain_or_stop "$STATE_DIR" "$JOB_SCRIPT"

set -x
srun python -u scripts/uq_campaign.py \
    --config "$CONFIG" \
    --out-dir "$OUT_DIR" \
    --device cuda \
    --skip-existing \
    --rows $(rows_for_family "$CONFIG" unrolled)
set +x

mark_complete "$STATE_DIR"
