#!/bin/bash
#SBATCH --job-name=eqb_campaign64
#SBATCH --mail-user=tobiasliaudat@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH -C v100-32g
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --qos=qos_gpu-t3
#SBATCH --output=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/submission-scripts/jean_zay/raw/equivariant_bootstrap/jobs/logs/R-%x_%A_%a.out
#SBATCH --error=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/submission-scripts/jean_zay/raw/equivariant_bootstrap/jobs/logs/R-%x_%A_%a.err
#SBATCH -A rbn@v100

# ===========================================================================
# The 64^2 screen: both reconstructors, the full ladder, the kappa sweep, M4's
# controls and the per-family singletons. Decides which family and which kappa
# go to 360^2. 34 arms x 3 seeds x 2 reconstructors = 204 rows.
#
# Self-chaining: resubmits itself with --dependency=afterany BEFORE starting
# work, and --skip-existing makes the successor resume rather than repeat. Safe
# to leave for a week.
#
#   sbatch campaign_64.sh
#
# To stop early:  touch \$OUTPUT_ROOT/campaign64/.COMPLETE  and scancel the
# pending successor.
#
# SIZING: the unrolled net is ~45x slower per draw than artifact_removal at
# 64^2. If prepare_cluster.sh projects more than ~2 days, cut `seeds` to [1]
# for the unrolled family in the config rather than crowding out the 360^2 jobs.
# ===========================================================================

export GPU_ARCH=v100
SUBMISSION_REPO=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/submission-scripts
JOB_SCRIPT=$SUBMISSION_REPO/jean_zay/raw/equivariant_bootstrap/jobs/campaign_64.sh
. $SUBMISSION_REPO/jean_zay/raw/equivariant_bootstrap/jobs/_common.sh

CONFIG=$CAMPAIGN_DIR/configs/campaign_64.yaml
OUT_DIR=$OUTPUT_ROOT/campaign64

chain_or_stop "$OUT_DIR" "$JOB_SCRIPT"

set -x
srun python -u scripts/uq_campaign.py \
    --config "$CONFIG" \
    --out-dir "$OUT_DIR" \
    --device cuda \
    --skip-existing
set +x

mark_complete "$OUT_DIR"
