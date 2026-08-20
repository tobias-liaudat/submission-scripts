#!/bin/bash
#SBATCH --job-name=eqb_cov64
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
# Raw bootstrap coverage at 64^2, no conformalisation. 19 arms x 3 seeds = 57
# rows, MC=128, 100 images -> 729,600 image-draws.
#
# MC came down from 256 on the ladder's evidence (configs/coverage_64_gpu.yaml
# `defaults.MC`), which halved this run. The rate is measured, not bracketed:
# 0.25 s per image-draw on a V100 for this problem, so ~51 h of GPU time. That is
# three 20 h links of the chain below -- or one round of an array, since
# `uq_campaign.py` shards rows with --task-id/--task-count and the rows are
# independent. The serial chain is kept because it needs no coordination; switch
# to the array when the queue, not the hours, is the constraint.
#
# Self-chaining and --skip-existing, so it survives a wall-time kill and resumes.
#
# The decisive arms are K_gap_1.10 / 1.25 / 1.50 / 2.00 -- a kappa sweep INSIDE
# one family, which breaks the kappa/family-type confound the local probe could
# not.
#
#   sbatch coverage_64_gpu.sh
#
# To stop early: touch \$OUTPUT_ROOT/coverage_64_gpu/.COMPLETE and scancel the
# pending successor.
# ===========================================================================

export GPU_ARCH=v100
SUBMISSION_REPO=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/submission-scripts
JOB_SCRIPT=$SUBMISSION_REPO/jean_zay/raw/equivariant_bootstrap/jobs/coverage_64_gpu.sh
. $SUBMISSION_REPO/jean_zay/raw/equivariant_bootstrap/jobs/_common.sh

CONFIG=$CAMPAIGN_DIR/configs/coverage_64_gpu.yaml
OUT_DIR=$OUTPUT_ROOT/coverage_64_gpu

chain_or_stop "$OUT_DIR" "$JOB_SCRIPT"

run_step python -u scripts/uq_campaign.py \
    --config "$CONFIG" \
    --out-dir "$OUT_DIR" \
    --device cuda \
    --skip-existing

mark_complete "$OUT_DIR"
