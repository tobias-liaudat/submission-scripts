#!/bin/bash
#SBATCH --job-name=eqb_cov64_arr
#SBATCH --mail-user=tobiasliaudat@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH -C v100-32g
#SBATCH --hint=nomultithread
#SBATCH --time=08:00:00
#SBATCH --qos=qos_gpu-t3
#SBATCH --array=0-15
#SBATCH --output=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/submission-scripts/jean_zay/raw/equivariant_bootstrap/jobs/logs/R-%x_%A_%a.out
#SBATCH --error=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/submission-scripts/jean_zay/raw/equivariant_bootstrap/jobs/logs/R-%x_%A_%a.err
#SBATCH -A rbn@v100

# ===========================================================================
# THE 64^2 COVERAGE CAMPAIGN, SHARDED ACROSS GPUs.
#
# Same config, same rows and the same total GPU time as `coverage_64_gpu.sh`;
# only the wall clock differs. 21 arms x 3 seeds = 63 rows at ~1 h each is ~56
# GPU-hours, which is three 20 h links of the serial chain or one round of this.
#
#   16 tasks x 4 rows  ~ 4 h        <- as configured
#    8 tasks x 8 rows  ~ 8 h
#   21 tasks x 3 rows  ~ 3 h
#
# `uq_campaign.py` does the sharding itself: `--task-id i --task-count n` takes
# rows[i::n], so the tasks partition the row list with no coordination and no
# shared state. Change --array and --task-count TOGETHER or the shards will
# overlap or leave gaps.
#
# NO SELF-CHAINING HERE, unlike the serial job. Sixty-three rows across sixteen
# tasks that each queue a successor is a chain per shard, and a `.COMPLETE` file
# that means different things to different tasks. Resuming is `--skip-existing`
# instead: resubmit the same array and every finished row is skipped, so a
# wall-time kill costs the rows that were in flight and nothing else.
#
# `--skip-existing` skips rows that SUCCEEDED, not rows that left a file. A
# failed row writes a record carrying `error` and `traceback` to the same path,
# so keying the resume on existence would skip the failures too and a resubmitted
# array would report nothing left to do while the gap went unnoticed until
# someone counted rows at analysis time. Failed rows are retried, and the log
# says `retry ... (previous attempt failed)` when that happens.
#
# BEFORE THE FIRST SUBMISSION run `prepare_cluster.sh` on a login node -- it
# prefetches DRUNet (compute nodes have no network) and builds the Briggs
# profile store, without which every briggs row spends ~15 min gridding.
#
#   sbatch coverage_64_gpu_array.sh
#   sbatch coverage_64_gpu_array.sh          # again, to resume; finished rows skip
#
# To stop early: scancel the array. There is no chain to interrupt.
# ===========================================================================

export GPU_ARCH=v100
SUBMISSION_REPO=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/submission-scripts
. $SUBMISSION_REPO/jean_zay/raw/equivariant_bootstrap/jobs/_common.sh

CONFIG=$CAMPAIGN_DIR/configs/coverage_64_gpu.yaml
OUT_DIR=$OUTPUT_ROOT/coverage_64_gpu

# Must match the --array range above.
TASK_COUNT=${TASK_COUNT:-16}
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

echo "[array] shard $TASK_ID of $TASK_COUNT  ->  $OUT_DIR"

set -x
srun python -u scripts/uq_campaign.py \
    --config "$CONFIG" \
    --out-dir "$OUT_DIR" \
    --device cuda \
    --skip-existing \
    --task-id "$TASK_ID" \
    --task-count "$TASK_COUNT"
set +x
