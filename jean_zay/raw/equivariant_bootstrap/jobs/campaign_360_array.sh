#!/bin/bash
#SBATCH --job-name=eqb_camp360_arr
#SBATCH --mail-user=tobiasliaudat@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -C a100
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --array=0-9
#SBATCH --output=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/submission-scripts/jean_zay/raw/equivariant_bootstrap/jobs/logs/R-%x_%A_%a.out
#SBATCH --error=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/submission-scripts/jean_zay/raw/equivariant_bootstrap/jobs/logs/R-%x_%A_%a.err
#SBATCH -A rbn@a100

# ===========================================================================
# THE 360^2 CAMPAIGN, SHARDED ACROSS A100s.
#
# Replaces `campaign_360_unrolled.sh` and `campaign_360_ar.sh`, which were built
# for the pre-D38 config: two reconstructors, one seed, RCPS on, and a 100 h
# serial walltime chosen when 360^2 was believed to be ~20x the cost of 64^2.
# It is not -- `unrolled/360` uses a PSF normal step rather than 64^2's tkbn
# NUFFT, and measures CHEAPER per image-draw despite 31.6x the pixels.
#
# 13 arms x 3 seeds = 39 rows. MEASURED at 37.6 min a row (smoke row 1454522:
# 0.1765 s per image-draw x MC 128 x 100 images), so ~24.5 A100-hours total.
#
# DEV TAKES TWO ROUNDS, AND THIS IS WHY.
#
#   qos_gpu_a100-dev    MaxWall 02:00:00    MaxSubmitJobsPerUser 10
#
# An array of N tasks counts as N submitted jobs, so anything past 10 is
# rejected outright with QOSMaxSubmitJobPerUserLimit. `%N` does not help: it
# throttles what RUNS, not what is submitted. So the array is ten tasks, and the
# 39 rows are split across TWO submissions of ten:
#
#   sbatch campaign_360_array.sh                    # round 1: tasks 0-9
#   ...wait for it to drain, then...
#   sbatch --array=10-19 campaign_360_array.sh      # round 2: tasks 10-19
#
# TASK_COUNT stays 20 for both, because `uq_campaign.py` takes rows[i::20]:
# round 1 covers i = 0..9 and round 2 covers i = 10..19, together the whole 39
# with no overlap and no gaps. Round 2 cannot be queued alongside round 1 -- a
# pending job still counts against the submit limit -- so it waits.
#
# Two rows per task at 37.6 min each is ~75 min, inside the 2 h cap with 45 min
# of slack.
#
# ONE SUBMISSION INSTEAD, on t3, if the wait on dev is worse than the queue:
#
#   TASK_COUNT=10 sbatch --array=0-9 --qos=qos_gpu_a100-t3 --time=05:00:00 \
#       campaign_360_array.sh
#
# That is 4 rows a task, ~2.5 h, well inside t3's 20 h and its 10000-job limit.
#
# THE FAILURE MODE TO KNOW ABOUT. `--skip-existing` resumes whole rows, not
# partial ones, so a task killed at the wall clock loses whatever row was in
# flight and a resubmit restarts it from the beginning. That is survivable here
# only because a row is 37.6 min against a 2 h cap; if MC or n_test ever grows,
# recheck this arithmetic before staying on dev.
#
# BEFORE THE FIRST SUBMISSION
#   bash prepare_cluster.sh                    # DRUNet, and the briggs check
#   CONFIG=campaign_360.yaml ROW=unrolled/ALLc_1.20/1 \
#   OUT_DIR=smoke_360_row GPU_ARCH=a100 \
#       sbatch -C a100 -A rbn@a100 --qos=qos_gpu_a100-dev --cpus-per-task=8 \
#              --time=02:00:00 smoke_64_row.sh
#
#   Bare names. `$CAMPAIGN_DIR` and `$OUTPUT_ROOT` are set in `_common.sh` at
#   runtime and are EMPTY in the shell you type the sbatch into.
#
# Then read, in the smoke log:
#   * `seconds_per_image_draw` -- multiply by MC x n_test (128 x 100) for the
#     real row cost. If that exceeds ~1.5 h, move off dev per the note above.
#   * that `batch_size: 4` did not OOM. 64^2 ran at 32 on a 32 GB V100 and this
#     is 31.6x the pixels per image; 4 is a guess with slack, not a measurement.
#
# WHAT TO READ WHEN IT LANDS
#   1. Does the 64^2 ordering hold? shelves > ALLc ~ shelves > briggs >
#      (radial ~ elliptical ~ geometric) > parametric.
#   2. `GS_shelves_1.05` (realized kappa 1.098) against `GS_shelves_1.10`
#      (1.200). At 64^2 every family's best was the lowest point on the grid, so
#      the optimum was never bracketed from below. This is the first test of
#      whether it lies lower still.
#   3. `ALLc_1.20` against `GS_shelves_1.10` -- the mixture against the best
#      single family at matched conditioning. Tied at 64^2 (+0.9 sigma paired).
#
#   sbatch campaign_360_array.sh                 # round 1
#   sbatch --array=10-19 campaign_360_array.sh   # round 2, after round 1 drains
#   sbatch campaign_360_array.sh                 # again to resume; finished rows skip
# ===========================================================================

export GPU_ARCH=a100
SUBMISSION_REPO=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/submission-scripts
. $SUBMISSION_REPO/jean_zay/raw/equivariant_bootstrap/jobs/_common.sh

CONFIG=${CONFIG:-campaign_360.yaml}
OUT_DIR=${OUT_DIR:-campaign360}

# Bare names resolved against the campaign layout; see smoke_64_row.sh.
case "$CONFIG" in */*) ;; *) CONFIG=$CAMPAIGN_DIR/configs/$CONFIG ;; esac
case "$OUT_DIR" in */*) ;; *) OUT_DIR=$OUTPUT_ROOT/$OUT_DIR ;; esac

# Must match the --array range above; `uq_campaign.py` takes rows[i::n].
TASK_COUNT=${TASK_COUNT:-20}
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

# `ROWS="$(OUT_DIR=$OUT_DIR CONFIG=$CONFIG bash missing_rows.sh)"` to resume a
# partial run one row per shard rather than wherever the partition left them.
ROWS=${ROWS:-}

echo "[array] shard $TASK_ID of $TASK_COUNT  ->  $OUT_DIR"
[ -n "$ROWS" ] && echo "[array] restricted to $(echo "$ROWS" | wc -w) row(s)"

run_step python -u scripts/uq_campaign.py \
    --config "$CONFIG" \
    --out-dir "$OUT_DIR" \
    --device cuda \
    --skip-existing \
    --task-id "$TASK_ID" \
    --task-count "$TASK_COUNT" \
    ${ROWS:+--rows $ROWS}
