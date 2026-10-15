#!/bin/bash
#SBATCH --job-name=eqb_360_uw
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
# THE 360^2 CAMPAIGN ON THE UNWEIGHTED-LOSS NETWORK.
#
# Same architecture as `unrolled/360`, trained with `weighted_loss: false` and
# otherwise identical (a diff of the resolved configs shows that flag and the run
# name and nothing else). Same 13 arms, seeds, images and coverages as
# `campaign_360.yaml`, so every row is paired with one already on disk.
#
# To separate two candidate explanations for how well the parametric bootstrap
# does, and the gallery run has already picked between them.
#
# The first guess was "head-room to the operator ceiling": parametric should look
# good wherever the reconstructor sits far below 36.78 dB, because then the error
# lives inside the measured subspace where resampling noise can see it. **That is
# wrong.** Measured on image 0 at MC=128, sigma/|residual| for the parametric arm:
#
#     unrolled_unweighted/360   PSNR 30.88   0.39   <- furthest below ceiling,
#     unrolled/64               PSNR 30.37   0.72       and the WORST parametric
#     unrolled/360              PSNR 33.95   0.77
#     wavpgd/64                 PSNR 30.49   0.85
#     wavpgd/360                PSNR 33.11   1.16
#
# Head-room does not order that list. What does is **learned prior versus fixed
# prior**. wavpgd is a convex solver whose map is the same function on every
# draw, so noise resampling explores essentially all of its variability and
# parametric is already near-calibrated (1.16). A network's error includes a
# systematic, content-dependent component -- the learned prior's bias -- which
# barely moves when the noise is resampled, so parametric cannot see it. The
# weaker the network, the larger that component: this run is 8 dB worse in
# validation than its weighted-loss sibling and its parametric sigma covers only
# 39 % of the error.
#
# So this campaign is the strongest test of the group action we have run: it has
# the most for the transforms to recover (0.39 -> 0.84 on that one image), and it
# is a *network*, so unlike the wavpgd comparison it is not confounded with the
# solver family. One training flag is the only difference from `campaign_360`.
#
# Registered before the run: the equivariant arms should improve raw coverage
# MORE here than in any campaign so far, and parametric should be the worst it
# has been. The gallery says that on one image; this says it on 100.
#
# WHAT TO READ:
#
#   psnr             the first thing to check, and `psnr_reference` is null for
#                    this run BY DESIGN -- the val/test gap on the sibling was
#                    2.7 dB, so a reference derived from validation would be a
#                    guess policed by psnr_tolerance_db. Fill runs.yaml in from
#                    the first row that lands. Expect roughly 29-30 dB.
#
#   RAW coverage     `coverage_curve.empirical_pre`, NOT the calibrated number.
#                    Every arm is scalar-calibrated to hit 0.9, so the calibrated
#                    coverage and J are nearly blind by construction; the raw
#                    value and the fitted lambda are where arms separate.
#
#   bin-1 coverage   at 360^2 the deviation is a hump, so `slope` differences two
#                    low end-bins and can call a bad curve flat (M45). Rank on
#                    max_deviation or bin 1.
#
# TWO ROUNDS OF TEN, ON THE DEV QUEUE. `qos_gpu_a100-dev` caps
# MaxSubmitJobsPerUser at **10** -- and that counts SUBMITTED array tasks, not
# running ones, so an `--array=0-38` is rejected outright with
# QOSMaxSubmitJobPerUserLimit. The 39 rows are therefore sharded 20 ways and run
# as two submissions of ten, which is the same pattern `campaign_360_array.sh`
# uses and for the same reason.
#
#   sbatch campaign_360_unweighted_array.sh                  # round 1: shards 0-9
#   sbatch --array=10-19 campaign_360_unweighted_array.sh     # round 2: shards 10-19
#
# Leave TASK_COUNT at 20 for BOTH; only --array changes. Round 2 can go in as
# soon as round 1 has finished (or drained below ten queued).
#
# WALL TIME. 39 rows over 20 shards is 2 rows each, and the weighted sibling
# measured 0.58 h per row (worst 0.63), so ~1.2 h per shard against the 2 h cap
# -- about 40 min of margin. `--skip-existing` makes a re-submission a no-op for
# finished rows, so a shard that does hit the wall is recovered by re-running it.
#
# Dev rather than t3 because turnaround is what matters here: 39 tasks queued on
# a 20 h QoS can sit for a long time, whereas dev is sized for exactly this.
# ===========================================================================

export GPU_ARCH=a100
SUBMISSION_REPO=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/submission-scripts
. $SUBMISSION_REPO/jean_zay/raw/equivariant_bootstrap/jobs/_common.sh

CONFIG=${CONFIG:-campaign_360_unweighted.yaml}
case "$CONFIG" in */*) ;; *) CONFIG=$CAMPAIGN_DIR/configs/$CONFIG ;; esac

# Out-dir follows the config's basename so two campaigns cannot land on each
# other: row files are `<family>/<arm>_seed<seed>.json` and collide across
# campaigns of the same family, where `--skip-existing` would then skip
# everything and report success.
OUT_DIR=${OUT_DIR:-$(basename "$CONFIG" .yaml)}
case "$OUT_DIR" in */*) ;; *) OUT_DIR=$OUTPUT_ROOT/$OUT_DIR ;; esac

[ -f "$CONFIG" ] || { echo "[360uw] no config at '$CONFIG'" >&2; exit 2; }

TASK_COUNT=${TASK_COUNT:-20}
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
ROWS=${ROWS:-}

# `uq_campaign.py` shards as rows[i::TASK_COUNT]. The invariant that matters is
# TASK_ID < TASK_COUNT: a task index at or above it does NOT get an empty shard,
# it gets a wrong one that overlaps another task's rows, so the campaign
# silently duplicates some rows and drops others. That is the failure a
# multi-round submission can actually cause -- mistyping --array=10-19 while
# leaving TASK_COUNT at 10 -- so check it rather than the row count, which is
# deliberately NOT equal to TASK_COUNT here.
if [ "$TASK_ID" -ge "$TASK_COUNT" ]; then
    echo "[360uw] TASK_ID=$TASK_ID is >= TASK_COUNT=$TASK_COUNT." >&2
    echo "[360uw] Shards overlap. Raise TASK_COUNT to cover the --array range." >&2
    exit 2
fi

ROWS_TOTAL=$(python -c "import yaml;c=yaml.safe_load(open('$CONFIG'));print(len(c['arms'])*len(c['seeds']))")
echo "[360uw] $ROWS_TOTAL rows over $TASK_COUNT shards (~$(( (ROWS_TOTAL + TASK_COUNT - 1) / TASK_COUNT )) per shard)"

echo "[360uw] shard $TASK_ID of $TASK_COUNT  config=$(basename "$CONFIG")  ->  $OUT_DIR"

run_step python -u scripts/uq_campaign.py \
    --config "$CONFIG" \
    --out-dir "$OUT_DIR" \
    --device cuda \
    --skip-existing \
    --task-id "$TASK_ID" \
    --task-count "$TASK_COUNT" \
    ${ROWS:+--rows $ROWS}
