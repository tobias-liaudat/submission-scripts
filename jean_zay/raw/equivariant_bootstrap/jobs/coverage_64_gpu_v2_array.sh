#!/bin/bash
#SBATCH --job-name=eqb_cov64_v2
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
# THE 64^2 COVERAGE CAMPAIGN, RERUN ON PER-IMAGE COVERAGES.
#
# Successor to `coverage_64_gpu_array.sh`, which it is otherwise a copy of. The
# only difference is OUT_DIR: this writes to `coverage_64_gpu_v2` and leaves the
# first campaign in place as the record of what the pooled code produced.
#
# WHY THE RERUN (plan/changelog.md D43)
# -------------------------------------
# `elliptical_tilt` took its second moment over the whole batch. These coverages
# are tracks at different position angles, so pooling 32 of them averages the
# anisotropy away -- axis ratio 1.13 against a per-image median of 1.40. The
# ellipse was a circle, which made the family `radial_tilt` under another name
# and made its own orientation controls identical to it: aligned 0.06023,
# rotated 0.06003, random 0.06015, radial 0.06035, four names and one transform.
# The M3 gate ("does aiming at the coverage help?") was never evaluated.
#
# Both tilts' knees and `briggs_tilt`'s weight profile pooled the same way and
# are fixed in the same change.
#
# WHY ALL 63 ROWS AND NOT JUST THE ELLIPTICAL ONES
# ------------------------------------------------
# Three families changed, so 33 of the 63 rows move; the rest would be carried
# over from a different code version. Rerunning everything costs ~26 GPU-hours
# more and buys a directory where every arm was measured once, under one commit,
# with no provenance footnote at analysis time. `--skip-existing` also works
# normally on a directory that starts empty, which the mixed layout would not.
#
# ALSO NEW: THE COMPOSITE-KAPPA SWEEP (changelog D45)
# ---------------------------------------------------
# `ALL_transforms` declared kappa 1.25 per family, but kappa multiplies over
# whatever fired and 2.5 of the five families fire per image -- so it realized a
# composite of 1.79 and was never comparable to the `GS_*_1.25` arms it sat
# beside. `ALLc_1.10 / 1.20 / 1.35` declare the **composite** instead and solve
# the per-family value backwards (1.038 / 1.074 / 1.123).
#
# `ALL_transforms` is kept, unchanged, as the bridge to the first campaign.
#
# WHAT TO READ WHEN IT LANDS
# --------------------------
# 1. `GS_elliptical_1.50` against `GC_ell_rotated_1.50` and `GC_ell_random_1.50`.
#    Separated -> the anisotropy is real and the family earns its place. Still
#    tied -> it is isotropic in effect on these coverages and can be dropped.
#    Either is an answer; the tie we have now is an artefact.
#
# 2. `ALLc_1.20` against `GS_shelves_1.10`. The shelves arm realizes kappa 1.196,
#    so this is the head-to-head between the best single family and the mixture
#    at matched conditioning. M35 measured the mixture beating the single-family
#    trend by 4.5 sigma at 1.79; if that holds at its own optimum, the mixture is
#    the recommendation and the whole per-family sweep was measuring the wrong
#    thing.
#
# 3. `kappa_realized.composite` on the ALLc rows, against the targets. Predicted
#    1.09 / 1.19 / 1.33 -- slightly under, because the shelves do not attain
#    their declared span on the grid.
#
# 24 arms x 3 seeds = 72 rows at ~1 h each is ~64 GPU-hours; at 16 shards that is
# 4-5 rows each, comfortably inside the 8 h wall time below.
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
# prefetches DRUNet, which compute nodes cannot download, and checks that
# `briggs_tilt` can grid a weight profile at all.
#
# There is no Briggs profile store to build any more (changelog D44). Per-image
# gridding costs ~6 ms a profile, so a briggs row grids its ~3900 distinct
# (image, R) pairs on demand in about 20 s of its ~54 minutes.
#
# Then smoke one elliptical row on the dev queue before spending the array:
#
#   ROW=unrolled/GS_elliptical_1.50/1 sbatch smoke_64_row.sh
#
#   sbatch coverage_64_gpu_v2_array.sh
#   sbatch coverage_64_gpu_v2_array.sh       # again, to resume; finished rows skip
#
# To stop early: scancel the array. There is no chain to interrupt.
# ===========================================================================

export GPU_ARCH=v100
SUBMISSION_REPO=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/submission-scripts
. $SUBMISSION_REPO/jean_zay/raw/equivariant_bootstrap/jobs/_common.sh

CONFIG=$CAMPAIGN_DIR/configs/coverage_64_gpu.yaml
OUT_DIR=${OUT_DIR:-$OUTPUT_ROOT/coverage_64_gpu_v2}

# Must match the --array range above.
TASK_COUNT=${TASK_COUNT:-16}
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

# Optional: a space-separated list of row keys (`family/arm/seed`) to restrict
# to. `--rows` filters BEFORE the sharding, so a resume that names only the rows
# it needs spreads them one per shard instead of leaving them wherever the
# original partition put them.
#
# That matters after a partial run: the sharding is deterministic, so resubmitting
# the whole array sends every missing row back to the same shard it was lost in.
# Correct either way -- `--skip-existing` will not redo finished rows -- but 16
# rows in 4 shards is 4 h where 16 rows in 16 shards is 1 h.
#
#   OUT_DIR=$OUTPUT_ROOT/coverage_64_gpu_v2 ROWS="$(bash missing_rows.sh)" \
#       sbatch coverage_64_gpu_v2_array.sh
#
# `missing_rows.sh` reads OUT_DIR too, so point it at v2 or it will report the
# first campaign's rows as done.
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
