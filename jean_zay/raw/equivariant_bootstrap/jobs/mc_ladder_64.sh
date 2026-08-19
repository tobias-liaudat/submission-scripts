#!/bin/bash
#SBATCH --job-name=eqb_mc_ladder
#SBATCH --mail-user=tobiasliaudat@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH -C v100-32g
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --qos=qos_gpu-dev
#SBATCH --array=0-2
#SBATCH --output=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/submission-scripts/jean_zay/raw/equivariant_bootstrap/jobs/logs/R-%x_%A_%a.out
#SBATCH --error=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/submission-scripts/jean_zay/raw/equivariant_bootstrap/jobs/logs/R-%x_%A_%a.err
#SBATCH -A rbn@v100

# ===========================================================================
# HOW MANY BOOTSTRAP DRAWS DO THE SELECTION METRICS NEED?
#
# Run this BEFORE any parameter sweep: every sweep inherits the draw count, so
# it is worth settling once rather than guessing.
#
# TWO quantities decide, and they need not settle together. W(c) is a mean over
# source pixels; the conditional-coverage slope is a *contrast* between two
# decile means, and contrasts carry more variance, so the slope should settle
# later. Both are tracked and the recommendation is the larger of the two.
# Observed on a cached 4-image probe: W(c) already converged at MC=16 while the
# slope had not.
#
# NOTE THE QUESTION CHANGED. The previous version of this job asked whether raw
# coverage was biased by too few draws. It is -- the q-quantile of MC draws is a
# downward-biased estimator -- but raw coverage is no longer the objective, and a
# scalar lambda absorbs a roughly uniform bias. What matters now is where the
# CALIBRATED WIDTH W(c) stops moving (plan/transform_selection_metrics.md 3).
# Raw coverage is still reported alongside, so the old question gets an answer
# too, but it does not decide.
#
# THE METHOD. One bootstrap per arm at MC_max, keeping every draw, then
# subsample. The saved draws are the inverse-transformed samples and the summary
# is exactly quantile(|samples - x_hat|, q), so a subset reproduces a smaller-MC
# run EXACTLY rather than approximately (pinned by
# tests/test_bootstrap.py::test_the_summary_is_recoverable_from_the_returned_draws).
# Two consequences: every rung is paired -- same images, same noise, same
# transform draws -- and the ladder costs MC_max instead of the sum of its rungs.
# The old job ran five separate bootstraps to get five rungs.
#
# THE ARMS. One array task each. These are not the families of interest; they
# bracket the range of map shapes the chosen MC has to serve:
#     0  P0_parametric  the reference, narrowest map
#     1  G_mixture      exact permutations, kappa = 1
#     2  S_tilt         radial_tilt kappa = 2, widest map, most draws expected
# If they disagree on the answer, take the largest.
#
# COST, MEASURED -- the README's 0.02-0.1 s per image-draw bracket is wrong and
# it cost three timed-out jobs (1099207, 1101322: all three arms killed at 2 h
# having finished 3 of 7 batches, saving nothing). The real rate on this problem,
# measured through `draw_samples` itself:
#
#     V100  0.24-0.27 s/image-draw      A100  0.21-0.23      H100  0.17-0.20
#
# and it barely responds to the usual levers -- batch 16 -> 50 moves the V100 by
# less than its run-to-run noise, and peak memory is 0.7 GiB of a 32 GiB card.
# The loop is bound inside the tkbn NUFFT, not by batch parallelism or GPU class,
# so there is no tuning here: the job has to be sized to the rate.
#
# At the original 100 images that is 3.4 h of bootstrap plus ~25 min of setup,
# saving and CPU-side ladder scoring -- 3.8 h against a 2 h QoS. Hence 32 images:
# 16,384 image-draws is ~68 min, and with setup and scoring the task lands near
# 1.4 h, inside the dev cap with room for a slow node.
#
# WHAT 32 IMAGES COSTS. The ladder is a WITHIN-arm paired comparison -- every
# rung reuses the same images, the same noise and the same transform draws -- so
# a smaller image count inflates the noise on the absolute W far more than on the
# rung-to-rung movement the plateau is actually read from. Expect `all_valid` to
# come back False, which per `ladder()`'s own comment does not invalidate the MC
# curve; it does mean these J values must not be used to rank the arms against
# each other. For that, rerun at N_IMAGES=100 on qos_gpu-t3 with --time=06:00:00.
#
# The draws are kept (~840 MB per arm). That is deliberate, and it has already
# paid for itself once: the conditional-coverage slope was added to the criterion
# after this job was written, and re-scoring the saved draws costs no GPU at all.
# The aggregation weights in W(c) are still an open choice, so it will likely pay
# again --
#     python scripts/mc_ladder.py --from-cache -o <out-dir>
#
#   sbatch mc_ladder_64.sh
# ===========================================================================

export GPU_ARCH=v100
SUBMISSION_REPO=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/submission-scripts
. $SUBMISSION_REPO/jean_zay/raw/equivariant_bootstrap/jobs/_common.sh

ARMS=(P0_parametric G_mixture S_tilt)
ARM=${ARMS[${SLURM_ARRAY_TASK_ID:-0}]}

N_IMAGES=${N_IMAGES:-32}
MC_MAX=${MC_MAX:-512}
BATCH_SIZE=${BATCH_SIZE:-16}
REPEATS=${REPEATS:-5}

OUT_DIR=$OUTPUT_ROOT/mc_ladder_64

echo "[mc_ladder] arm=$ARM  n_images=$N_IMAGES  MC_max=$MC_MAX  batch=$BATCH_SIZE"
echo "[mc_ladder] $((N_IMAGES * MC_MAX)) image-draws; output -> $OUT_DIR"

set -x
srun python -u scripts/mc_ladder.py \
    --arm "$ARM" \
    --mc-max "$MC_MAX" \
    --n-images "$N_IMAGES" \
    --batch-size "$BATCH_SIZE" \
    --repeats "$REPEATS" \
    --device cuda \
    --out-dir "$OUT_DIR"
set +x
