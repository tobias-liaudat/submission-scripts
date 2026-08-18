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
# HOW MANY BOOTSTRAP DRAWS DOES W(c) NEED?
#
# Run this BEFORE any parameter sweep: every sweep inherits the draw count, so
# it is worth settling once rather than guessing.
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
# COST. 100 images x 512 draws = 51,200 image-draws per task. At the campaign
# README's own bracket of 0.02-0.1 s per image-draw on a V100 that is 17-85 min,
# so each task fits the 2 h dev QoS with room. It is the array that keeps it
# there: all three arms in one task could exceed 2 h at the pessimistic rate.
#
# The draws are kept (~840 MB per arm). That is deliberate: the aggregation
# weights in W(c) are still an open choice, and keeping them means the ladder can
# be re-scored under a different objective with no GPU at all --
#     python scripts/mc_ladder.py --from-cache -o <out-dir>
#
#   sbatch mc_ladder_64.sh
# ===========================================================================

export GPU_ARCH=v100
SUBMISSION_REPO=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/submission-scripts
. $SUBMISSION_REPO/jean_zay/raw/equivariant_bootstrap/jobs/_common.sh

ARMS=(P0_parametric G_mixture S_tilt)
ARM=${ARMS[${SLURM_ARRAY_TASK_ID:-0}]}

N_IMAGES=${N_IMAGES:-100}
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
