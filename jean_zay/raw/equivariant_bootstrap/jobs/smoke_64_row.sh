#!/bin/bash
#SBATCH --job-name=eqb_smoke64
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
#SBATCH --output=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/submission-scripts/jean_zay/raw/equivariant_bootstrap/jobs/logs/R-%x_%j.out
#SBATCH --error=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/submission-scripts/jean_zay/raw/equivariant_bootstrap/jobs/logs/R-%x_%j.err
#SBATCH -A rbn@v100

# ===========================================================================
# THE GPU PREFLIGHT, BEFORE COMMITTING ~64 GPU-HOURS TO THE ARRAY.
#
# Two steps, cheapest first, and the second only runs if the first passes.
#
# 1. THE TEST SUITE, ON A GPU. This is the point of the job, and it is two
#    minutes. `prepare_cluster.sh` runs pytest on a LOGIN node, where 13 tests
#    in `test_transform_contract.py` skip for want of a GPU -- and those 13 are
#    the ones that matter here.
#
#    D43 moved coverage selection to a per-image path: `coverage_for(uv, i, B)`
#    hands back `uv[index]`, which on the cluster is a slice of a GPU tensor,
#    and it then goes through `as_host_coverage` / `coverage_key` once per image
#    per draw. That exact path has broken before -- it raised
#    `quantile() q tensor must be on the same device as the input tensor` from
#    `radial_tilt.sample`, and the comment on the fix says it was "not reachable
#    from a CPU-only test run". It names `radial_tilt` and `elliptical_tilt`,
#    which are the two families D43 rewrote.
#
# 2. ONE CAMPAIGN ROW, end to end on real data. `ALLc_1.20` by default because
#    it is the one row that exercises everything changed at once: radial,
#    elliptical and briggs tilts (D43), on-the-fly Briggs gridding with no
#    prebuilt store (D44), and the composite-kappa arithmetic (D45).
#
# WHAT TO READ IN THE LOG:
#
#   the pytest summary
#       "641 passed, 11 skipped". On a login node it is "628 passed, 24
#       skipped": the 13 GPU-gated tests in `test_transform_contract.py` turn
#       from skips into passes. **If it still says 24 skipped, the node gave no
#       GPU and this step proved nothing** -- which is the failure mode worth
#       watching for, because it is silent.
#
#   the per-row summary line
#       `kappa_realized` should show a genuine spread rather than a constant,
#       and for `ALLc_1.20` the composite mean should land near **1.19**
#       (predicted; slightly under the 1.20 target because the shelves do not
#       attain their declared span on the grid). PASS should include
#       `kappa_within_declared_bound` -- the realized maximum inside the top of
#       the declared support, which for this arm is 2.04 (D35, D38, D45).
#
# The output goes to a SCRATCH directory, deliberately. Writing it into the
# campaign's own out-dir would leave `--skip-existing` treating the row as done,
# so the array would inherit a row produced by a different submission.
#
# CONFIG, ROW, OUT_DIR and GPU_ARCH are all overridable, so this smokes any
# campaign rather than only the 64-pixel one. The #SBATCH lines below are the
# 64-pixel defaults; sbatch command-line flags beat them, which is how a 360
# row gets onto an A100:
#
#   CONFIG=campaign_360.yaml ROW=unrolled/ALLc_1.20/1 \
#   OUT_DIR=smoke_360_row GPU_ARCH=a100 \
#       sbatch -C a100 -A rbn@a100 --qos=qos_gpu_a100-dev --cpus-per-task=8 \
#              --time=02:00:00 smoke_64_row.sh
#
#   Bare names, NOT $CAMPAIGN_DIR/... -- those variables live in `_common.sh`
#   and are empty in the shell you type this into.
#
#   sbatch smoke_64_row.sh
# ===========================================================================

export GPU_ARCH=${GPU_ARCH:-v100}
SUBMISSION_REPO=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/submission-scripts
. $SUBMISSION_REPO/jean_zay/raw/equivariant_bootstrap/jobs/_common.sh

# CONFIG and OUT_DIR accept a BARE NAME, resolved here against the campaign's
# own layout. That is not sugar: `$CAMPAIGN_DIR` and `$OUTPUT_ROOT` are defined
# in `_common.sh`, which this job sources at RUNTIME, so a submitter who writes
# `CONFIG=$CAMPAIGN_DIR/configs/x.yaml sbatch ...` has those expand to empty in
# their own shell and silently submits `--config /configs/x.yaml`. Taking a bare
# name removes the trap instead of documenting it.
CONFIG=${CONFIG:-coverage_64_gpu.yaml}
OUT_DIR=${OUT_DIR:-smoke_64_row}
ROW=${ROW:-unrolled/ALLc_1.20/1}

case "$CONFIG" in */*) ;; *) CONFIG=$CAMPAIGN_DIR/configs/$CONFIG ;; esac
case "$OUT_DIR" in */*) ;; *) OUT_DIR=$OUTPUT_ROOT/$OUT_DIR ;; esac

# Before pytest, not after: a mistyped path should cost a second, not the two
# minutes of a GPU test suite followed by a traceback.
if [ ! -f "$CONFIG" ]; then
    echo "[smoke] no config at '$CONFIG'." >&2
    echo "[smoke] pass a bare name (CONFIG=campaign_360.yaml) or an absolute path." >&2
    echo "[smoke] available:" >&2
    ls -1 "$CAMPAIGN_DIR/configs" | sed 's/^/[smoke]   /' >&2
    exit 2
fi

echo "[smoke] pytest on the GPU, then row=$ROW  ->  $OUT_DIR"

# `run_step` exits the job on a non-zero status, so the row below never starts
# if the suite fails -- which is the whole point of ordering it first.
run_step env OMP_NUM_THREADS=1 python -m pytest tests -q

echo "[smoke] suite passed; running one campaign row"

run_step python -u scripts/uq_campaign.py \
    --config "$CONFIG" \
    --out-dir "$OUT_DIR" \
    --device cuda \
    --rows "$ROW"
