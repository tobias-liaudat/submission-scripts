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
# ONE ROW OF THE 64^2 CAMPAIGN, BEFORE COMMITTING 56 GPU-HOURS TO THE ARRAY.
#
# `briggs_tilt` is the row to run, and not because it is representative -- it is
# the opposite. It is the only family whose cost model has been verified at 8
# images rather than 100, and the only one that depends on a prebuilt profile
# store landing in the right place under the right name.
#
# WHAT IT PROVES, in the order the log prints it:
#
#   "briggs: 164 prebuilt profiles from briggs_profiles_img64_n100_b32.pt"
#       The store built by `prepare_cluster.sh` was found. If this line is
#       ABSENT the job still runs and still gets the right answer -- it just
#       grids on demand and spends ~15 min doing it (plan/changelog.md D39).
#       A missing line means the keys missed: check that `batch_size` and
#       `n_test` in the config still match what the store was built for.
#
#   the per-row summary line
#       `kappa_realized` should show a genuine spread rather than a constant,
#       since kappa is drawn per image now, and PASS should include
#       `kappa_within_declared_bound` -- the check that the realized maximum
#       respects the top of the declared support (D35, D38).
#
# The output goes to a SCRATCH directory, deliberately. Writing it into the
# campaign's own out-dir would leave `--skip-existing` treating the row as done,
# so the array would inherit a row produced by a different submission.
#
#   sbatch smoke_64_row.sh
# ===========================================================================

export GPU_ARCH=v100
SUBMISSION_REPO=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/submission-scripts
. $SUBMISSION_REPO/jean_zay/raw/equivariant_bootstrap/jobs/_common.sh

CONFIG=$CAMPAIGN_DIR/configs/coverage_64_gpu.yaml
OUT_DIR=${OUT_DIR:-$OUTPUT_ROOT/smoke_64_row}
ROW=${ROW:-unrolled/GS_briggs_1.50/1}

echo "[smoke] row=$ROW  ->  $OUT_DIR"

set -x
srun python -u scripts/uq_campaign.py \
    --config "$CONFIG" \
    --out-dir "$OUT_DIR" \
    --device cuda \
    --rows "$ROW"
set +x
