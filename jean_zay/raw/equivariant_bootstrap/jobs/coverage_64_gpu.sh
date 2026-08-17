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
# Raw bootstrap coverage at 64^2, no conformalisation. 14 arms x 3 seeds = 42
# rows, MC=256, 100 images -> 1,075,200 image-draws.
#
# Self-chaining and --skip-existing, so it survives a wall-time kill and resumes.
# At 0.05 s/image-draw this is ~15 h (one job); at 0.10 s it is ~30 h (two links).
# Read the real rate off prepare_cluster.sh's --dry-run before trusting either.
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

set -x
srun python -u scripts/uq_campaign.py \
    --config "$CONFIG" \
    --out-dir "$OUT_DIR" \
    --device cuda \
    --skip-existing
set +x

mark_complete "$OUT_DIR"
