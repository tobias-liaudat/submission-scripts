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
#SBATCH --output=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/submission-scripts/jean_zay/raw/equivariant_bootstrap/jobs/logs/R-%x_%A_%a.out
#SBATCH --error=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/submission-scripts/jean_zay/raw/equivariant_bootstrap/jobs/logs/R-%x_%A_%a.err
#SBATCH -A rbn@v100

# ===========================================================================
# RUN THIS BEFORE coverage_64_gpu.sh. It is short and it decides how the main
# sweep is read.
#
# THE PROBLEM. The q-quantile of MC draws is a downward-biased estimator of the
# population quantile. That bias lowers measured coverage for EVERY arm, so part
# of the "the parametric bootstrap under-covers" finding may be a sampling
# artifact. Worse: the kappa that looked best (1.5) is whatever inflates the
# interval back to the right size -- so the apparent optimum is contaminated by
# the same bias and should FALL as MC rises.
#
# THE MEASUREMENT. The same two arms at MC = 32, 64, 128, 256, 512. If the
# parametric arm's coverage at nominal 0.9 climbs and then plateaus, the plateau
# is the real value and the gap to it at MC=32 is the artifact. If it does not
# move, the under-coverage is real and MC=32 was adequate all along.
#
# Two arms only -- the reference and the local winner -- so it stays short.
#
#   sbatch mc_ladder_64.sh
# ===========================================================================

export GPU_ARCH=v100
SUBMISSION_REPO=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/submission-scripts
. $SUBMISSION_REPO/jean_zay/raw/equivariant_bootstrap/jobs/_common.sh

CONFIG=$CAMPAIGN_DIR/configs/coverage_64_gpu.yaml

set -x
for MC in 32 64 128 256 512; do
    srun python -u scripts/uq_campaign.py \
        --config "$CONFIG" \
        --out-dir "$OUTPUT_ROOT/mc_ladder_64/mc$MC" \
        --device cuda \
        --skip-existing \
        --mc "$MC" \
        --rows unrolled/P0_parametric/1 unrolled/K_gap_1.50/1
done
set +x
