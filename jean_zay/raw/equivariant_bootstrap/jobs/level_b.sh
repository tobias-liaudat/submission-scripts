#!/bin/bash
#SBATCH --job-name=eqb_level_b
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
# RUN THIS FIRST. Minutes of compute, and it answers the question the whole
# campaign is built around: is the unrolled network more equivariant than the
# artifact-removal one?
#
# Reports E_eq/E_true per family under three noise modes. If the spectral
# ratios do not rise for the unrolled net, the M25 finding does not transfer
# and the expensive 360^2 jobs can be descoped before they are submitted.
#
#   sbatch level_b.sh
# ===========================================================================

export GPU_ARCH=v100
SUBMISSION_REPO=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/submission-scripts
. $SUBMISSION_REPO/jean_zay/raw/equivariant_bootstrap/jobs/_common.sh

set -x
srun python -u scripts/level_b.py \
    --img-size 64 360 \
    --n-images 8 \
    --device cuda \
    --out-dir "$OUTPUT_ROOT/level_b"
