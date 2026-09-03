#!/bin/bash
#SBATCH --job-name=eqb_gallery
#SBATCH --mail-user=tobiasliaudat@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -C a100
#SBATCH --hint=nomultithread
#SBATCH --time=04:00:00
#SBATCH --qos=qos_gpu_a100-t3
#SBATCH --output=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/submission-scripts/jean_zay/raw/equivariant_bootstrap/jobs/logs/R-%x_%j.out
#SBATCH --error=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/submission-scripts/jean_zay/raw/equivariant_bootstrap/jobs/logs/R-%x_%j.err
#SBATCH -A rbn@a100

# ===========================================================================
# ONE DIAGNOSTIC PANEL PER RECONSTRUCTOR: is the null space too small?
#
# The campaign found the parametric bootstrap nearly as good as the group
# action on this problem (M43), which is what you would expect if the operator
# destroyed little and noise propagation dominated the error. This puts the
# pieces of that question side by side for one sky: the uv coverage against the
# FULL image grid, the dirty image, the reconstruction, and the bootstrap spread
# under three arms on a shared colour scale.
#
# Four figures -- unrolled and wavpgd, at 64 and 360.
#
# WHAT TO READ:
#
#   "% of cells filled" in the uv title
#       The null-space number. Fraction of Fourier cells INSIDE the band limit
#       that any visibility lands in; the rest is unconstrained and filled in by
#       the prior. 34.2% at 64^2 for image 0. Everything outside the dashed
#       circle is null space by construction (super_resolution = 1.5).
#       **This is the knob to change if the comparison is to be made harder** --
#       fewer visibilities, shorter observation, or a larger super_resolution.
#
#   dirty vs ground truth
#       If the dirty image already shows the sky's structure clearly, the
#       operator is not destroying much and the parametric bootstrap has little
#       to miss. That is the visual form of the same question.
#
#   the residual and the three sigma maps
#       ALL FOUR share one colour scale, and each reports its source-masked mean.
#       The sigma panels also give the ratio sigma/|residual| BEFORE calibration
#       -- an uncertainty map is only useful if its magnitude matches the error.
#       For unrolled/64 that runs 0.57 -> 0.71 -> 0.93 across the three arms,
#       which is what the group action buys, stated as a number.
#
#   Parametric captures noise propagation ONLY; what the group action adds on
#   top of it is the part of the error that resampling noise cannot see.
#
# CACHING. The arrays land in `<out-dir>/cache/*.pt`, so re-plotting is free:
#
#   python scripts/reconstruction_gallery.py -c <configs> --replot -o <out-dir>
#
# needs no GPU and no operator at all. Change the figure, not the bootstrap.
#
# COST. MC x 3 arms reconstructions of a single image per figure. The nets are
# minutes; wavpgd at 360^2 is the long pole (batch size 1 is inefficient for it),
# hence 4 h on t3 rather than the 2 h dev cap.
#
#   sbatch reconstruction_gallery.sh
#   MC=64 IMAGE_INDEX=7 sbatch reconstruction_gallery.sh     # cheaper / other sky
# ===========================================================================

export GPU_ARCH=a100
SUBMISSION_REPO=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/submission-scripts
. $SUBMISSION_REPO/jean_zay/raw/equivariant_bootstrap/jobs/_common.sh

MC=${MC:-128}
IMAGE_INDEX=${IMAGE_INDEX:-0}
OUT_DIR=${OUT_DIR:-reconstruction_gallery}
case "$OUT_DIR" in */*) ;; *) OUT_DIR=$OUTPUT_ROOT/$OUT_DIR ;; esac

echo "[gallery] MC=$MC image=$IMAGE_INDEX -> $OUT_DIR"

run_step python -u scripts/reconstruction_gallery.py \
    --config "$CAMPAIGN_DIR/configs/coverage_64_gpu.yaml" \
             "$CAMPAIGN_DIR/configs/campaign_360.yaml" \
             "$CAMPAIGN_DIR/configs/wavpgd_64.yaml" \
             "$CAMPAIGN_DIR/configs/wavpgd_360.yaml" \
    --image-index "$IMAGE_INDEX" \
    --mc "$MC" \
    --device cuda \
    -o "$OUT_DIR"
