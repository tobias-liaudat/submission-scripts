#!/bin/bash
#SBATCH --job-name=eqb_iter_tc
#SBATCH --mail-user=tobiasliaudat@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -C a100
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --output=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/submission-scripts/jean_zay/raw/equivariant_bootstrap/jobs/logs/R-%x_%j.out
#SBATCH --error=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/submission-scripts/jean_zay/raw/equivariant_bootstrap/jobs/logs/R-%x_%j.err
#SBATCH -A rbn@a100

# ===========================================================================
# THE TWO NUMBERS THAT DECIDE WHETHER A CLASSICAL-SOLVER CAMPAIGN HAPPENS (D48)
#
# The batched PSF proximal-gradient solver replaces the looped one so that a
# non-learned reconstructor can run inside a campaign, demonstrating that the
# equivariant bootstrap does not depend on our trained nets. Two things are
# unknown and both are measured here, cheapest first:
#
# 1. THE REGULARISATION STRENGTH. A classical solver has no checkpoint, so its
#    parameters *are* the reconstructor and they have to be pinned deliberately.
#    Only one knob acts: for wavPrior the weight `reg_strength` in
#    lambda = c*sig*sqrt(op_norm)
#    (upstream's `g_param` is swallowed by deepinv's `*args` and has never done
#    anything), and for the Lipschitz DnCNN `denoiser_scale` (that net ignores
#    its sigma argument entirely). `tune_iterative.py` sweeps whichever applies.
#
# 2. THE COST. `cost_iterative.py` sweeps batch_size and max_iter and reports
#    `row_hours = seconds_per_image_draw * 12800 / 3600`.
#
# ALREADY RUN ONCE (job 1516871, 12 min). Those numbers stand, with one caveat:
# they predate `batch_separable=True` becoming the default, which stops the
# wavelet dictionary's Dykstra consensus per image so a reconstruction no longer
# depends on what else was in its batch. That costs ~7 % (M39), so re-run this to
# get the exact figures the campaign should be sized on.
#
# WHAT IT MEASURED (A100, standard db1-db8 dictionary, max_iter=100):
#
#     64^2   batch 32   0.323 s/image-draw   1.15 h/row
#     360^2  batch 32   0.620 s/image-draw   2.20 h/row
#     (unrolled/360, for reference: 0.163 s = 0.58 h/row)
#
#   tuned reg_strength   0.088 for BOTH sizes. In lambda units the refined
#                        optima were 0.074 (64^2, 29.11 dB) and 0.105 (360^2,
#                        32.76 dB); 0.088 is their geometric mean and costs
#                        under 0.2 dB against either (D50, M41).
#   trained-net PSNR     31.05 dB at 64^2, 37.54 dB at 360^2
#
# So the standard dictionary is affordable and **is not being shrunk**: db4 alone
# would be ~14x cheaper and is not needed. The batch axis is what pays for it --
# 4 -> 32 is an ~8x improvement per image-draw, which is exactly what the batched
# PSF rewrite existed to buy.
#
# WHAT TO READ IN THE LOG:
#
#   the tuning line "-> best ..."
#       PSNR must land ABOVE the dirty image and BELOW the trained net (31.05 dB
#       at 64^2, 37.54 dB at 360^2). It did, with 2.4 and 4.8 dB of headroom --
#       a worse reconstructor with room to be uncertain, which is the point. A
#       classical solver *beating* the trained one means something is mis-wired.
#       "WARNING: the optimum is at the edge of the grid" means widen --grid.
#       THE KEY CHECK: the optimum should land at the SAME reg_strength at both
#       sizes (~0.088). A residual split means the sqrt(op_norm) EXPONENT is
#       wrong, not the constant -- the measured split is 1.43x, which is 0.7
#       sigma from -1/2 and costs under 0.2 dB, so it is not worth chasing
#       without a third operator scale to fit against.
#
#   the cost table
#       Watch that row_hours keeps falling with batch_size. At 360^2 a PSF kernel
#       is 4 MB per image, so the largest batches may OOM -- reported, not fatal.
#
export GPU_ARCH=${GPU_ARCH:-a100}
SUBMISSION_REPO=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/submission-scripts
. $SUBMISSION_REPO/jean_zay/raw/equivariant_bootstrap/jobs/_common.sh

FAMILY=${FAMILY:-wavpgd}
SIZES=${SIZES:-"64 360"}
OUT_DIR=${OUT_DIR:-iterative_tune_cost}
case "$OUT_DIR" in */*) ;; *) OUT_DIR=$OUTPUT_ROOT/$OUT_DIR ;; esac

echo "[iter] family=$FAMILY sizes=$SIZES -> $OUT_DIR"

# Tuning first: it is the cheaper of the two, and a solver whose regularisation
# is wrong is not worth timing.
run_step python -u scripts/tune_iterative.py \
    --family "$FAMILY" \
    --img-size $SIZES \
    --n-images 8 \
    --device cuda \
    -o "$OUT_DIR"

run_step python -u scripts/cost_iterative.py \
    --family "$FAMILY" \
    --img-size $SIZES \
    --batch-size 4 16 32 64 \
    --max-iter 30 100 \
    --device cuda \
    -o "$OUT_DIR"

# Refine around c = 0.2 at BOTH sizes on a finer grid. Under D49 the two
# optima should now coincide; this is the measurement that confirms or refutes
# the sqrt(op_norm) exponent, so it runs at both sizes rather than just 360.
run_step python -u scripts/tune_iterative.py \
    --family "$FAMILY" \
    --img-size $SIZES \
    --n-images 8 \
    --grid 0.05 0.07 0.088 0.12 0.18 \
    --device cuda \
    -o "$OUT_DIR/refine"
