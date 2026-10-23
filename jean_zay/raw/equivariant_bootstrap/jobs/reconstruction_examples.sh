#!/bin/bash
#SBATCH --job-name=eqb_recex
#SBATCH --mail-user=tobiasliaudat@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH -C h100
#SBATCH --hint=nomultithread
#SBATCH --time=00:40:00
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --output=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/submission-scripts/jean_zay/raw/equivariant_bootstrap/jobs/logs/R-%x_%j.out
#SBATCH --error=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/submission-scripts/jean_zay/raw/equivariant_bootstrap/jobs/logs/R-%x_%j.err
#SBATCH -A rbn@h100

# ===========================================================================
# LOOK AT THE RECONSTRUCTIONS BEFORE BUYING THE CAMPAIGN.
#
# One row per cell of the null-space probe, five panels each: ground truth, uv
# coverage, dirty image, reconstruction, |residual|. No bootstrap -- one solve
# per cell, so this is minutes rather than the ~10 GPU-h of stage 1.
#
# WHAT IT IS FOR. The probe's whole case rests on VLA-B being a *harder* problem
# rather than a broken one. The numbers say 25.35 dB against MeerKAT's 33.80,
# with the dirty image at 14.33 dB -- but a PSNR cannot distinguish "hard and
# honest" from "degenerate", which is exactly why KAT-7 was rejected on a number
# (0.04 % of the energy measured) that looked merely extreme. Eyes on the images
# first.
#
#   sbatch reconstruction_examples.sh
#   IMAGES="0 1 2 3 4" sbatch reconstruction_examples.sh
#
# WHAT TO LOOK FOR, in order:
#
#   1. Is the VLA-B reconstruction still a GALAXY? Recognisable morphology,
#      right position, right total flux. If it is a smooth blob or a field of
#      wavelet artifacts, the condition is degenerate and the probe should move
#      to a milder array rather than being run.
#
#   2. Does the 35 dB row look better than the 20 dB row? Under MeerKAT it
#      should, clearly (+7.45 dB). Under VLA-B it should look almost identical
#      (+0.54 dB) -- that near-identity IS the result, and seeing it in the
#      images is the cleanest check that the bias-limited claim is real and not
#      an artifact of the PSNR summary.
#
#   3. Where does the VLA-B residual live? If it sits on the extended, smooth
#      structure, the missing short baselines are doing what the design says.
#      If it is concentrated on the bright compact core, something else is
#      wrong -- that is where the measured visibilities are.
#
#   4. The uv panels, M, and whether the VLA-B scatter looks like a sparse
#      sampling of the same plane or like a different experiment.
#
# THEN the open design question: VLA-B has 4 212 visibilities against MeerKAT's
# 24 192, so the two differ in BOTH uv geometry and total SNR (sigma ~ 1/sqrt(M)
# per visibility). `scripts/vla_cadence.py` lays out the cadences that would
# match M and what each costs; decide from these images plus that table.
# ===========================================================================

export GPU_ARCH=h100
SUBMISSION_REPO=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/submission-scripts
. $SUBMISSION_REPO/jean_zay/raw/equivariant_bootstrap/jobs/_common.sh

OUT_DIR=${OUT_DIR:-reconstruction_examples}
IMG_SIZE=${IMG_SIZE:-360}
IMAGES=${IMAGES:-"0 1 2"}
FAMILIES=${FAMILIES:-"wavpgd wavpgd_isnr35 wavpgd_vlab4h wavpgd_vlab4h_isnr35"}
# The per-cell tuned constants (D56). Empty -> each family's runs.yaml value,
# which is the MeerKAT-tuned 0.088 everywhere and WRONG for the VLA cells
# (M55/M58): rendering them at 0.088 is what made the first VLA figures
# uninterpretable. Keep this in step with FAMILIES.
REGS=${REGS:-"0.16 0.088 0.7 2.0"}

# Flags, and an unknown one is an error -- see O10, where a job silently ran its
# defaults because bash drops unparsed arguments and then reported success.
while [ $# -gt 0 ]; do
    case "$1" in
        --out-dir)       OUT_DIR=$2;   shift 2 ;;
        --families)      FAMILIES=$2;  shift 2 ;;
        --reg-strength)  REGS=$2;      shift 2 ;;
        --images)        IMAGES=$2;    shift 2 ;;
        --img-size)      IMG_SIZE=$2;  shift 2 ;;
        --overwrite)     OVERWRITE=1;  shift ;;
        -h|--help)
            echo "usage: sbatch reconstruction_examples.sh [--out-dir D]"
            echo "         [--families \"..\"] [--reg-strength \"..\"]"
            echo "         [--images \"..\"] [--img-size N] [--overwrite]"
            echo
            echo "  --reg-strength takes one value per family, or one for all."
            echo "  Defaults are the D56 four-cell campaign at its tuned c."
            exit 0 ;;
        *) echo "[recex] unknown argument '$1'. Try --help." >&2; exit 2 ;;
    esac
done

case "$OUT_DIR" in */*) ;; *) OUT_DIR=$OUTPUT_ROOT/$OUT_DIR ;; esac

if [ -z "${OVERWRITE:-}" ] && [ -f "$OUT_DIR/summary.json" ]; then
    echo "[recex] REFUSING: $OUT_DIR/summary.json exists." >&2
    echo "[recex]   families  $FAMILIES" >&2
    echo "[recex]   c         ${REGS:-<registry>}" >&2
    echo "[recex] Pass --out-dir <new>, or --overwrite to replace it." >&2
    exit 2
fi

n_fam=$(echo "$FAMILIES" | wc -w); n_reg=$(echo "$REGS" | wc -w)
if [ -n "$REGS" ] && [ "$n_reg" -ne 1 ] && [ "$n_reg" -ne "$n_fam" ]; then
    echo "[recex] REFUSING: $n_reg values in --reg-strength for $n_fam" >&2
    echo "[recex] families. Pass one each, or one for all." >&2
    exit 2
fi

# The same guard as the campaign: the first n_images in SORTED order must come
# from the held-out test split, or the panels are training images.
N_IMG_FILES=$(find -L "$UQ_IMAGES_DIR" -maxdepth 1 -name '*.npy' | wc -l)
if [ "$N_IMG_FILES" -gt 1000 ]; then
    echo "[recex] REFUSING: $UQ_IMAGES_DIR holds $N_IMG_FILES .npy files;" >&2
    echo "[recex] expected the 500-image held-out test split." >&2
    exit 2
fi

echo "[recex] $IMG_SIZE^2  images=[$IMAGES]  ->  $OUT_DIR"
echo "[recex] families : $FAMILIES"
echo "[recex] c        : ${REGS:-<runs.yaml>}"

run_step python -u scripts/reconstruction_examples.py \
    --img-size "$IMG_SIZE" \
    --families $FAMILIES \
    --images $IMAGES \
    ${REGS:+--reg-strength $REGS} \
    --device cuda \
    -o "$OUT_DIR"
