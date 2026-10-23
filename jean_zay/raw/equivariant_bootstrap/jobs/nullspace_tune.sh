#!/bin/bash
#SBATCH --job-name=eqb_nstune
#SBATCH --mail-user=tobiasliaudat@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH -C h100
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --output=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/submission-scripts/jean_zay/raw/equivariant_bootstrap/jobs/logs/R-%x_%j.out
#SBATCH --error=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/submission-scripts/jean_zay/raw/equivariant_bootstrap/jobs/logs/R-%x_%j.err
#SBATCH -A rbn@h100

# ===========================================================================
# STEP 1 OF THE NULL-SPACE PROBE. Run this BEFORE `nullspace_2x2_array.sh`.
#
# WHY IT IS A SEPARATE JOB
# ------------------------
# A classical solver's parameters ARE the reconstructor: there is no checkpoint,
# so `reg_strength` is not a tuning detail but the thing being run. D49 ties it
# to the condition through `lambda = c * sig * sqrt(op_norm)`:
#
#   * the ISNR change is absorbed EXACTLY -- `sig` is derived from `isnr_db`, so
#     c should not move between the 20 dB and 35 dB cells at all.
#   * the ARRAY change is absorbed only APPROXIMATELY. `sqrt(op_norm)` took care
#     of 81 % of the 22x gap between the 64^2 and 360^2 conditions (M41), which
#     is why one c = 0.088 serves both -- but VLA-B has a fifth of MeerKAT's
#     visibilities and a completely different radial profile, and nothing
#     guarantees the residual is as small here.
#
# So the two VLA-B cells are the ones that may genuinely need a different c.
# Running all four costs minutes and gives the comparison for free.
#
# IT ALSO PRODUCES THE NUMBER THE CAMPAIGN POLICES ITSELF WITH. Every cell has
# `psnr_reference: null` in runs.yaml except the control, and
# `psnr_tolerance_db` cannot police a reference nobody measured -- which is why
# `nullspace_2x2_360.yaml` ships without the tolerance and the array job
# refuses to start until it is added. With a brand-new uv bank the wiring risk
# (uv / op_norm / psf_fft / imaging_weights, none of them stored anywhere) is at
# its highest, and PSNR is the only thing that catches it.
#
# WHAT TO READ
# ------------
#   operator_ceiling   printed per family. Expect ~34 dB for the MeerKAT cells
#                      and ~16 dB for VLA-B. If VLA-B does NOT drop, the bank is
#                      not being picked up -- check `artifacts_dir` in the output.
#
#   best c             should land near 0.088 for the MeerKAT cells (that is the
#                      pinned value, measured at exactly that condition) and may
#                      differ for VLA-B. An argmax at either END of the grid
#                      means the grid is wrong, not that the optimum is there.
#
#   PSNR at the optimum  this is what goes into `psnr_reference`. For the
#                      control it should reproduce 32.68 dB; if it does not, the
#                      rebuilt `data/images/` is not the original test split and
#                      NOTHING in this probe is comparable to earlier campaigns.
#
# THEN: write the four numbers into `runs.yaml`, add `psnr_tolerance_db: 3.0`
# to the campaign config, and launch stage 1.
#
# TWO PASSES, AND HOW TO ASK FOR THE SECOND
# -----------------------------------------
# Pass 1 (the default) sweeps the script's own wide grid -- 0.0015 to 0.5 in
# 3.2x steps -- which is built to BRACKET an optimum, not to resolve one. It
# left two questions open (changelog O7):
#
#   * the pinned c = 0.088 is not ON that grid, so the control cell was never
#     compared against its recorded 32.68 dB except by interpolating (~31.7,
#     which is 0.8 standard errors low at n = 16: consistent, but not a check).
#   * `wavpgd_vlab_isnr35` reported `optimum_at_grid_edge: true` at c = 0.5, so
#     its optimum is unknown and could lie anywhere above.
#
# Pass 2 answers both, and `--refine` IS pass 2 -- one flag, so the grid cannot
# be mistyped:
#
#   sbatch nullspace_tune.sh --refine
#
# equivalently
#
#   sbatch nullspace_tune.sh --out-dir nullspace_tune_refine --n-images 100 \
#       --grid "0.088 0.16 0.3 0.5 1.0 2.0"
#
# WHY FLAGS AND NOT ENVIRONMENT VARIABLES. `OUT_DIR=... sbatch nullspace_tune.sh`
# does work -- sbatch defaults to --export=ALL -- but it is one dropped
# backslash in a pasted multi-line command away from silently running the
# DEFAULT configuration, which is what happened on 2026-10-20: two jobs
# (1333746, 1333875) wrote nothing because the overwrite guard caught them
# pointing at pass 1's directory. Arguments after the script name are passed
# through by sbatch and cannot be lost that way. The environment variables are
# still honoured, as defaults that the flags override.
#
# n_images sets the standard error of every number here. Pass 1 used 16, where
# the per-image PSNR spread of 5.1 dB at VLA-B leaves SE ~1.3 dB -- fine for
# locating an optimum, too coarse to pin a `psnr_reference` the campaign then
# polices at 3 dB. `--refine` uses 100, which is also exactly the `n_test` the
# campaign scores, so the reference is measured on the SAME images.
#
# ~1.1 s per image at the expensive end, so 6 points x 100 images x 4 families
# is ~45 min -- inside the 2 h dev limit. Raise --time if you widen the grid.
# ===========================================================================

export GPU_ARCH=h100
SUBMISSION_REPO=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/submission-scripts
. $SUBMISSION_REPO/jean_zay/raw/equivariant_bootstrap/jobs/_common.sh

# Environment variables are the defaults; the flags below override them.
OUT_DIR=${OUT_DIR:-nullspace_tune}
N_IMAGES=${N_IMAGES:-16}
GRID=${GRID:-}
FAMILIES=${FAMILIES:-"wavpgd wavpgd_isnr35 wavpgd_vlab wavpgd_vlab_isnr35"}
OVERWRITE=${OVERWRITE:-}

usage() {
    sed -n '/^# TWO PASSES/,/^# ====/p' "$0" | sed 's/^# \?//'
    echo "flags: --refine | --out-dir D | --n-images N | --grid \"..\" |"
    echo "       --families \"..\" | --overwrite"
}

while [ $# -gt 0 ]; do
    case "$1" in
        --refine)
            OUT_DIR=nullspace_tune_refine
            N_IMAGES=100
            GRID="0.088 0.16 0.3 0.5 1.0 2.0"
            shift ;;
        --out-dir)   OUT_DIR=$2;   shift 2 ;;
        --n-images)  N_IMAGES=$2;  shift 2 ;;
        --grid)      GRID=$2;      shift 2 ;;
        --families)  FAMILIES=$2;  shift 2 ;;
        --overwrite) OVERWRITE=1;  shift ;;
        -h|--help)   usage; exit 0 ;;
        *)
            echo "[nstune] unknown argument '$1'." >&2
            usage >&2
            exit 2 ;;
    esac
done

# A bare name is resolved against $OUTPUT_ROOT, as in `wavpgd_array.sh`.
# $OUTPUT_ROOT is set by `_common.sh` and is EMPTY in the shell you type the
# sbatch into, so writing `--out-dir $OUTPUT_ROOT/foo` there would expand to
# `/foo` and fail on a read-only root. Pass the bare name.
case "$OUT_DIR" in */*) ;; *) OUT_DIR=$OUTPUT_ROOT/$OUT_DIR ;; esac

# `tune_iterative.py` writes `tune_<family>.json`, so a second pass into the
# same OUT_DIR silently destroys the first -- and the two passes are not
# interchangeable: the wide default grid is what shows the optimum is bracketed,
# the narrow one is what pins the value. Keep both.
if [ -z "$OVERWRITE" ]; then
    for family in $FAMILIES; do
        if [ -f "$OUT_DIR/tune_$family.json" ]; then
            echo "[nstune] REFUSING: $OUT_DIR/tune_$family.json exists." >&2
            echo "[nstune]" >&2
            echo "[nstune] Resolved configuration:" >&2
            echo "[nstune]   out-dir   $OUT_DIR" >&2
            echo "[nstune]   n-images  $N_IMAGES" >&2
            echo "[nstune]   grid      ${GRID:-<script default, wide>}" >&2
            echo "[nstune]" >&2
            echo "[nstune] If you meant the second pass, that is NOT this" >&2
            echo "[nstune] configuration -- run it as" >&2
            echo "[nstune]   sbatch nullspace_tune.sh --refine" >&2
            echo "[nstune] Pass --overwrite only to genuinely replace pass 1." >&2
            exit 2
        fi
    done
fi

for family in $FAMILIES; do
    echo "=============================================================="
    echo "[nstune] $family  (n_images=$N_IMAGES, 360^2, grid=${GRID:-default})"
    echo "=============================================================="
    run_step python -u scripts/tune_iterative.py \
        --family "$family" \
        --img-size 360 \
        --n-images "$N_IMAGES" \
        --device cuda \
        ${GRID:+--grid $GRID} \
        -o "$OUT_DIR"
done

echo "[nstune] done -> $OUT_DIR"
