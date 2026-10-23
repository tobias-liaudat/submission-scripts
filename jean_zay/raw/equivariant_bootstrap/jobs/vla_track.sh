#!/bin/bash
#SBATCH --job-name=eqb_vlatrack
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
# DOES A LONGER TRACK MAKE VLA-B TRACTABLE? Tune first, then look.
#
# At 1 hour the VLA-B reconstruction loses the spiral morphology entirely
# (M55). Earth rotation is the only lever that adds real uv coverage -- finer
# cadence just adds redundancy (M54) -- so this extends the track to 2 h and
# 4 h while holding the visibility count at MeerKAT's, isolating coverage from
# sample count:
#
#   vla-b 1h/52s    M = 24 219   2.55 % cells  10.85 % energy  ceil 15.33 dB
#   vla-b 2h/104s   M = 24 219   4.19 %        17.05 %              15.75 dB
#   vla-b 4h/208s   M = 24 219   6.43 %        24.15 %              16.30 dB
#   meerkat 1h/300s M = 24 192   3.52 %        98.00 %              33.44 dB
#
# TWO STAGES IN ONE JOB, and the order is the point. M55's VLA figures were
# rendered at MeerKAT's c = 0.088 and are uninterpretable because of it: the
# ISNR gain came out NEGATIVE, which is what an under-regularised solver does
# when you take its noise away. Judging a condition from a mis-tuned solve is
# the mistake that wasted that round; so this TUNES each cell first and renders
# at the measured optimum. The tuned values go into the figure via
# `--reg-strength`, not into `runs.yaml` -- pinning a constant is a deliberate
# act (D53/O9), not a side effect of a figure job.
#
#   sbatch vla_track.sh
#   IMAGES="0 1 2 3" sbatch vla_track.sh
#
# WHAT TO LOOK FOR:
#
#   1. Does the spiral morphology come back by 4 h? That is the whole question.
#      The energy fraction more than doubles (10.9 -> 24.2 %) while the
#      cell-rounded ceiling moves only +1.0 dB; those two disagree because the
#      ceiling is dominated by the bright core, which is already measured,
#      while the energy counts the extended emission, which is what is missing.
#      The images arbitrate.
#
#   2. Does the ISNR gain turn POSITIVE once c is tuned? If it stays negative
#      at the optimum, the tuning story is wrong and something else is.
#
#   3. Where does the tuned c land? If VLA needs c an order of magnitude above
#      MeerKAT's 0.088, D53's one-constant-for-every-cell is dead and the
#      campaign must tune per cell (O9).
#
#   4. If 4 h is still a blob: the array is too extreme and the next lever is a
#      THINNED MEERKAT -- subsampling its 64 dishes thins the dense core
#      continuously, where the bundled tables offer only the two extremes.
# ===========================================================================

export GPU_ARCH=h100
SUBMISSION_REPO=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/submission-scripts
. $SUBMISSION_REPO/jean_zay/raw/equivariant_bootstrap/jobs/_common.sh

OUT_DIR=${OUT_DIR:-vla_track}
IMAGES=${IMAGES:-"0 1 2"}
# MeerKAT first as the reference, then the VLA series short to long.
FAMILIES=${FAMILIES:-"wavpgd wavpgd_isnr35 \
wavpgd_vlabM wavpgd_vlabM_isnr35 \
wavpgd_vlab2h wavpgd_vlab2h_isnr35 \
wavpgd_vlab4h wavpgd_vlab4h_isnr35"}
# Only the VLA cells are re-tuned; MeerKAT's 0.088 is already pinned on 100
# images (M53) and re-tuning it would change the reference mid-comparison.
TUNE_FAMILIES=${TUNE_FAMILIES:-"wavpgd_vlabM wavpgd_vlab2h wavpgd_vlab4h"}
# Reaches an order of magnitude above 0.088: pass 1 put the VLA optimum at the
# top of a grid that stopped at 0.5, so a grid that stops low would only
# rediscover its own edge.
GRID=${GRID:-"0.088 0.25 0.7 2.0 5.0 12.0"}
TUNE_IMAGES=${TUNE_IMAGES:-32}
# Out-dir names whose `tune/` is also searched for a tuned c. A run that tunes
# only the ISNR-35 cells still has to render the 20 dB rows at THEIR optimum,
# which lives in the earlier run's directory; without this they would fall back
# to the registry's MeerKAT-tuned 0.088 and the figure would mix a tuned row
# with an untuned one.
ALSO_FROM=${ALSO_FROM:-}

# FLAGS, and an UNKNOWN ONE IS AN ERROR. Without this block bash silently drops
# every argument after the script name, so `sbatch vla_track.sh --out-dir X
# --tune-families Y` re-ran the DEFAULT configuration and overwrote the previous
# results with an identical copy (job 1348622, 2026-10-20). A job that ignores
# what it was asked to do and reports success is worse than one that fails.
while [ $# -gt 0 ]; do
    case "$1" in
        --out-dir)       OUT_DIR=$2;      shift 2 ;;
        --families)      FAMILIES=$2;     shift 2 ;;
        --tune-families) TUNE_FAMILIES=$2; shift 2 ;;
        --grid)          GRID=$2;         shift 2 ;;
        --images)        IMAGES=$2;       shift 2 ;;
        --tune-images)   TUNE_IMAGES=$2;  shift 2 ;;
        --also-from)     ALSO_FROM="$ALSO_FROM $2"; shift 2 ;;
        --overwrite)     OVERWRITE=1;     shift ;;
        -h|--help)
            echo "usage: sbatch vla_track.sh [--out-dir D] [--families \"..\"]"
            echo "         [--tune-families \"..\"] [--grid \"..\"] [--images \"..\"]"
            echo "         [--tune-images N] [--overwrite]"
            echo
            echo "  tunes each --tune-families cell, then renders --families at"
            echo "  the tuned c.  A family with no tune file keeps its runs.yaml"
            echo "  value; an _isnr35 family falls back to its 20 dB twin's tune."
            exit 0 ;;
        *)
            echo "[vlatrack] unknown argument '$1'. Try --help." >&2
            exit 2 ;;
    esac
done

case "$OUT_DIR" in */*) ;; *) OUT_DIR=$OUTPUT_ROOT/$OUT_DIR ;; esac
TUNE_DIR=$OUT_DIR/tune

TUNE_SEARCH=$TUNE_DIR
for name in $ALSO_FROM; do
    case "$name" in */*) TUNE_SEARCH="$TUNE_SEARCH $name/tune" ;;
                      *) TUNE_SEARCH="$TUNE_SEARCH $OUTPUT_ROOT/$name/tune" ;;
    esac
done

# Same reasoning as `nullspace_tune.sh`: a second pass into a populated out-dir
# destroys the first, and the two are not interchangeable.
if [ -z "${OVERWRITE:-}" ] && [ -f "$OUT_DIR/summary.json" ]; then
    echo "[vlatrack] REFUSING: $OUT_DIR/summary.json exists." >&2
    echo "[vlatrack]   out-dir        $OUT_DIR" >&2
    echo "[vlatrack]   tune-families  $TUNE_FAMILIES" >&2
    echo "[vlatrack]   grid           $GRID" >&2
    echo "[vlatrack] Pass --out-dir <new> for a new condition, or --overwrite" >&2
    echo "[vlatrack] to genuinely replace this one." >&2
    exit 2
fi

N_IMG_FILES=$(find -L "$UQ_IMAGES_DIR" -maxdepth 1 -name '*.npy' | wc -l)
if [ "$N_IMG_FILES" -gt 1000 ]; then
    echo "[vlatrack] REFUSING: $UQ_IMAGES_DIR holds $N_IMG_FILES .npy files;" >&2
    echo "[vlatrack] expected the 500-image held-out test split." >&2
    exit 2
fi

mkdir -p "$TUNE_DIR"

# --- Stage 1: tune each VLA cell -------------------------------------------
for family in $TUNE_FAMILIES; do
    echo "=============================================================="
    echo "[vlatrack] tuning $family  (n=$TUNE_IMAGES, grid=$GRID)"
    echo "=============================================================="
    run_step python -u scripts/tune_iterative.py \
        --family "$family" --img-size 360 --n-images "$TUNE_IMAGES" \
        --device cuda --grid $GRID -o "$TUNE_DIR"
done

# --- Stage 2: render at the tuned c ----------------------------------------
# One value per family, in the order FAMILIES lists them. A cell with no tune
# file keeps its registry value; the ISNR-35 twin of a tuned cell inherits the
# 20 dB optimum, because `lambda = c*sig*sqrt(op_norm)` already carries the
# ISNR and a separate c per ISNR would confound the two.
REGS=$(python - "$TUNE_SEARCH" "$FAMILIES" <<'PY'
import json, pathlib, sys

search = [pathlib.Path(d) for d in sys.argv[1].split()]
families = sys.argv[2].split()
registry = None


def tuned(name):
    """Best c for `name` from the first tune file that has one."""
    for d in search:
        path = d / f"tune_{name}.json"
        if path.is_file():
            return json.loads(path.read_text())["360"]["best"]["reg_strength"]
    return None


out = []
for fam in families:
    # This cell's OWN sweep first. Only when it has none does it fall back to
    # the 20 dB twin -- the other order silently ignored an ISNR-35 sweep,
    # because every _isnr35 name was rewritten to its twin before looking.
    c = tuned(fam)
    if c is None and fam.endswith("_isnr35"):
        c = tuned(fam[: -len("_isnr35")])
    if c is None:
        if registry is None:
            import yaml
            registry = yaml.safe_load(open("scripts/configs/runs.yaml"))["runs"]
        c = registry[f"{fam}/360"].get("reg_strength", 0.088)
    out.append(f"{c:g}")
print(" ".join(out))
PY
)
echo "[vlatrack] tune path: $TUNE_SEARCH"
echo "[vlatrack] families : $FAMILIES"
echo "[vlatrack] c used   : $REGS"

run_step python -u scripts/reconstruction_examples.py \
    --img-size 360 --families $FAMILIES --images $IMAGES \
    --reg-strength $REGS --device cuda -o "$OUT_DIR"

echo "[vlatrack] done -> $OUT_DIR"
