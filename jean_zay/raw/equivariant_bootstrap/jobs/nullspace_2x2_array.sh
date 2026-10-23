#!/bin/bash
#SBATCH --job-name=eqb_ns2x2
#SBATCH --mail-user=tobiasliaudat@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH -C h100
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --array=0-3
#SBATCH --output=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/submission-scripts/jean_zay/raw/equivariant_bootstrap/jobs/logs/R-%x_%A_%a.out
#SBATCH --error=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/submission-scripts/jean_zay/raw/equivariant_bootstrap/jobs/logs/R-%x_%A_%a.err
#SBATCH -A rbn@h100

# ===========================================================================
# THE NULL-SPACE PROBE, 2x2: {MeerKAT, VLA-B} x {20 dB, 35 dB}, wavPGD at 360^2.
#
# The science is in `configs/nullspace_2x2_360.yaml`; this file is the launcher.
# In one line: the parametric bootstrap's raw coverage is good in exactly one
# cell of every campaign we have run (`wavpgd/360`, 0.917 against a nominal
# 0.9), and this asks whether a sparser array and/or a quieter observation
# breaks it -- and whether the group action then repairs it.
#
# RUN STAGE 1 FIRST. It is the gate:
#
#   --stage 1  P0_parametric only -- 4 cells x 1 seed =  4 rows  (--array=0-3)
#   --stage 2  everything -- 6 arms x 4 cells x 1 seed = 24 rows (--array=0-23)
#
# If stage 1 shows the parametric arm still covering well in all four cells,
# there is nothing for the group action to repair and stage 2 is 20 rows of
# wasted H100 time: change the setting further (the next lever is the image
# class, which needs new data) rather than running it.
#
#   sbatch --array=0-3  nullspace_2x2_array.sh --stage 1        #  4 rows
#   sbatch --array=0-23 nullspace_2x2_array.sh --stage 2        # 24 rows
#
# `--stage` rather than `STAGE=`: the environment form works, but it is one
# dropped backslash away from silently running stage 1, and stage 2 is ~228
# GPU-h. `--array` is sbatch's own and must still precede the script name.
#
# Stage 2 re-runs the whole row list with `--skip-existing`, so the 12 rows
# stage 1 already wrote are skipped rather than repeated. TASK_COUNT is derived
# from STAGE and must match `--array`; the guard below catches a mismatch.
#
# PREREQUISITES -- all three, or the numbers mean nothing:
#
#   1. `data/images/` holds the 500 HELD-OUT TEST symlinks, not the 13 400-file
#      corpus. `problems.load_images` takes the first n_images in SORTED order,
#      and sorted over the full corpus the first 100 are 89 train / 4 val /
#      7 test -- the probe would be scored mostly on images the solver's
#      wavelet prior is agnostic to but which every NETWORK comparison would be
#      contaminated by, and it would not be comparable to any earlier campaign.
#      The guard below counts the directory and refuses above 1000 files.
#
#   2. `nullspace_tune.sh` has run, its PSNRs are in `runs.yaml` as
#      `psnr_reference`, and `psnr_tolerance_db: 3.0` is in the campaign config.
#      The guard below refuses without it. With a brand-new uv bank, PSNR is the
#      only check that catches a uv / op_norm / psf_fft mis-wiring, all of which
#      are silent.
#
#   3. The VLA-B bank exists:
#      `outputs/artifacts/probe_img360_cov128_vlab_finufft/uv_bank.pt`.
#      Built by `UQsuite/scripts/build_uv_bank.py`; `--check-against` verifies
#      the generator reproduces the production MeerKAT bank bit for bit, which
#      is what makes coverage k the same patch of sky in both banks.
#
# WALL TIME. **2.49 h per row, measured** -- the `seconds` field of
# `wavpgd_360/wavpgd/P0_parametric_seed1.json`, same solver, same batch 50,
# same 360^2 condition (0.70 s per image-draw x 100 images x 128 draws). The
# earlier 3.8 h here was an interpolation from M42 and was conservative. One
# row per array task, so --time=10h is ~4x headroom.
#
# Stage 1 is ~10 GPU-h, stage 2 a further ~50. The VLA-B rows should not
# differ: the cost is the db1-db8 wavelet prox, not the operator, and the PSF
# normal step is an FFT whose size does not depend on the visibility count.
#
# WHAT TO READ, IN ORDER:
#
#   1. CONTROL REPRODUCES? `wavpgd/P0_parametric/{1,2,3}` here must match
#      `outputs/uq_campaign/wavpgd_360/wavpgd/P0_parametric_seed*.json`. Same
#      config, same arms, same seeds -- the only thing that changed is that
#      `data/images/` was rebuilt on 2026-10-19 after the `uem` project was
#      erased. A mismatch means the restored split is not the original one, and
#      every cross-campaign comparison in `plan/project_overview.md` is void.
#
#   2. `coverage_curve.empirical_pre` at level 0.9, per cell. This is the whole
#      experiment. The registered prediction, made before the run: parametric
#      raw coverage falls monotonically in BOTH knobs, i.e.
#         meerkat/20 (0.917, known) > meerkat/35 ~ vlab/20 > vlab/35.
#
#   3. `psnr` against the reference. A row that fails tolerance is a wiring
#      bug, not a bad arm -- stop and investigate rather than reading its
#      coverage.
#
#   4. (stage 2) the kappa that lands on nominal, read off the shelves ladder.
#      Prediction: it rises with the same ordering as (2). At 64^2 the unrolled
#      net needed ~1.48, the 360^2 weighted net ~1.21.
#
# RESUME. `--skip-existing` makes a resubmission a no-op for finished rows.
#   ROWS="$(OUT_DIR=nullspace_2x2_360 CONFIG=nullspace_2x2_360.yaml bash missing_rows.sh)"
# ===========================================================================

export GPU_ARCH=h100
SUBMISSION_REPO=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/submission-scripts
. $SUBMISSION_REPO/jean_zay/raw/equivariant_bootstrap/jobs/_common.sh

# Environment variables are the defaults; the flags below override them. Flags
# exist because `STAGE=2 sbatch ...` is one dropped backslash away from silently
# running stage 1 instead -- which is exactly how two `nullspace_tune.sh` jobs
# were lost on 2026-10-20. Arguments after the script name are passed through by
# sbatch and cannot go missing that way. Here the stakes are higher than in the
# tune job: stage 2 is ~228 GPU-h.
CONFIG=${CONFIG:-nullspace_2x2_360.yaml}
STAGE=${STAGE:-1}
OUT_DIR=${OUT_DIR:-}
ROWS=${ROWS:-}
TASK_COUNT=${TASK_COUNT:-}

while [ $# -gt 0 ]; do
    case "$1" in
        --stage)      STAGE=$2;      shift 2 ;;
        --config)     CONFIG=$2;     shift 2 ;;
        --out-dir)    OUT_DIR=$2;    shift 2 ;;
        --rows)       ROWS=$2;       shift 2 ;;
        --task-count) TASK_COUNT=$2; shift 2 ;;
        -h|--help)
            echo "usage: sbatch [--array=0-N] nullspace_2x2_array.sh \\"
            echo "           [--stage 1|2] [--config F] [--out-dir D] \\"
            echo "           [--rows \"a/b/c ...\"] [--task-count N]"
            echo
            echo "  --stage 1   P0 only, 12 rows   (sbatch --array=0-11)"
            echo "  --stage 2   all arms, 72 rows  (sbatch --array=0-71)"
            exit 0 ;;
        *)
            echo "[ns2x2] unknown argument '$1'. Try --help." >&2
            exit 2 ;;
    esac
done

case "$CONFIG" in */*) ;; *) CONFIG=$CAMPAIGN_DIR/configs/$CONFIG ;; esac

# Derived from the config basename for the same reason `wavpgd_array.sh` does
# it: a row's path is `<out_dir>/<family>/<arm>_seed<seed>.json`, so two
# campaigns sharing a family name and a fixed out-dir would have
# `--skip-existing` skip everything and report success.
OUT_DIR=${OUT_DIR:-$(basename "$CONFIG" .yaml)}
case "$OUT_DIR" in */*) ;; *) OUT_DIR=$OUTPUT_ROOT/$OUT_DIR ;; esac

if [ ! -f "$CONFIG" ]; then
    echo "[ns2x2] no config at '$CONFIG'." >&2
    ls -1 "$CAMPAIGN_DIR/configs" | sed 's/^/[ns2x2]   /' >&2
    exit 2
fi

# --- Guard 1: the image directory is the test split, not the corpus ---------
# See PREREQUISITES above. This is the failure that is silent in every other
# respect: the campaign runs, the PSNRs look fine, and the images are wrong.
N_IMG_FILES=$(find -L "$UQ_IMAGES_DIR" -maxdepth 1 -name '*.npy' | wc -l)
if [ "$N_IMG_FILES" -gt 1000 ]; then
    echo "[ns2x2] REFUSING: $UQ_IMAGES_DIR holds $N_IMG_FILES .npy files." >&2
    echo "[ns2x2] Expected the 500-image held-out TEST split. Sorted over the" >&2
    echo "[ns2x2] full corpus the first 100 are 89 train / 4 val / 7 test, so" >&2
    echo "[ns2x2] this campaign would not be a held-out evaluation and would" >&2
    echo "[ns2x2] not be comparable to any earlier one. See the directory's" >&2
    echo "[ns2x2] PROVENANCE.md for the rebuild recipe." >&2
    exit 2
fi
if [ "$N_IMG_FILES" -lt 100 ]; then
    echo "[ns2x2] REFUSING: $UQ_IMAGES_DIR holds only $N_IMG_FILES .npy files," >&2
    echo "[ns2x2] fewer than the n_test=100 this config asks for." >&2
    exit 2
fi

# --- Guard 2: the PSNR references have been measured -----------------------
if ! grep -q '^\s*psnr_tolerance_db:' "$CONFIG"; then
    echo "[ns2x2] REFUSING: $CONFIG has no psnr_tolerance_db." >&2
    echo "[ns2x2] Three of the four cells ship psnr_reference: null. Run" >&2
    echo "[ns2x2]   sbatch nullspace_tune.sh" >&2
    echo "[ns2x2] first, write the measured PSNRs into runs.yaml, then add" >&2
    echo "[ns2x2]   psnr_tolerance_db: 3.0" >&2
    echo "[ns2x2] to the config's defaults. PSNR is the ONLY check that catches" >&2
    echo "[ns2x2] a uv/op_norm/psf_fft mis-wiring, and the bank is brand new." >&2
    exit 2
fi

# --- Stage selection -------------------------------------------------------
# STAGE is resolved at the top, from --stage or the environment.
case "$STAGE" in
    1)
        # The gate: the non-equivariant arm in all four cells.
        ROWS=${ROWS:-"$(python - "$CONFIG" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
print(" ".join(f"{fam}/P0_parametric/{seed}"
                for fam in cfg["families"] for seed in cfg["seeds"]))
PY
)"}
        ;;
    2)
        ROWS=${ROWS:-}
        ;;
    *)
        echo "[ns2x2] STAGE must be 1 or 2, got '$STAGE'." >&2
        exit 2
        ;;
esac

# MUST match --array. `uq_campaign.py` applies --rows FIRST and then shards
# rows[task_id::task_count], so a TASK_COUNT below the row count silently makes
# each task do several rows (slow but correct) and one above it leaves tasks
# idle. The check below catches the case that actually loses data: a task id
# outside the shard range.
if [ -n "$ROWS" ]; then
    DEFAULT_TASK_COUNT=$(echo "$ROWS" | wc -w)
else
    DEFAULT_TASK_COUNT=$(python - "$CONFIG" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
print(len(cfg["families"]) * len(cfg["arms"]) * len(cfg["seeds"]))
PY
)
fi
TASK_COUNT=${TASK_COUNT:-$DEFAULT_TASK_COUNT}
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

if [ "$TASK_ID" -ge "$TASK_COUNT" ]; then
    echo "[ns2x2] REFUSING: task $TASK_ID of a $TASK_COUNT-row shard --" >&2
    echo "[ns2x2] --array exceeds the row count, so shards overlap." >&2
    exit 2
fi

echo "[ns2x2] stage $STAGE  shard $TASK_ID of $TASK_COUNT"
echo "[ns2x2] config=$(basename "$CONFIG")  ->  $OUT_DIR"
echo "[ns2x2] images=$UQ_IMAGES_DIR ($N_IMG_FILES files)"
[ -n "$ROWS" ] && echo "[ns2x2] restricted to $(echo "$ROWS" | wc -w) row(s)"

run_step python -u scripts/uq_campaign.py \
    --config "$CONFIG" \
    --out-dir "$OUT_DIR" \
    --device cuda \
    --skip-existing \
    --task-id "$TASK_ID" \
    --task-count "$TASK_COUNT" \
    ${ROWS:+--rows $ROWS}
