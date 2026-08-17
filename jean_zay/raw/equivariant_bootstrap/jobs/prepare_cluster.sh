#!/bin/bash
# Preflight for the equivariant-bootstrap UQ campaign.
# Run on a LOGIN NODE, not under sbatch.
#
#   bash prepare_cluster.sh
#
# Every check either passes or fails loudly, naming what is missing and how to
# fix it. Nothing here is expensive; the point is to turn failures that would
# otherwise surface hours into a GPU job into failures that surface now, while
# someone is watching.
#
# It changes exactly one thing on disk: it populates TORCH_HOME with the DRUNet
# weights. Everything else is read-only.

set -uo pipefail

RADIO_ROOT=${RADIO_ROOT:-/lustre/fswork/projects/rech/ney/ulx23va/projects/radio}
CODE_REPO=${CODE_REPO:-$RADIO_ROOT/repos/UQsuite}
SUBMISSION_REPO=${SUBMISSION_REPO:-$RADIO_ROOT/repos/submission-scripts}
RRT_DIR=${RRT_DIR:-$RADIO_ROOT/repos/radio_reconstruction_tools}
RADIO_TOOLS_DIR=${RADIO_TOOLS_DIR:-$RADIO_ROOT/repos/radio_tools}
COVERAGE_DIR=${COVERAGE_DIR:-$RADIO_ROOT/repos/coverage_plots}
CAMPAIGN_DIR=$SUBMISSION_REPO/jean_zay/raw/equivariant_bootstrap

export TORCH_HOME=${TORCH_HOME:-$WORK/.cache/torch}
export HF_HOME=${HF_HOME:-$WORK/.cache/huggingface}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-$WORK/.cache}
export OMP_NUM_THREADS=1
export KMP_DUPLICATE_LIB_OK=TRUE
export UQ_MODELS_DIR=${UQ_MODELS_DIR:-$RADIO_ROOT/outputs/trained_models}
export UQ_ARTIFACTS_DIR=${UQ_ARTIFACTS_DIR:-$RADIO_ROOT/outputs/artifacts}
export UQ_IMAGES_DIR=${UQ_IMAGES_DIR:-$RADIO_ROOT/data/images}

FAILURES=0
step() { printf '\n\033[1m== %s\033[0m\n' "$1"; }
ok()   { printf '   ok    %s\n' "$1"; }
bad()  { printf '   FAIL  %s\n' "$1" >&2; FAILURES=$((FAILURES + 1)); }

. "$SUBMISSION_REPO/jean_zay/env_configs/equivariant_bootstrap.sh"

# ---------------------------------------------------------------------------
step "1. repositories"
# ---------------------------------------------------------------------------
for repo in "$CODE_REPO" "$RRT_DIR" "$RADIO_TOOLS_DIR" "$COVERAGE_DIR" "$SUBMISSION_REPO"; do
    if [ -d "$repo/.git" ]; then
        if git -C "$repo" pull --ff-only >/dev/null 2>&1; then
            ok "$(basename "$repo") @ $(git -C "$repo" rev-parse --short HEAD)"
        else
            bad "$(basename "$repo"): git pull --ff-only failed (local changes?)"
        fi
    else
        bad "$repo is not a git checkout"
    fi
done

# The campaign code is newer than the last cluster run; if the push has not
# happened yet, this is where it shows up rather than inside a GPU job.
for f in "$CODE_REPO/scripts/uq_campaign.py" "$CODE_REPO/scripts/level_b.py" \
         "$CODE_REPO/scripts/problems.py" "$CODE_REPO/uqsuite/transforms/modulation.py" \
         "$CODE_REPO/uqsuite/diagnostics/coverage.py"; do
    [ -f "$f" ] && ok "$(basename "$f")" || bad "missing $f -- has the campaign work been pushed?"
done

# `updates` is the only branch with reconstructor.py (the artifact_removal
# family) and the imaging_weights argument to _load_unrolled.
branch=$(git -C "$RRT_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null)
[ "$branch" = "updates" ] \
    && ok "radio_reconstruction_tools on 'updates'" \
    || bad "radio_reconstruction_tools is on '$branch', needs 'updates'"

# radio_tools resolves a GPLv3 submodule at runtime; without it the NUFFT
# backends raise ModuleNotFoundError: No module named 'pysrc'.
[ -n "$(ls -A "$RADIO_TOOLS_DIR/extern/RI-measurement-operator/pysrc" 2>/dev/null)" ] \
    && ok "radio_tools submodule initialised" \
    || bad "radio_tools submodule empty: git -C $RADIO_TOOLS_DIR submodule update --init --recursive"

cd "$CODE_REPO" || { bad "cannot cd to $CODE_REPO"; exit 1; }

python -c "import coverage_plots, sys; sys.exit(0 if hasattr(coverage_plots,'TARP') else 1)" 2>/dev/null \
    && ok "coverage_plots has the streaming TARP" \
    || bad "coverage_plots lacks TARP -- pre-campaign version; the diagnostics will fail at use"

# ---------------------------------------------------------------------------
step "2. data and checkpoints"
# ---------------------------------------------------------------------------
# The hours come from the rbn allocation but the checkpoints live under the ney
# project tree. Those are independent; read access to both is required.
if python - <<'PY'
import sys, pathlib
sys.path.insert(0, "scripts")
import problems
bad = 0
for key in problems.available():
    family, size = key.rsplit("/", 1)
    try:
        run = problems.resolve(family, int(size))
        print(f"   ok    {key:24s} {run.model_type:16s} {problems.find_checkpoint(run).name}")
    except Exception as exc:
        print(f"   FAIL  {key:24s} {type(exc).__name__}: {exc}", file=sys.stderr)
        bad += 1
images = problems._roots(problems.load_registry())["images"]
n = len(list(images.glob("*.npy"))) if images.is_dir() else 0
print(f"   {'ok   ' if n else 'FAIL '} {n} images in {images}")
sys.exit(1 if (bad or not n) else 0)
PY
then ok "all runs resolve"; else bad "one or more runs did not resolve (see above)"; fi

# ---------------------------------------------------------------------------
step "3. DRUNet weights (THE blocker)"
# ---------------------------------------------------------------------------
# Compute nodes have no network, and both unrolled runs load DRUNet with
# `path_denoiser: null`. Without this, those jobs die minutes in.
mkdir -p "$TORCH_HOME"
if python "$RRT_DIR/scripts/prefetch_drunet.py" >/dev/null 2>&1; then
    ok "DRUNet cached under $TORCH_HOME"
else
    bad "prefetch_drunet.py failed. GPU nodes cannot download; the unrolled jobs will die."
fi

# ---------------------------------------------------------------------------
step "4. test suite and the no-checkpoint example"
# ---------------------------------------------------------------------------
python -m pytest tests -q >/tmp/uq_pytest.log 2>&1 \
    && ok "$(tail -1 /tmp/uq_pytest.log)" \
    || bad "pytest failed; see /tmp/uq_pytest.log"

python examples/radio/run_bootstrap_rcps.py \
    -c examples/radio/configs/smoke.yaml --device cpu >/tmp/uq_smoke.log 2>&1 \
    && ok "radio example ran end to end" \
    || bad "radio example failed; see /tmp/uq_smoke.log"

# ---------------------------------------------------------------------------
step "5. dry runs -- measured cost on this hardware"
# ---------------------------------------------------------------------------
# The recorded anchors are laptop CPU numbers and do not transfer. This prints
# the real per-draw cost and the projected hours per arm-seed, which is what the
# wall-times in the job scripts should be checked against.
for config in "$CAMPAIGN_DIR/configs/coverage_64_gpu.yaml" \
              "$CAMPAIGN_DIR/configs/campaign_64.yaml" \
              "$CAMPAIGN_DIR/configs/campaign_360.yaml"; do
    echo "   --- $(basename "$config")"
    python scripts/uq_campaign.py -c "$config" --dry-run --device cpu --allow-cpu \
        2>/dev/null | sed 's/^/   /' | tail -4
done

# ---------------------------------------------------------------------------
step "summary"
# ---------------------------------------------------------------------------
if [ "$FAILURES" -ne 0 ]; then
    printf '\n\033[1m%d check(s) failed. Fix them before submitting.\033[0m\n' "$FAILURES" >&2
    exit 1
fi

cat <<EOF

   All checks passed. Submit in this order (or run ./submit_all.sh):

     cd $CAMPAIGN_DIR/jobs
     sbatch mc_ladder_64.sh            # ~1 h; decides how to read coverage_64_gpu
     sbatch coverage_64_gpu.sh         # raw coverage, no conformalisation
     sbatch level_b.sh                 # minutes; run and READ this first
     sbatch campaign_64.sh             # the screen: which family, which kappa
     sbatch campaign_360_ar.sh         # confirmation, cheap reconstructor
     sbatch campaign_360_unrolled.sh   # confirmation, expensive one

   All four self-chain until every row has an output JSON, and skip rows already
   done. To stop one early:

     touch \$OUTPUT_ROOT/campaign64/.COMPLETE     # and scancel the successor

   Results land in \$OUTPUT_ROOT/<campaign>/<family>/<arm>_seed<k>.json, each
   with a cached calibration set beside it -- so RCPS, the coverage curves and
   any metric invented later can be recomputed on a laptop with no GPU.

   OUTPUT_ROOT = $RADIO_ROOT/outputs/uq_campaign
EOF
