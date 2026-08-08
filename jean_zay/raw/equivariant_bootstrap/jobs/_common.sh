#!/bin/bash
# Shared setup for the equivariant-bootstrap UQ campaign. Sourced, never run.
#
# Everything a submitter needs to change lives in the header of the individual
# job script; this file holds only what is identical across all of them.
#
# Expects the caller to have set GPU_ARCH (v100 | a100 | h100) and
# SUBMISSION_REPO before sourcing.

set -uo pipefail

RADIO_ROOT=${RADIO_ROOT:-/lustre/fswork/projects/rech/ney/ulx23va/projects/radio}
CODE_REPO=${CODE_REPO:-$RADIO_ROOT/repos/UQsuite}
SUBMISSION_REPO=${SUBMISSION_REPO:-$RADIO_ROOT/repos/submission-scripts}
CAMPAIGN_DIR=$SUBMISSION_REPO/jean_zay/raw/equivariant_bootstrap

# --- Environment ----------------------------------------------------------
# No conda on Jean Zay: the packages live in the PyTorch module's Python.
# `carb_v2` is the laptop environment and must never appear in a job script.
case "${GPU_ARCH:-v100}" in
    a100) . "$SUBMISSION_REPO/jean_zay/env_configs/equivariant_bootstrap_a100.sh" ;;
    h100) . "$SUBMISSION_REPO/jean_zay/env_configs/equivariant_bootstrap_h100.sh" ;;
    *)    . "$SUBMISSION_REPO/jean_zay/env_configs/equivariant_bootstrap.sh" ;;
esac

# Caches on $WORK: $HOME is quota-limited, and -- the reason this matters --
# compute nodes have NO NETWORK. Both unrolled runs load DRUNet with
# `path_denoiser: null`, so without a pre-populated TORCH_HOME they try to
# download it and die minutes in. `prepare_cluster.sh` fills it on a login node.
export TORCH_HOME=${TORCH_HOME:-$WORK/.cache/torch}
export HF_HOME=${HF_HOME:-$WORK/.cache/huggingface}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-$WORK/.cache}
export TMPDIR=${JOBSCRATCH:-/tmp}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export KMP_DUPLICATE_LIB_OK=TRUE

# --- Where the data is ----------------------------------------------------
# The checkpoints live under the `ney` project tree while the compute hours come
# from `rbn`; those are independent, and read access to both is required.
# `UQsuite/scripts/configs/runs.yaml` reads these three and nothing else, which
# is what lets one registry serve both the laptop and the cluster.
export UQ_MODELS_DIR=${UQ_MODELS_DIR:-$RADIO_ROOT/outputs/trained_models}
export UQ_ARTIFACTS_DIR=${UQ_ARTIFACTS_DIR:-$RADIO_ROOT/outputs/artifacts}
export UQ_IMAGES_DIR=${UQ_IMAGES_DIR:-$RADIO_ROOT/data/images}

export OUTPUT_ROOT=${OUTPUT_ROOT:-$RADIO_ROOT/outputs/uq_campaign}

cd "$CODE_REPO"

# --- Self-chaining --------------------------------------------------------
# The successor is queued BEFORE the work starts, with `afterany`, so a
# wall-time kill still hands over. `--skip-existing` on the runner makes the
# successor resume rather than repeat; together they survive a week unattended.
#
#   .COMPLETE        written by the last link; stops the chain
#   .resubmit_count  capped, so a job that fails instantly cannot loop forever
MAX_RESUBMITS=${MAX_RESUBMITS:-20}

chain_or_stop() {
    local state_dir="$1" job_script="$2"
    mkdir -p "$state_dir"
    if [ -f "$state_dir/.COMPLETE" ]; then
        echo "[chain] $state_dir/.COMPLETE exists; nothing to do."
        exit 0
    fi
    local count
    count="$(cat "$state_dir/.resubmit_count" 2>/dev/null || echo 0)"
    if [ "$count" -ge "$MAX_RESUBMITS" ]; then
        echo "[chain] reached MAX_RESUBMITS=$MAX_RESUBMITS; stopping." >&2
        exit 1
    fi
    echo $((count + 1)) > "$state_dir/.resubmit_count"
    sbatch --dependency=afterany:"$SLURM_JOB_ID" "$job_script"
}

mark_complete() {
    touch "$1/.COMPLETE"
    echo "[chain] campaign finished; wrote $1/.COMPLETE"
}

# Rows for one reconstructor family, read from the campaign config so the job
# script never restates the arm list.
rows_for_family() {
    local config="$1" family="$2"
    python - "$config" "$family" <<'PY'
import sys, yaml
config, family = sys.argv[1], sys.argv[2]
cfg = yaml.safe_load(open(config))
print(" ".join(f"{family}/{arm}/{seed}"
               for arm in cfg["arms"] for seed in cfg["seeds"]))
PY
}
