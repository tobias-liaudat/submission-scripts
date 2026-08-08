#!/bin/bash
# ===========================================================================
# Launch the three weighted_loss=false large runs (run on a login node, NOT
# with sbatch — this script only calls sbatch).
#
#   ./submit_large_runs_wloss_false.sh
#
# These are ablations of the existing weighted-loss campaign: identical nets,
# data, UV banks, splits and schedules, with model.weighted_loss flipped to
# false (each sequential stage supervised on its final layer alone). The img64
# op/finufft chain additionally swaps the PSF data-fidelity gradient for the
# explicit NUFFT one on the finufft backend.
#
# There is NO builder stage: every chain REUSES a pinned artifacts dir that the
# weighted campaign already built, and reuse is read-only (the derived-cache
# stamp — backend finufft, imaging_weights briggs — matches what these configs
# request, so op-norms/PSF banks are loaded, never recomputed). That also means
# these chains may run concurrently with the weighted ones sharing those dirs.
# The check below fails fast if a condition is missing rather than letting a
# 4-GPU job burn its ~30 min DDP wait and die.
#
# Each chain resubmits itself until training completes (.TRAINING_COMPLETE).
# Runs land in outputs/trained_models/<BASE_NAME>_<timestamp>/ — every name
# carries the _wloss_false token so these models are never confused with the
# weighted ones.
# ===========================================================================

set -euo pipefail

JOBS_DIR=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/submission-scripts/jean_zay/raw/radio_reconstruction_tools/large/jobs
ARTIFACTS_ROOT=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/outputs/artifacts

CHAINS=(
    "$JOBS_DIR/large_run_img64_unrolled_n15_wloss_false_prior_DRUNet_PSF_finufft_4gpus_a100.sh"
    "$JOBS_DIR/large_run_img64_unrolled_n15_wloss_false_prior_DRUNet_op_finufft_4gpus_a100.sh"
    "$JOBS_DIR/large_run_img360_unrolled_n15_wloss_false_prior_DRUNet_PSF_finufft_4gpus_a100.sh"
)

# Pinned conditions the chains reuse (img64 op/finufft shares the img64 PSF dir).
CONDITIONS=(
    "$ARTIFACTS_ROOT/large_img64_cov1000_ntrain4000_psf_finufft"
    "$ARTIFACTS_ROOT/large_img360_cov1000_ntrain4000_psf_finufft"
)

for cond in "${CONDITIONS[@]}"; do
    for f in uv_bank.pt splits.json provenance.json op_norms.pt psf_bank.pt; do
        if [ ! -f "$cond/$f" ]; then
            echo "[wloss_false] MISSING $cond/$f — the pinned condition is not" \
                 "complete. Build it with the matching build_artifacts_large_*" \
                 "job before submitting." >&2
            exit 1
        fi
    done
    echo "[wloss_false] condition OK: $cond"
done

for job in "${CHAINS[@]}"; do
    chain_id=$(sbatch --parsable "$job")
    echo "[wloss_false] training chain submitted: job $chain_id ($(basename "$job"))"
done

echo "[wloss_false] all submitted. Monitor with: squeue -u \$USER"
