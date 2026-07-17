#!/bin/bash
# ===========================================================================
# Re-submit ALL smoke-test jobs at once, with no dependency ordering (run on a
# login node, NOT with sbatch — this script only calls sbatch).
#
#   ./submit_smoketests_rerun.sh
#
# Use this once the shared artifacts dir
# (.../outputs/artifacts/smoketest_64px_cov4_ntrain16) has been populated by a
# previous campaign (see submit_smoketests.sh for the initial, ordered launch):
# every job only reads the stored UV bank / image splits, so they can all run
# concurrently. Config changes that alter the derived-cache stamp (e.g.
# imaging_weights) make each job recompute op_norms.pt/psf_bank.pt from the
# stored bank — the writes are atomic (a concurrent rerun is safe, just
# redundant) and the pinned condition itself is never rewritten.
# ===========================================================================

set -euo pipefail

JOBS_DIR=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/submission-scripts/jean_zay/raw/radio_reconstruction_tools/jobs

JOBS=(
    "$JOBS_DIR/unrolled_n10_prior_DRUNet_op_finufft_smoketest_v100.sh"
    "$JOBS_DIR/unrolled_n10_prior_DRUNet_op_tkbn_smoketest_v100.sh"
    "$JOBS_DIR/unrolled_psf_n10_prior_DRUNet_op_finufft_smoketest_v100.sh"
    "$JOBS_DIR/unrolled_n10_prior_DRUNet_op_finufft_smoketest_a100.sh"
    "$JOBS_DIR/unrolled_psf_n10_prior_DRUNet_op_finufft_smoketest_a100.sh"
    "$JOBS_DIR/unrolled_n10_prior_DRUNet_op_finufft_smoketest_h100.sh"
    "$JOBS_DIR/unrolled_n10_prior_DRUNet_op_finufft_smoketest_multigpu_a100.sh"
)

for job in "${JOBS[@]}"; do
    job_id=$(sbatch --parsable "$job")
    echo "[smoketests-rerun] submitted: job $job_id ($(basename "$job"))"
done

echo "[smoketests-rerun] all ${#JOBS[@]} jobs submitted. Monitor with: squeue -u \$USER"
