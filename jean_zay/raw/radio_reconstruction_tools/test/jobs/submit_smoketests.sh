#!/bin/bash
# ===========================================================================
# Submit the full smoke-test campaign in the right order (run on a login node,
# NOT with sbatch — this script only calls sbatch).
#
#   ./submit_smoketests.sh
#
# Stage 1  (creator): the single-GPU finufft V100 smoke job runs alone and
#          populates the shared artifacts dir
#          (.../outputs/artifacts/smoketest_64px_cov4_ntrain16: uv_bank.pt,
#          splits.json, op_norms.pt, provenance.json).
# Stage 2  (afterok:creator): every other smoke job — tkbn V100, PSF V100,
#          finufft A100, PSF A100, finufft H100 and the 2-GPU DDP A100 job —
#          reuses the stored condition, so all backends/machines are exercised
#          on identical coverages/images. Submitting them concurrently is safe
#          once the artifacts exist (they only read the bank; backend changes
#          merely recompute the derived caches with a warning).
#
# If the creator fails, the dependents stay pending with DependencyNeverSatisfied
# — scancel them, fix, and rerun this script.
# ===========================================================================

set -euo pipefail

JOBS_DIR=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/submission-scripts/jean_zay/raw/radio_reconstruction_tools/jobs

CREATOR=$JOBS_DIR/unrolled_n10_prior_DRUNet_op_finufft_smoketest_v100.sh
DEPENDENTS=(
    "$JOBS_DIR/unrolled_n10_prior_DRUNet_op_tkbn_smoketest_v100.sh"
    "$JOBS_DIR/unrolled_psf_n10_prior_DRUNet_op_finufft_smoketest_v100.sh"
    "$JOBS_DIR/unrolled_n10_prior_DRUNet_op_finufft_smoketest_a100.sh"
    "$JOBS_DIR/unrolled_psf_n10_prior_DRUNet_op_finufft_smoketest_a100.sh"
    "$JOBS_DIR/unrolled_n10_prior_DRUNet_op_finufft_smoketest_h100.sh"
    "$JOBS_DIR/unrolled_n10_prior_DRUNet_op_finufft_smoketest_multigpu_a100.sh"
)

creator_id=$(sbatch --parsable "$CREATOR")
echo "[smoketests] creator (artifact builder) submitted: job $creator_id ($(basename "$CREATOR"))"

for job in "${DEPENDENTS[@]}"; do
    dep_id=$(sbatch --parsable --dependency=afterok:"$creator_id" "$job")
    echo "[smoketests] dependent submitted: job $dep_id ($(basename "$job")) [afterok:$creator_id]"
done

echo "[smoketests] all submitted. Monitor with: squeue -u \$USER"
