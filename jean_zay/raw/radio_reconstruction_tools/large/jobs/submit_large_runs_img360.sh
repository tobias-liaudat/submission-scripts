#!/bin/bash
# ===========================================================================
# Launch the two img360 large runs in the right order (run on a login node,
# NOT with sbatch — this script only calls sbatch).
#
#   ./submit_large_runs.sh
#
# Stage 1: the single-GPU artifacts builder pre-builds BOTH pinned conditions
#          (1000-coverage UV bank, splits, op-norms, PSF bank) and verifies the
#          two dirs pin the identical bank/splits. This must finish before any
#          4-GPU launch: DDP ranks only wait ~30 min for a missing condition.
# Stage 2 (afterok:builder): the two self-chaining 4-GPU training chains
#          (tkbn and PSF/finufft). Each chain resubmits itself until training
#          completes (.TRAINING_COMPLETE) — expect ~1 link for PSF and several
#          20h links for tkbn (see outputs/benchmarks/).
#
# If the builder fails, the chains stay pending with DependencyNeverSatisfied —
# scancel them, fix, and rerun this script.
# ===========================================================================

set -euo pipefail

JOBS_DIR=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/submission-scripts/jean_zay/raw/radio_reconstruction_tools/large/jobs

BUILDER=$JOBS_DIR/build_artifacts_large_img360_a100.sh
CHAINS=(
    "$JOBS_DIR/large_run_img360_unrolled_n15_prior_DRUNet_PSF_finufft_4gpus_h100.sh"
)

builder_id=$(sbatch --parsable "$BUILDER")
echo "[large] artifacts builder submitted: job $builder_id ($(basename "$BUILDER"))"

for job in "${CHAINS[@]}"; do
    chain_id=$(sbatch --parsable --dependency=afterok:"$builder_id" "$job")
    echo "[large] training chain submitted: job $chain_id ($(basename "$job")) [afterok:$builder_id]"
done

echo "[large] all submitted. Monitor with: squeue -u \$USER"
