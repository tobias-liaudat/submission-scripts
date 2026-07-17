#!/bin/bash
# ===========================================================================
# Submit the full 15-layer training benchmark campaign (run on a login node,
# NOT with sbatch — this script only calls sbatch).
#
#   ./submit_benchmarks.sh
#
# 6 jobs on the A100 dev queue: {finufft, tkbn, psf} x {1 GPU, 4-GPU DDP}.
# The finufft 1-GPU job runs first as the CREATOR of the shared benchmark
# artifacts dir (.../outputs/artifacts/bench15_64px_cov16_ntrain2048); the
# other five follow with afterok so every variant is timed on the identical
# pinned condition. Results land as JSON in .../outputs/benchmarks/;
# summarise with:
#   python scripts/benchmark/summarize_benchmarks.py \
#       /lustre/fswork/projects/rech/ney/ulx23va/projects/radio/outputs/benchmarks
# ===========================================================================

set -euo pipefail

JOBS_DIR=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/submission-scripts/jean_zay/raw/radio_reconstruction_tools/benchmark/jobs

CREATOR=$JOBS_DIR/bench15_finufft_1gpu_a100.sh
DEPENDENTS=(
    "$JOBS_DIR/bench15_finufft_4gpu_a100.sh"
    "$JOBS_DIR/bench15_tkbn_1gpu_a100.sh"
    "$JOBS_DIR/bench15_tkbn_4gpu_a100.sh"
    "$JOBS_DIR/bench15_psf_1gpu_a100.sh"
    "$JOBS_DIR/bench15_psf_4gpu_a100.sh"
)

creator_id=$(sbatch --parsable "$CREATOR")
echo "[bench] creator (artifact builder) submitted: job $creator_id ($(basename "$CREATOR"))"

for job in "${DEPENDENTS[@]}"; do
    dep_id=$(sbatch --parsable --dependency=afterok:"$creator_id" "$job")
    echo "[bench] dependent submitted: job $dep_id ($(basename "$job")) [afterok:$creator_id]"
done

echo "[bench] all submitted. Monitor with: squeue -u \$USER"
