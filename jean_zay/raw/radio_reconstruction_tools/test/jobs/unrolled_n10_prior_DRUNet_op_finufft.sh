#!/bin/bash
#SBATCH --job-name=rrt_unrolled_DRUNet_finufft    # nom du job
#SBATCH --mail-user=tobiasliaudat@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --ntasks=1                   # nombre total de tache MPI (= nombre total de GPU)
#SBATCH --ntasks-per-node=1          # nombre de tache MPI par noeud (= nombre de GPU par noeud)
#SBATCH --gres=gpu:1                 # nombre de GPU
#SBATCH --cpus-per-task=10           # coeurs CPU par tache (un quart d'un noeud V100)
#SBATCH -C v100-32g
#SBATCH --hint=nomultithread         # hyperthreading desactive
#SBATCH --time=20:00:00              # max de la QoS par defaut (qos_gpu-t3)
#SBATCH --output=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/submission-scripts/jean_zay/raw/radio_reconstruction_tools/jobs/logs/R-%x_%A_%a.out
#SBATCH --error=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/submission-scripts/jean_zay/raw/radio_reconstruction_tools/jobs/logs/R-%x_%A_%a.err
#SBATCH -A ney@v100                  # allocation (radio reconstruction, for now)

# ===========================================================================
# Self-chaining training job.
#
# Training can exceed the 20h QoS wall-time, so this job AUTO-RESUBMITS itself
# until training is complete, each relaunch resuming from the last checkpoint
# (sequential mode auto-resumes finished/interrupted stages from STAGE_DONE +
# last.ckpt).
#
# How it works:
#   * A RUN_ID identifies one training chain. The first (manual) submission
#     generates it; each auto-resubmitted link inherits the SAME RUN_ID as $1,
#     so every link shares the run directory and resumes the previous one.
#   * The successor is submitted with --dependency=afterany BEFORE training, so
#     the chain survives even a hard wall-time kill (SIGKILL leaves no chance to
#     resubmit afterwards).
#   * On a clean finish the job drops a .TRAINING_COMPLETE marker; the queued
#     successor sees it and exits immediately, ending the chain.
#   * A resubmit cap guards against a non-progressing infinite loop.
#
# Launch (from the logs dir, as usual):
#   cd .../jobs/logs && sbatch ../unrolled_n10_prior_DRUNet_op_finufft.sh
# ===========================================================================

set -x

# --- Paths -----------------------------------------------------------------
CODE_REPO=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/radio_reconstruction_tools
SUBMISSION_REPO=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/submission-scripts
JOB_SCRIPT=$SUBMISSION_REPO/jean_zay/raw/radio_reconstruction_tools/jobs/unrolled_n10_prior_DRUNet_op_finufft.sh
CONFIG=$SUBMISSION_REPO/jean_zay/raw/radio_reconstruction_tools/configs/unrolled_n10_prior_DRUNet_op_finufft.yaml
OUTPUT_ROOT=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/outputs/trained_models

# --- Chain identity / stop conditions --------------------------------------
# RUN_ID = stable run-name shared by all links of this chain. First submission
# mints it (with a timestamp); resubmissions inherit it via $1.
BASE_NAME=unrolled_n10_prior_DRUNet_op_finufft
RUN_ID="${1:-${BASE_NAME}_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="$OUTPUT_ROOT/$RUN_ID"
COMPLETE_MARKER="$RUN_DIR/.TRAINING_COMPLETE"
RESUBMIT_COUNT_FILE="$RUN_DIR/.resubmit_count"
MAX_RESUBMITS=20

mkdir -p "$RUN_DIR"

# Stop the chain if a previous link already finished training.
if [ -f "$COMPLETE_MARKER" ]; then
    echo "[chain] $RUN_ID already complete ($COMPLETE_MARKER); nothing to do."
    exit 0
fi

# Stop the chain if it has resubmitted too many times without finishing.
count=$(cat "$RESUBMIT_COUNT_FILE" 2>/dev/null || echo 0)
if [ "$count" -ge "$MAX_RESUBMITS" ]; then
    echo "[chain] reached MAX_RESUBMITS=$MAX_RESUBMITS for $RUN_ID; stopping. " \
         "Inspect the logs (training is not progressing) before relaunching."
    exit 1
fi
echo $((count + 1)) > "$RESUBMIT_COUNT_FILE"

# Schedule the successor NOW (before training) so the chain continues even if
# this job is hard-killed at the wall-time limit. It runs after this job ends
# for ANY reason and no-ops if this job completes training.
sbatch --dependency=afterany:"$SLURM_JOB_ID" "$JOB_SCRIPT" "$RUN_ID"

# --- Environment -----------------------------------------------------------
. $SUBMISSION_REPO/jean_zay/env_configs/radio_reconstruction_tools.sh
export TMPDIR=$JOBSCRATCH
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_DUPLICATE_LIB_OK=TRUE
export WANDB_MODE=offline            # no internet on compute nodes

cd $CODE_REPO

# Sequential mode reuses $OUTPUT_ROOT/$RUN_ID and auto-resumes its stages. Stages,
# metrics.csv and resolved_config land under $RUN_DIR; the offline wandb run under
# $OUTPUT_ROOT/wandb (sync later from a login node with `wandb sync`).
srun python -u scripts/train_unrolled_cluster.py \
    --config "$CONFIG" \
    --device cuda \
    --log-dir "$OUTPUT_ROOT" \
    --run-name "$RUN_ID"
status=$?

# Mark complete only on a clean finish (a wall-time kill never reaches here, so
# the queued successor will resume instead).
if [ "$status" -eq 0 ]; then
    touch "$COMPLETE_MARKER"
    echo "[chain] $RUN_ID training finished cleanly; COMPLETE marker dropped."
fi
exit $status
