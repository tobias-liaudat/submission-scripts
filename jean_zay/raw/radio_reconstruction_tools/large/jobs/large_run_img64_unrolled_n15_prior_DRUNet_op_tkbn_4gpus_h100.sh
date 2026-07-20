#!/bin/bash
#SBATCH --job-name=rrt_large_img64_tkbn_4gpu
#SBATCH --mail-user=tobiasliaudat@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --ntasks=4                   # 1 tache MPI par GPU (DDP: doit = runtime.devices)
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4                 # un noeud H100 complet (4 GPU/noeud)
#SBATCH --cpus-per-task=24           # coeurs CPU par tache (un quart de noeud H100: 96/4)
#SBATCH -C h100                      # partition H100 (gpu_p6)
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00              # max QoS H100 (qos_gpu_h100-t3); chained for longer runs
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --output=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/submission-scripts/jean_zay/raw/radio_reconstruction_tools/large/jobs/logs/R-%x_%A_%a.out
#SBATCH --error=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/submission-scripts/jean_zay/raw/radio_reconstruction_tools/large/jobs/logs/R-%x_%A_%a.err
#SBATCH -A ney@h100                  # allocation H100 (radio reconstruction, for now)

# ===========================================================================
# Self-chaining 4-GPU (DDP) H100 training job for the img64 large run (tkbn).
# The H100 QoS caps wall-time at 20h, so this job resubmits itself with
# --dependency=afterany BEFORE training (survives a hard wall-time kill); each
# relaunch resumes from the last checkpoint (sequential mode auto-resumes its
# stages). A .TRAINING_COMPLETE marker stops the chain on clean finish; a
# resubmit cap guards against a non-progressing loop.
# PREREQUISITE: the pinned artifacts dir must exist (run the builder job
# first — submit_large_runs.sh chains it automatically).
# Launch:  cd .../large/jobs/logs && sbatch ../large_run_img64_unrolled_n15_prior_DRUNet_op_tkbn_4gpus_h100.sh
# ===========================================================================

set -uo pipefail

CODE_REPO=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/radio_reconstruction_tools
SUBMISSION_REPO=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/submission-scripts
JOB_SCRIPT=$SUBMISSION_REPO/jean_zay/raw/radio_reconstruction_tools/large/jobs/large_run_img64_unrolled_n15_prior_DRUNet_op_tkbn_4gpus_h100.sh
CONFIG=$SUBMISSION_REPO/jean_zay/raw/radio_reconstruction_tools/large/configs/large_run_img64_unrolled_n15_prior_DRUNet_op_tkbn_4gpus.yaml
OUTPUT_ROOT=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/outputs/trained_models

BASE_NAME=large_run_img64_unrolled_n15_prior_DRUNet_op_tkbn_4gpus
RUN_ID="${1:-${BASE_NAME}_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="$OUTPUT_ROOT/$RUN_ID"
COMPLETE_MARKER="$RUN_DIR/.TRAINING_COMPLETE"
RESUBMIT_COUNT_FILE="$RUN_DIR/.resubmit_count"
MAX_RESUBMITS=15

mkdir -p "$RUN_DIR"

if [ -f "$COMPLETE_MARKER" ]; then
    echo "[chain] $RUN_ID already complete; nothing to do."
    exit 0
fi
count="$(cat "$RESUBMIT_COUNT_FILE" 2>/dev/null || echo 0)"
if [ "$count" -ge "$MAX_RESUBMITS" ]; then
    echo "[chain] reached MAX_RESUBMITS=$MAX_RESUBMITS for $RUN_ID; stopping."
    exit 1
fi
echo $((count + 1)) > "$RESUBMIT_COUNT_FILE"

# Schedule the successor BEFORE training so the chain survives a hard kill.
sbatch --dependency=afterany:"$SLURM_JOB_ID" "$JOB_SCRIPT" "$RUN_ID"

# --- Environment (H100: arch/h100 pre-module before pytorch) ---------------
. $SUBMISSION_REPO/jean_zay/env_configs/radio_reconstruction_tools_h100.sh
export TMPDIR=$JOBSCRATCH
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_DUPLICATE_LIB_OK=TRUE
export WANDB_MODE=offline

set -x
cd $CODE_REPO
status=0
srun python -u scripts/train_unrolled_cluster.py \
    --config "$CONFIG" \
    --device cuda \
    --log-dir "$OUTPUT_ROOT" \
    --run-name "$RUN_ID" || status=$?
set +x

if [ "$status" -eq 0 ]; then
    touch "$COMPLETE_MARKER"
    echo "[chain] $RUN_ID training finished cleanly; COMPLETE marker dropped."
fi
exit "$status"
