#!/bin/bash
#SBATCH --job-name=rrt_smoke_unrolled_DRUNet_finufft_h100    # nom du job
#SBATCH --mail-user=tobiasliaudat@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --ntasks=1                   # nombre total de tache MPI (= nombre total de GPU)
#SBATCH --ntasks-per-node=1          # nombre de tache MPI par noeud (= nombre de GPU par noeud)
#SBATCH --gres=gpu:1                 # nombre de GPU
#SBATCH --cpus-per-task=24           # coeurs CPU par tache (un quart d'un noeud H100: 96/4)
#SBATCH -C h100                      # partition H100 (gpu_p6)
#SBATCH --hint=nomultithread         # hyperthreading desactive
#SBATCH --time=02:00:00              # smoke test: court
#SBATCH --qos=qos_gpu_h100-dev       # dev queue H100 (<= 2h, scheduling rapide)
#SBATCH --output=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/submission-scripts/jean_zay/raw/radio_reconstruction_tools/jobs/logs/R-%x_%A_%a.out
#SBATCH --error=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/submission-scripts/jean_zay/raw/radio_reconstruction_tools/jobs/logs/R-%x_%A_%a.err
#SBATCH -A ney@h100                  # allocation H100 (radio reconstruction, for now)

set -x

# --- Paths -----------------------------------------------------------------
CODE_REPO=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/radio_reconstruction_tools
SUBMISSION_REPO=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/submission-scripts
CONFIG=$SUBMISSION_REPO/jean_zay/raw/radio_reconstruction_tools/configs/unrolled_n10_prior_DRUNet_op_finufft_smoketest.yaml
OUTPUT_ROOT=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/outputs/trained_models

# Sequential mode mints its own timestamped run directory under $OUTPUT_ROOT
#   $OUTPUT_ROOT/<run_name>_smoketest_h100_<YYYYmmdd_HHMMSS>/stage_NNlayers/
# and auto-resumes unfinished stages, so we pass a STABLE --log-dir/--run-name
# (no pre-computed timestamp) to let that resume logic find the run.
RUN_NAME=unrolled_n10_prior_DRUNet_op_finufft

# --- Environment (H100: arch/h100 pre-module before pytorch) ---------------
. $SUBMISSION_REPO/jean_zay/env_configs/radio_reconstruction_tools_h100.sh
export TMPDIR=$JOBSCRATCH
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_DUPLICATE_LIB_OK=TRUE
export WANDB_MODE=offline            # no internet on compute nodes

mkdir -p "$OUTPUT_ROOT"
cd $CODE_REPO

# Per-stage checkpoints, metrics.csv and the resolved config land under
# $OUTPUT_ROOT/${RUN_NAME}_smoketest_h100_<ts>/; the shared offline wandb run
# under $OUTPUT_ROOT/wandb/.
srun python -u scripts/train_unrolled_cluster.py \
    --config $CONFIG \
    --device cuda \
    --log-dir $OUTPUT_ROOT \
    --run-name ${RUN_NAME}_smoketest_h100
