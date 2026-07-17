#!/bin/bash
#SBATCH --job-name=rrt_bench15_finufft_4gpu_a100
#SBATCH --mail-user=tobiasliaudat@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --ntasks=4                   # 1 tache MPI par GPU (DDP: doit = --devices)
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8            # coeurs CPU par tache (un huitieme d'un noeud A100: 64/8)
#SBATCH -C a100                      # partition A100 (gpu_p5)
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00              # dev queue max; le benchmark s arrete a 90 min
#SBATCH --qos=qos_gpu_a100-dev       # dev queue A100 (<= 2h, scheduling rapide)
#SBATCH --output=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/submission-scripts/jean_zay/raw/radio_reconstruction_tools/benchmark/jobs/logs/R-%x_%A_%a.out
#SBATCH --error=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/submission-scripts/jean_zay/raw/radio_reconstruction_tools/benchmark/jobs/logs/R-%x_%A_%a.err
#SBATCH -A ney@a100                  # allocation A100 (radio reconstruction, for now)

# Timing benchmark: 15-layer finufft training on 4 A100 GPU(s), a few real
# epochs + extrapolation to full fixed-depth and sequential trainings. Results
# land as JSON in $RESULTS_DIR (summarise with summarize_benchmarks.py).

set -x

CODE_REPO=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/radio_reconstruction_tools
SUBMISSION_REPO=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/submission-scripts
CONFIG=$SUBMISSION_REPO/jean_zay/raw/radio_reconstruction_tools/benchmark/configs/bench15_unrolled_finufft.yaml
RESULTS_DIR=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/outputs/benchmarks

# --- Environment (A100: arch/a100 pre-module before pytorch) ---------------
. $SUBMISSION_REPO/jean_zay/env_configs/radio_reconstruction_tools_a100.sh
export TMPDIR=$JOBSCRATCH
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_DUPLICATE_LIB_OK=TRUE
export WANDB_MODE=offline

mkdir -p "$RESULTS_DIR"
cd $CODE_REPO

srun python -u scripts/benchmark/benchmark_training.py \
    --config $CONFIG \
    --devices 4 \
    --strategy ddp \
    --epochs 3 \
    --max-duration-min 90 \
    --label bench15_finufft_4gpu \
    --output-json $RESULTS_DIR/bench15_finufft_4gpu.json
