#!/bin/bash
#SBATCH --job-name=rrt_bench_img360_psf_4gpu_a100
#SBATCH --mail-user=tobiasliaudat@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --ntasks=4                   # 1 tache MPI par GPU pour l'etape DDP
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8            # coeurs CPU par tache (un huitieme d'un noeud A100: 64/8)
#SBATCH -C a100                      # partition A100 (gpu_p5)
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00              # dev queue max; le benchmark s'arrete a 60 min
#SBATCH --qos=qos_gpu_a100-dev       # dev queue A100 (<= 2h, scheduling rapide)
#SBATCH --output=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/submission-scripts/jean_zay/raw/radio_reconstruction_tools/benchmark/jobs/logs/R-%x_%A_%a.out
#SBATCH --error=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/submission-scripts/jean_zay/raw/radio_reconstruction_tools/benchmark/jobs/logs/R-%x_%A_%a.err
#SBATCH -A ney@a100                  # allocation A100 (radio reconstruction, for now)

# Self-contained img360 timing benchmark (psf, 4-GPU DDP), sized for the dev
# queue. Step 1 builds its own SMALL pinned condition (16 coverages at 360^2)
# single-process — required because under DDP non-zero ranks only wait ~30 min
# for a missing condition. Step 2 runs the real 4-GPU benchmark. Per-step cost
# is coverage-count independent, so the estimates transfer directly to the
# 1000-coverage large run (whose own condition the large builder job creates).

set -x

CODE_REPO=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/radio_reconstruction_tools
SUBMISSION_REPO=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/submission-scripts
CONFIG=$SUBMISSION_REPO/jean_zay/raw/radio_reconstruction_tools/benchmark/configs/bench_img360_PSF_finufft.yaml
RESULTS_DIR=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/outputs/benchmarks

# --- Environment (A100: arch/a100 pre-module before pytorch) ---------------
. $SUBMISSION_REPO/jean_zay/env_configs/radio_reconstruction_tools_a100.sh
export TMPDIR=$JOBSCRATCH
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_DUPLICATE_LIB_OK=TRUE
export WANDB_MODE=offline

mkdir -p "$RESULTS_DIR"
cd $CODE_REPO

# Step 1: single-process build of the small pinned condition (idempotent).
srun --ntasks=1 --ntasks-per-node=1 python -u scripts/build_artifacts.py \
    --config $CONFIG --device cuda || exit 1

# Step 2: 4-GPU DDP timing benchmark (capped at 60 min; per-step stats survive
# an early stop).
srun --ntasks=4 --ntasks-per-node=4 python -u scripts/benchmark/benchmark_training.py \
    --config $CONFIG \
    --devices 4 \
    --strategy ddp \
    --epochs 2 \
    --max-duration-min 60 \
    --label bench_img360_psf_4gpu \
    --output-json $RESULTS_DIR/bench_img360_psf_4gpu.json
