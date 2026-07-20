#!/bin/bash
#SBATCH --job-name=rrt_validate_large_img360_tkbn
#SBATCH --mail-user=tobiasliaudat@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24           # coeurs CPU par tache (un quart d'un noeud H100: 96/4)
#SBATCH -C h100                      # partition H100 (gpu_p6)
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --qos=qos_gpu_h100-dev       # dev queue H100 (<= 2h, scheduling rapide)
#SBATCH --output=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/submission-scripts/jean_zay/raw/radio_reconstruction_tools/validation/jobs/logs/R-%x_%A_%a.out
#SBATCH --error=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/submission-scripts/jean_zay/raw/radio_reconstruction_tools/validation/jobs/logs/R-%x_%A_%a.err
#SBATCH -A ney@h100                  # allocation H100 (radio reconstruction, for now)

# Validate the finished img360 tkbn (unrolled) large run: plots the training metrics, computes
# reconstruction PSNR on the held-out TEST split, and renders the qualitative
# panel. Everything (model type, backend, image size, imaging weights) is read
# from the run's resolved_config.yaml, so this job only needs the RUN_DIR.
# The script validates the DEEPEST stage that has a checkpoint, so it also works
# on a run still climbing the greedy-depth schedule (it reports the depth).
# Outputs land in <RUN_DIR>/validation/.

set -x

CODE_REPO=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/radio_reconstruction_tools
SUBMISSION_REPO=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/submission-scripts
OUTPUT_ROOT=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/outputs/trained_models

# Auto-resolve the newest run directory for this experiment. The sequential run
# mints "<RUN_ID>_<timestamp>"; we pick the newest such dir that actually holds
# a resolved_config.yaml (this excludes the single-timestamp chain-marker dir).
BASE=large_run_img360_unrolled_n15_prior_DRUNet_op_tkbn_4gpus
RUN_DIR=""
for d in $(ls -dt "$OUTPUT_ROOT/${BASE}"_*_* 2>/dev/null); do
    if [ -f "$d/resolved_config.yaml" ]; then RUN_DIR="$d"; break; fi
done
if [ -z "$RUN_DIR" ]; then
    echo "[validate] ERROR: no run dir with resolved_config.yaml matching ${BASE}_*_* under $OUTPUT_ROOT"
    echo "[validate] (has this training run started / produced its first stage yet?)"
    exit 1
fi
echo "[validate] resolved RUN_DIR=$RUN_DIR"

# --- Environment (H100: arch/h100 pre-module before pytorch) ---------------
. $SUBMISSION_REPO/jean_zay/env_configs/radio_reconstruction_tools_h100.sh
export TMPDIR=$JOBSCRATCH
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_DUPLICATE_LIB_OK=TRUE

cd $CODE_REPO

# --n-eval / --batch-size tuned for this image size + backend (img360 tkbn (unrolled)).
srun python -u scripts/validation/validate_unrolled.py \
    --run-dir "$RUN_DIR" \
    --n-eval 100 \
    --n-panel 5 \
    --isnr 20 \
    --batch-size 4 \
    --device cuda
