#!/bin/bash
#SBATCH --job-name=rrt_large_img360_build_artifacts
#SBATCH --mail-user=tobiasliaudat@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8            # un huitieme d'un noeud A100 (64/8)
#SBATCH -C a100                      # partition A100 (gpu_p5)
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00              # 1000 op-norms x2 backends at 360^2 + PSF bank; large margin
#SBATCH --qos=qos_gpu_a100-t3
#SBATCH --output=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/submission-scripts/jean_zay/raw/radio_reconstruction_tools/large/jobs/logs/R-%x_%A_%a.out
#SBATCH --error=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/submission-scripts/jean_zay/raw/radio_reconstruction_tools/large/jobs/logs/R-%x_%A_%a.err
#SBATCH -A ney@a100                  # allocation A100 (radio reconstruction, for now)

# Single-process builder for the two large-run pinned conditions. MUST run
# before the first 4-GPU launch of either chain: under DDP, non-zero ranks wait
# at most ~30 min for a missing condition, and building 1000 op-norms takes
# longer. Builds both per-backend artifacts dirs, then verifies their UV banks
# and image splits are bit-identical (same seed + bank params -> deterministic),
# so the two chains train on exactly the same condition.

set -euo pipefail

CODE_REPO=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/radio_reconstruction_tools
SUBMISSION_REPO=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/submission-scripts
CONFIG_DIR=$SUBMISSION_REPO/jean_zay/raw/radio_reconstruction_tools/large/configs
CONFIG_TKBN=$CONFIG_DIR/large_run_img360_unrolled_n15_prior_DRUNet_op_tkbn_4gpus.yaml
CONFIG_PSF=$CONFIG_DIR/large_run_img360_unrolled_n15_prior_DRUNet_PSF_finufft_4gpus.yaml
ART_TKBN=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/outputs/artifacts/large_img360_cov1000_ntrain4000_tkbn
ART_PSF=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/outputs/artifacts/large_img360_cov1000_ntrain4000_psf_finufft

# --- Environment (A100: arch/a100 pre-module before pytorch) ---------------
. $SUBMISSION_REPO/jean_zay/env_configs/radio_reconstruction_tools_a100.sh
export TMPDIR=$JOBSCRATCH
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_DUPLICATE_LIB_OK=TRUE
export WANDB_MODE=offline

set -x
cd $CODE_REPO

srun python -u scripts/build_artifacts.py --config "$CONFIG_PSF" --device cuda
srun python -u scripts/build_artifacts.py --config "$CONFIG_TKBN" --device cuda

# Cross-check: the two dirs must pin the identical bank + splits.
srun python -u - "$ART_PSF" "$ART_TKBN" <<'EOF'
import json, sys
from pathlib import Path
import torch

a, b = Path(sys.argv[1]), Path(sys.argv[2])
ua = torch.load(a / "uv_bank.pt", map_location="cpu")
ub = torch.load(b / "uv_bank.pt", map_location="cpu")
assert torch.equal(ua, ub), "UV banks differ between the two artifacts dirs!"
sa = json.loads((a / "splits.json").read_text())
sb = json.loads((b / "splits.json").read_text())
assert sa == sb, "image splits differ between the two artifacts dirs!"
print(f"[verify] OK: identical uv_bank {tuple(ua.shape)} and splits across both dirs")
EOF

echo "[build_artifacts] both conditions built and verified."
