#!/bin/bash
#SBATCH --job-name=phentax_compare_roots
#SBATCH --account=ney@v100
#SBATCH --constraint=v100-32g
#SBATCH --qos=qos_gpu-dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --hint=nomultithread
#SBATCH --time=01:00:00
#SBATCH --output=/lustre/fswork/projects/rech/ney/ulx23va/projects/LISA/outputs/testing/jobs/logs/%x_%j.out
#SBATCH --error=/lustre/fswork/projects/rech/ney/ulx23va/projects/LISA/outputs/testing/jobs/logs/%x_%j.err

set -euo pipefail
module purge
module load pytorch-gpu/py3/2.8.0
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Scripts are tracked here; all outputs go to /lustre/fswork/projects/rech/ney/ulx23va/projects/LISA/outputs/testing/hm_diag.
# NOTE: SLURM copies this script to the node's spool dir, so ${BASH_SOURCE[0]}
# does NOT resolve to the submission path -- the location is absolute here.
SCRIPTS=/lustre/fswork/projects/rech/ney/ulx23va/projects/LISA/repos/submission-scripts/jean_zay/raw/lisajax_sbi/phentax/scripts
cd "$SCRIPTS"
# Capture the root values + TDI signals under a tag, e.g.
#   sbatch --export=ALL,TAG=base jobs/compare_roots.slurm   # before a change
#   sbatch --export=ALL,TAG=fix  jobs/compare_roots.slurm   # after it
# then diff:  python compare_roots.py --diff base fix
python -u compare_roots.py "${TAG:?set TAG=<tag>, e.g. --export=ALL,TAG=base}"
