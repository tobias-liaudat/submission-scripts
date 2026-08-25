#!/bin/bash
# End-to-end proof that the get_time_of_frequency fix works: the eta sweep, the
# mode isolation on the previously-fatal vector, and the full parameter grid
# scan for both mode configurations. Pre-fix, chi1z=0.5775 fails; after, none do.
#SBATCH --job-name=phentax_verify_fix
#SBATCH --account=ney@v100
#SBATCH --constraint=v100-32g
#SBATCH --qos=qos_gpu-dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --hint=nomultithread
#SBATCH --time=01:30:00
#SBATCH --output=/lustre/fswork/projects/rech/ney/ulx23va/projects/LISA/outputs/testing/jobs/logs/%x_%j.out
#SBATCH --error=/lustre/fswork/projects/rech/ney/ulx23va/projects/LISA/outputs/testing/jobs/logs/%x_%j.err

set -euo pipefail
module purge
module load pytorch-gpu/py3/2.8.0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# NOTE: SLURM copies this script to the node's spool dir, so ${BASH_SOURCE[0]}
# does NOT resolve to the submission path -- the location is absolute here.
SCRIPTS=/lustre/fswork/projects/rech/ney/ulx23va/projects/LISA/repos/submission-scripts/jean_zay/raw/lisajax_sbi/phentax/scripts
cd "$SCRIPTS"

echo "===== ETA SWEEP ====="
python -u eta_probe.py
echo "===== MODE ISOLATION ====="
python -u mode_isolate.py
echo "===== GRID SCAN (higher modes) ====="
python -u scan_grid.py /lustre/fswork/projects/rech/ney/ulx23va/projects/LISA/outputs/testing/data/test_jean_zay_dataset_phentax_hm.h5 25
echo "===== GRID SCAN (2,2) ====="
python -u scan_grid.py /lustre/fswork/projects/rech/ney/ulx23va/projects/LISA/outputs/testing/data/test_jean_zay_dataset_phentax.h5 25
