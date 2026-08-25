#!/bin/bash
# Regression gate: the phentax suite plus the phentax-related lisajax_sbi tests.
# Baseline before the fix was 1070 passed / 17 passed; both must be unchanged.
#SBATCH --job-name=phentax_test_suites
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

echo "===== PHENTAX SUITE ====="
cd /lustre/fswork/projects/rech/ney/ulx23va/projects/LISA/repos/phentax && python -m pytest tests/ -q --no-header -p no:cacheprovider || true
echo "===== LISAJAX_SBI PHENTAX TESTS ====="
cd /lustre/fswork/projects/rech/ney/ulx23va/projects/LISA/repos/lisajax_sbi && python -m pytest \
    tests/test_phentax_waveform.py tests/test_simulator_batching.py \
    tests/test_vmap_likelihood.py -q --no-header -p no:cacheprovider || true
