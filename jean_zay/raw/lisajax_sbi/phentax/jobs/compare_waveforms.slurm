#!/bin/bash
# Before/after waveform comparison plots. Needs BOTH phentax versions, so it
# materialises a pristine pre-fix copy from git into the OUTPUT tree (it is
# generated data -- deliberately not tracked here) and imports it via sys.path.
# Set PHENTAX_REF to compare against something other than the committed HEAD.
#SBATCH --job-name=phentax_compare_waveforms
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
# NOTE: SLURM copies this script to the node's spool dir, so ${BASH_SOURCE[0]}
# does NOT resolve to the submission path -- the location is absolute here.
SCRIPTS=/lustre/fswork/projects/rech/ney/ulx23va/projects/LISA/repos/submission-scripts/jean_zay/raw/lisajax_sbi/phentax/scripts

PHENTAX_REPO=/lustre/fswork/projects/rech/ney/ulx23va/projects/LISA/repos/phentax
OLD_SRC=/lustre/fswork/projects/rech/ney/ulx23va/projects/LISA/outputs/testing/hm_diag/phentax_old_src
PHENTAX_REF="${PHENTAX_REF:-HEAD}"

echo "Materialising pre-fix phentax ($PHENTAX_REF) into $OLD_SRC"
rm -rf "$OLD_SRC"; mkdir -p "$OLD_SRC"
cp -r "$PHENTAX_REPO/src/phentax" "$OLD_SRC/"
find "$OLD_SRC" -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
git -C "$PHENTAX_REPO" show "$PHENTAX_REF:src/phentax/core/phase.py" \
    > "$OLD_SRC/phentax/core/phase.py"

cd "$SCRIPTS"
echo "===== PREVIOUS (pre-fix) phentax ====="
python -u compare_waveforms.py --variant old || true
echo "===== CURRENT (fixed) phentax ====="
python -u compare_waveforms.py --variant new
echo "===== PLOT ====="
python -u compare_waveforms.py --plot
