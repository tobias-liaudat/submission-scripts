#!/bin/bash
#SBATCH --job-name=eqb_maxiter
#SBATCH --mail-user=tobiasliaudat@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -C a100
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --output=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/submission-scripts/jean_zay/raw/equivariant_bootstrap/jobs/logs/R-%x_%j.out
#SBATCH --error=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/submission-scripts/jean_zay/raw/equivariant_bootstrap/jobs/logs/R-%x_%j.err
#SBATCH -A rbn@a100

# ===========================================================================
# IS wavPGD AT 360^2 UNDER-CONVERGED, OR IS THE PROBLEM JUST EASY?
#
# THE QUESTION. The parametric bootstrap behaves as the equivariant-bootstrap
# paper predicts in three of our four (reconstructor, size) cases -- it
# under-covers badly, because at the operator's information ceiling the residual
# error IS null-space error and resampling noise cannot see it. The exception is
# wavpgd/360, where it OVER-covers (raw coverage 0.917).
#
# That case is also the only one running well below the ceiling:
#
#     unrolled/360   ceiling 36.78   achieved 37.54   -> at ceiling
#     unrolled/64    ceiling 30.73   achieved 31.05   -> at ceiling
#     wavpgd/64      ceiling 30.73   achieved 30.15   -> at ceiling
#     wavpgd/360     ceiling 36.78   achieved 33.69   -> 3.1 dB BELOW
#
# ("ceiling" = PSNR of the truth projected onto the Fourier cells a visibility
# lands in: the missing-data limit for a *linear* reconstruction. Not a hard
# bound -- the nets exceed it by inferring null-space content -- but the gap says
# whether missing data or the solver is what limits.)
#
# So the anomaly may be the SOLVER, not the simulation: 100 PGD iterations with a
# fixed wavelet prior may simply not have converged to what the data supports at
# 360^2. If PSNR climbs toward 36.8 dB with more iterations, nothing about the uv
# simulation needs changing and the null-space story is intact.
#
# 64^2 runs too, as the control: it is AT its ceiling already, so its PSNR should
# be flat in max_iter. If both climb, the reading is under-convergence
# everywhere; if only 360 climbs, the diagnosis above is confirmed.
#
# WHAT TO READ:
#
#   the "operator ceiling" line, printed per size before the sweep
#   whether 360^2 PSNR rises toward it, and where it flattens
#   the cost column -- iterations are linear, so 1600 is 16x the campaign's 100
#
# IF IT CLIMBS, RE-TUNE BEFORE CONCLUDING. `reg_strength` was tuned at
# max_iter=100 (M42), i.e. against an under-converged solver. lambda defines the
# minimiser and more iterations only get closer to it, so the tuned value should
# still be roughly right -- but "roughly" is worth checking with a reg_strength
# sweep at the new max_iter before any campaign is re-run on it.
#
#   sbatch wavpgd_maxiter.sh
#   GRID="100 400 1600" sbatch wavpgd_maxiter.sh
# ===========================================================================

export GPU_ARCH=a100
SUBMISSION_REPO=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/submission-scripts
. $SUBMISSION_REPO/jean_zay/raw/equivariant_bootstrap/jobs/_common.sh

GRID=${GRID:-"100 200 400 800 1600"}
SIZES=${SIZES:-"64 360"}
OUT_DIR=${OUT_DIR:-wavpgd_maxiter}
case "$OUT_DIR" in */*) ;; *) OUT_DIR=$OUTPUT_ROOT/$OUT_DIR ;; esac

echo "[maxiter] sizes=$SIZES grid=$GRID -> $OUT_DIR"

# --super-resolution must match the campaign configs (both use 1.5); it sets the
# band limit the ceiling is computed against, and a wrong value would move the
# reference line this whole job is read against.
run_step python -u scripts/tune_iterative.py \
    --family wavpgd \
    --img-size $SIZES \
    --sweep max_iter \
    --grid $GRID \
    --n-images 8 \
    --super-resolution 1.5 \
    --device cuda \
    -o "$OUT_DIR"
