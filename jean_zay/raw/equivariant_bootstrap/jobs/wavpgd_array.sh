#!/bin/bash
#SBATCH --job-name=eqb_wavpgd
#SBATCH --mail-user=tobiasliaudat@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH -C h100
#SBATCH --hint=nomultithread
#SBATCH --time=08:00:00
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --array=0-14
#SBATCH --output=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/submission-scripts/jean_zay/raw/equivariant_bootstrap/jobs/logs/R-%x_%A_%a.out
#SBATCH --error=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/submission-scripts/jean_zay/raw/equivariant_bootstrap/jobs/logs/R-%x_%A_%a.err
#SBATCH -A rbn@h100

# ===========================================================================
# THE RECONSTRUCTOR-INDEPENDENCE CAMPAIGN, ON H100s.
#
# The equivariant bootstrap run with a CLASSICAL reconstructor -- proximal
# gradient, db1-db8 wavelet dictionary, PSF normal operator (D48) -- so the
# method's result does not rest on a network we trained ourselves.
#
# ONE ROW PER ARRAY TASK. 5 arms x 3 seeds = 15 rows, `--array=0-14`,
# `TASK_COUNT=15`. Unlike the a100-dev arrays this needs no two-round dance:
# `qos_gpu_h100-t3` allows 10000 submitted jobs against dev's 10.
#
#   sbatch wavpgd_array.sh                                   # 64^2  (default)
#   CONFIG=wavpgd_360.yaml sbatch --time=20:00:00 wavpgd_array.sh   # 360^2
#
#   The out-dir follows the config's basename, so the two cannot collide and
#   `OUT_DIR` only needs setting to override that. BOTH MAY RUN AT ONCE: 30 tasks
#   against a 512-GPU / 10000-job t3 limit, distinct out-dirs, and logs keyed on
#   the array job id rather than the job name.
#
#   Bare names, NOT $CAMPAIGN_DIR/... -- those live in `_common.sh` and are
#   empty in the shell you type this into.
#
# WALL TIME. Measured on an A100 at max_iter=100 with the pinned reg_strength
# (M42), per row of 12 800 reconstructions: 64^2 **0.51 h**, 360^2 **3.43 h** at
# batch 64. These configs run batch 50 (it divides n_test evenly), which
# interpolates to ~0.64 h and ~3.8 h per row -- so ~10 h and ~57 h for the 15
# rows of each size. The 8 h default
# is ~13x headroom at 64^2; for 360^2 pass `--time=20:00:00` (the t3 ceiling),
# which is ~6x. H100 should be faster than A100, but the hot loop is many small
# wavelet transforms rather than dense matmul, so do NOT assume the usual
# speedup -- confirm with `cost_iterative.py --device cuda` on an H100 before
# trusting a tighter wall clock.
#
# BEFORE THE ARRAY, ONE ROW. The classical path is newer than everything else
# here, so smoke it:
#
#   CONFIG=wavpgd_64.yaml ROW=wavpgd/ALLc_1.20/1 OUT_DIR=smoke_wavpgd \
#   GPU_ARCH=h100 sbatch -C h100 -A rbn@h100 --qos=qos_gpu_h100-dev \
#       --cpus-per-task=24 smoke_64_row.sh
#
# WHAT TO READ:
#
#   psnr   must sit near the run's reference -- 29.05 dB at 64^2, 32.68 dB at
#          360^2, both measured at the pinned reg_strength=0.088 -- and BELOW the trained net (31.05 / 37.54). A classical solver
#          beating the net means `uv`, `op_norm` or `psf_fft` is mis-wired, which
#          is silent in every other respect. `psnr_tolerance_db: 3.0` turns that
#          into a row-level FAIL.
#
#   J and slope   the three-point story is P0_parametric -> G_geometric ->
#          best ALLc. At 360^2 the arms separated barely on J but cleanly on the
#          conditional-coverage slope; read both, and do not treat a null on J
#          as a null overall.
#
#   which ALLc wins   the reason three kappa targets run rather than one. The
#          optimum tracks reconstructor headroom, and this solver has more of it
#          than either net, so expect the winner ABOVE the net's (ALLc_1.20 at
#          64^2, ALLc_1.10 at 360^2). A winner at the top of the grid means the
#          grid is too low, not that kappa does not matter.
#
# RESUME. `--skip-existing` makes a resubmission a no-op for finished rows, so
# re-running the array is the resume mechanism. To re-shard only what is left:
#   ROWS="$(OUT_DIR=wavpgd_360 CONFIG=wavpgd_360.yaml bash missing_rows.sh)"
# ===========================================================================

export GPU_ARCH=h100
SUBMISSION_REPO=/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/submission-scripts
. $SUBMISSION_REPO/jean_zay/raw/equivariant_bootstrap/jobs/_common.sh

CONFIG=${CONFIG:-wavpgd_64.yaml}

# Bare names resolved against the campaign layout; see smoke_64_row.sh for why
# this is not sugar.
case "$CONFIG" in */*) ;; *) CONFIG=$CAMPAIGN_DIR/configs/$CONFIG ;; esac

# THE OUT-DIR DEFAULTS FROM THE CONFIG, and that is load-bearing rather than
# tidy. A row's path is `<out_dir>/<family>/<arm>_seed<seed>.json`, and `family`
# is `wavpgd` at BOTH image sizes -- so the 64^2 and 360^2 rows have identical
# filenames and are told apart only by the out-dir. With a fixed default, running
# the 360^2 array and forgetting `OUT_DIR=` would point it at the 64^2 results,
# where `--skip-existing` would find every filename already present and skip the
# whole campaign: a job that reports success and does nothing. Deriving the
# default from the config's basename removes that failure mode instead of
# documenting it.
OUT_DIR=${OUT_DIR:-$(basename "$CONFIG" .yaml)}
case "$OUT_DIR" in */*) ;; *) OUT_DIR=$OUTPUT_ROOT/$OUT_DIR ;; esac

if [ ! -f "$CONFIG" ]; then
    echo "[wavpgd] no config at '$CONFIG'." >&2
    ls -1 "$CAMPAIGN_DIR/configs" | sed 's/^/[wavpgd]   /' >&2
    exit 2
fi

# MUST match --array above; `uq_campaign.py` takes rows[i::n]. Change the two
# together or rows are silently dropped.
TASK_COUNT=${TASK_COUNT:-15}
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
ROWS=${ROWS:-}

echo "[wavpgd] shard $TASK_ID of $TASK_COUNT  config=$(basename "$CONFIG")  ->  $OUT_DIR"
[ -n "$ROWS" ] && echo "[wavpgd] restricted to $(echo "$ROWS" | wc -w) row(s)"

run_step python -u scripts/uq_campaign.py \
    --config "$CONFIG" \
    --out-dir "$OUT_DIR" \
    --device cuda \
    --skip-existing \
    --task-id "$TASK_ID" \
    --task-count "$TASK_COUNT" \
    ${ROWS:+--rows $ROWS}
