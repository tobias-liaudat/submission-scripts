# equivariant_bootstrap — UQ transform campaign

Jean Zay jobs for the `UQsuite` equivariant-bootstrap transform programme: which group
action produces uncertainty maps whose shape tracks the true reconstruction error.

```text
configs/campaign_64.yaml    the 64^2 screen  — ladder + kappa sweep + M4 controls (34 arms)
configs/campaign_360.yaml   the 360^2 confirmation — ladder only (8 arms)
jobs/prepare_cluster.sh     login-node preflight. RUN FIRST.
jobs/submit_all.sh          submits the four jobs in order
jobs/level_b.sh             ~10 min. Read this before spending on the rest.
jobs/campaign_64.sh         the screen, self-chaining
jobs/campaign_360_ar.sh     confirmation, artifact-removal
jobs/campaign_360_unrolled.sh   confirmation, PSF-unrolled (the expensive one)
jobs/_common.sh             shared env, paths and chaining helpers (sourced)
```

Environment comes from `../../env_configs/equivariant_bootstrap{,_a100,_h100}.sh`.
Code lives in `$RADIO_ROOT/repos/UQsuite`; these scripts only supply configuration.

## Quick start

```bash
cd $SUBMISSION_REPO/jean_zay/raw/equivariant_bootstrap/jobs
bash prepare_cluster.sh     # checks paths, prefetches DRUNet, prints measured cost
bash submit_all.sh
```

Results: `$RADIO_ROOT/outputs/uq_campaign/<campaign>/<family>/<arm>_seed<k>.json`,
each with a cached calibration set beside it.

## Read this before interpreting anything

**`plan/cluster_handoff.md` in the project repo** is the full briefing: why the campaign
exists, the decision rules fixed in advance, the open questions and the traps. The two
things most likely to waste a day:

- **Compute nodes have no network**, and both unrolled runs load DRUNet with
  `path_denoiser: null`. `prepare_cluster.sh` prefetches it; without that they die minutes in.
- **PSNR is the wiring check and it is decisive.** `uv`, `op_norm`, `psf_fft` and the
  imaging-weight scheme thread through several layers, none is stored in any checkpoint, and
  any being wrong degrades the reconstruction *silently*. A `psnr_within_tolerance` failure
  means a wiring bug — stop rather than continuing the campaign.

## Unverified in the SBATCH headers

Written without cluster access; check before the first submission.

- `qos_gpu-t3`, `qos_gpu-t4` and `qos_gpu-dev` are taken from existing scripts in this repo.
  **`qos_gpu_a100-t3` and `qos_gpu_a100-t4` are guesses** by analogy with the
  `qos_gpu_h100-t3` the training jobs use.
- The jobs request `rbn@v100` and `rbn@a100`. Run `idrproj` to confirm those allocations
  exist. If there is no `rbn` a100 allocation, move the 360^2 jobs to `-C v100-32g`,
  `--qos=qos_gpu-t3`, `-A rbn@v100` and expect them to be slower.
- Wall-times were set from laptop CPU measurements. `prepare_cluster.sh` prints the measured
  per-draw cost and projected hours per arm-seed on the real hardware — use those.

## Self-chaining

Each `campaign_*` job queues its successor with `--dependency=afterany` *before* starting
work, so a wall-time kill hands over rather than losing progress. `--skip-existing` makes the
successor resume rather than repeat. A `.COMPLETE` marker stops the chain; `.resubmit_count`
caps it at 20 so a job failing instantly cannot loop.

Expect one pending job per campaign in `squeue` — that is the successor, not a double
submission.

```bash
touch $RADIO_ROOT/outputs/uq_campaign/campaign64/.COMPLETE   # stop after the current link
scancel <pending successor id>                                # stop now
```

`level_b.sh` does not chain; it is a single short job.
