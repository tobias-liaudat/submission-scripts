# equivariant_bootstrap — UQ transform campaign

Jean Zay jobs for the `UQsuite` equivariant-bootstrap transform programme: which group
action produces uncertainty maps whose shape tracks the true reconstruction error.

```text
configs/coverage_64_gpu.yaml  the 64^2 sweep (19 arms x 3 seeds) -- NEEDS REWRITING, see below
configs/campaign_64.yaml      the 64^2 screen — ladder + kappa sweep + M4 controls
configs/campaign_360.yaml     the 360^2 confirmation — ladder only (8 arms)
jobs/prepare_cluster.sh       login-node preflight. RUN FIRST.
jobs/mc_ladder_64.sh          array of 3. How many draws does W(c) need? RUN FIRST.
jobs/coverage_64_gpu.sh       raw-coverage sweep, self-chaining
jobs/submit_all.sh            submits the campaign jobs in order
jobs/level_b.sh               ~10 min. Read this before spending on the rest.
jobs/campaign_64.sh           the screen, self-chaining
jobs/campaign_360_ar.sh       confirmation, artifact-removal
jobs/campaign_360_unrolled.sh confirmation, PSF-unrolled (the expensive one)
jobs/_common.sh               shared env, paths and chaining helpers (sourced)
```

## The current line of work

**The selection criterion is fixed** in `plan/transform_selection_metrics.md`: minimise the
**calibrated interval width** `W(c)` — the interval width needed to reach coverage `c` once a
scalar `lambda` has been fitted on a disjoint split — subject to a prior-departure constraint
and numerical validity. Smallest `J = sum_c w_c W(c)` wins.

What that demotes, and why. A scalar can always be fitted to reach nominal coverage, so **raw
coverage and raw width no longer decide anything** — they are properties of the calibration,
not of the transform. `rho_masked`, previously the primary metric, is demoted for the same
reason and is *subsumed*: a well-shaped map is precisely one that needs less inflation to
reach `c`, which is what `W(c)` measures directly. Realized `kappa` is a constraint and a
knob, never an objective. All are still reported; none decide.

```bash
sbatch mc_ladder_64.sh        # FIRST: how many draws does W(c) need?
```

### Before submitting anything else

- **`coverage_64_gpu.yaml` is not yet updated for this.** It still carries the `gap` arms,
  which are out of scope pending the prior gate; its `levels` omit `0.95`, so `W(0.95)` cannot
  be computed; and its `MC` should come from the ladder rather than being assumed. **Run the
  ladder, then rewrite the config.**
- The prior-departure gate `D_prior` is **on hold** — defined, not built. Arms are judged
  against it by inspection, which is what parks the `gap` family. The spectral tilts are kept
  on judgement, and that judgement is the first thing to point the gate at when it exists.

### Revised 2026-08-18 — the gap selector was broken, and two families were missing

The gap selector pooled the batch and ranked cells on **absolute** uv density, so every
candidate landed in a thin annulus at 86–93 % of the band limit inside a 38° wedge
(changelog D34). At this config's `batch_size` that is where the `gap` arms would have
gone — meaning `gap`, `gap_random` and `gap_rotated` would all have perturbed the same
sparse region, and the M4 gate would have come back null again at 8× the statistics for
reasons that have nothing to do with the physics.

Selection is now per image and ranked on density *relative to the same radius ring*.
`C_gap_pooled_1.50` runs the superseded selector so the fix can be priced on the same
images and seeds.

Also added: `elliptical_tilt` with its two orientation controls, and `briggs_tilt`. Until
Briggs existed, plan §5.2's M3 gate — *prefer the radial tilt unless Briggs beats it at
matched realized κ* — could not be evaluated; the radial tilt was unopposed.

**Sizing changed.** 19 arms × 3 seeds = **57 rows**, 256 MC × 100 images = **1,459,200
image-draws** (was 42 rows / 1,075,200). At 0.05 s/draw that is 20.3 h — *just over* the
20 h walltime, so this now certainly needs its successor job. `prepare_cluster.sh` prints
the measured per-draw cost on the real hardware; use that, not this estimate.

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
