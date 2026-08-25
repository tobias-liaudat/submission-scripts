# phentax `get_time_of_frequency` — diagnosis and validation

Scripts and jobs from the investigation and fix of a phentax crash that hit the
LISA SBI pipeline on Jean Zay:

```
_EquinoxRuntimeError: The maximum number of steps was reached in the nonlinear solver.
```

**Cause.** `time_of_freq` evaluates `omega` on a time axis shifted by `tt0`
before `tEarly` and unshifted after, so the residual it bisects is
*discontinuous* at `tEarly`. When the requested `f_min` is not attained anywhere
in the bracket — a thin sliver of `(eta, chi1z)` — that jump still straddles
zero, so `optx.Bisection` accepts the bracket, converges the interval, and then
exhausts all `max_steps` because `|residual| < atol` can never hold near a jump
whose smallest attainable magnitude is ~2e-6.

**Fix** (in the phentax fork, `src/phentax/core/phase.py`): pick the continuous
branch that actually brackets a root — `[t_low, tEarly]` shifted or
`[tEarly, t_high]` unshifted — then run a single `optx.root_find` with
`throw=False` and an explicit `flip=True`. When neither branch holds a root the
frequency is genuinely unattainable and `tEarly` is returned, that seam being
the best estimate of the crossing.

Ruled out by measurement, so they do not need re-testing: `max_steps` (~41 steps
suffice), the bracket width, `expand_if_necessary`, NaN in the residual, the
termination tolerance (`1e-12` → `1e-9` byte-identical), and the higher modes
(fails identically with `None`).

## Layout

| path | tracked | contents |
|---|---|---|
| `scripts/` | yes | the analysis scripts |
| `jobs/` | yes | one SLURM job per script |
| `data/failing_theta.json` | yes | the parameter vector that triggered the crash |
| `$OUT/hm_diag/` | no | every `.npz`, `.png` and the regenerated pre-fix phentax copy |
| `$OUT/jobs/logs/` | no | SLURM stdout/stderr |

where `$OUT` = `/lustre/fswork/projects/rech/ney/ulx23va/projects/LISA/outputs/testing`.
Scripts read tracked inputs from `data/` and write **all** output under `$OUT`,
so the repo stays free of generated data.

## Running

```bash
sbatch jobs/verify_fix.slurm          # the headline proof; ~25 min
sbatch jobs/compare_waveforms.slurm   # before/after waveform plots
sbatch jobs/run_test_suites.slurm     # regression gate (expect 1070 + 17 passed)
```

Two jobs are parameterised by a tag, to capture the same measurement either side
of a change:

```bash
sbatch --export=ALL,TAG=base jobs/compare_roots.slurm   # before
sbatch --export=ALL,TAG=fix  jobs/compare_roots.slurm   # after
python scripts/compare_roots.py --diff base fix
```

`compare_waveforms.slurm` needs both phentax versions at once. It materialises a
pristine pre-fix copy from git (`PHENTAX_REF`, default `HEAD`) into `$OUT` and
imports it via `sys.path`, so your working phentax clone is never touched.

## What each script does

**Diagnosis — how the cause was isolated**

| script | question it answers |
|---|---|
| `scan_grid.py` | which parameter, swept across its full prior range, fails? (→ `chi1z`) |
| `eta_probe.py` | how far does the failure extend in eta at that `chi1z`? |
| `mode_isolate.py` | is it the higher modes? (no — fails with every mode set) |
| `preflight.py` | flip orientation, segment continuity, and the `t_low_fit` knob |
| `discont.py` | is the crossing a real root or the `tEarly` jump? (**the answer**) |
| `probe_tof.py` | the residual across the bracket: finite? monotonic? bracketed? |
| `tol_test.py` | does loosening `atol`/`rtol` help? (no, `1e-12` → `1e-9` identical) |
| `tol_propagate.py` | control: confirms the tolerance override really reaches the solver |
| `nan_mode.py` | does `EQX_ON_ERROR=nan` isolate the bad source? (no — zeroes all 8) |
| `corruption.py` | is that zeroing silent corruption? (yes — an all-zero waveform) |

**Validation — evidence the fix is safe**

| script | measures |
|---|---|
| `compare_roots.py` | `Mt_min`/`Mt_ref` and TDI signals before vs after |
| `compare_waveforms.py` | the waveforms themselves, plus `\|Δh\|`, as plots |
| `regress.py` | 24 working reference sources must stay unchanged |
| `noise_floor.py` | the codebase's own reproducibility floor (chunk size 1 vs 8) |

`noise_floor.py` is what makes the accuracy verdict meaningful: `CLAUDE.md`
notes signals are not bit-reproducible across chunk sizes (2nd-generation TDI
amplifies reassociation by ~1e5), so a change is acceptable if it stays inside
that measured floor rather than inside an invented threshold. The fix came out
~1000x inside it.

## Jean Zay notes

- `ney@v100` with `-C v100-32g`, `module load pytorch-gpu/py3/2.8.0`.
- **20 cores, not 10.** Memory scales with allocated cores; 10 cores (40 GB) is
  OOM-killed on anything touching the likelihood, because JAX captures the
  (18000, 18000) float64 inverse covariance (2.59 GB) as a jit constant once for
  the IS likelihood and again for `ISPosteriorForTARP`.
- There is no `ney@cpu` allocation — `sbatch` rejects it as an invalid
  account/partition combination.
