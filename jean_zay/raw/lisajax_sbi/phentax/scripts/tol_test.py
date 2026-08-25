"""Test whether the failure is optimistix's absolute-residual termination.

Bisection terminates only when BOTH
    |lower-upper| < atol + rtol*|y|      and     |g| < atol
hold.  The second is a pure ABSOLUTE residual test.  Near the root g is very
flat (dg/dt ~ 1.5e-8), so if imr_omega's numerical noise floor sits above
atol=1e-12, |g| can never get under it and bisection burns all max_steps.

If that is the story, loosening atol fixes it outright.
"""

import json

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import h5py  # noqa: E402

from lisajax_sbi.simulator import LISASimulator  # noqa: E402
from lisajax_sbi.utils import DataGenerationConfig  # noqa: E402
from lisajax_sbi.waveforms import build_waveform_model  # noqa: E402

T = "/lustre/fswork/projects/rech/ney/ulx23va/projects/LISA/outputs/testing"
with h5py.File(f"{T}/data/test_jean_zay_dataset_phentax_hm.h5", "r") as f:
    cfg = DataGenerationConfig.model_validate(json.loads(f.attrs["config"]))
    n_total = f["xs"].shape[0]
    idx = np.arange(n_total)
    np.random.default_rng(0).shuffle(idx)
    thetas_test = f["thetas"][sorted(idx[int(n_total * 0.9):][:100])]

names = [p.name for p in cfg.params]
i_eta, i_chi = names.index("reduced_mass"), names.index("chi1z")
base = thetas_test[0].astype(np.float64).copy()
base[i_chi] = 0.5775

for tol in (1e-12, 1e-11, 1e-10, 1e-9):
    wf = build_waveform_model(
        "phentax", delta_t=cfg.dt, f_min=cfg.f_min, f_ref=cfg.f_ref,
        higher_modes=cfg.higher_modes,
    )
    try:
        object.__setattr__(wf._imr, "atol", tol)
        object.__setattr__(wf._imr, "rtol", tol)
    except Exception as exc:  # noqa: BLE001
        print(f"could not set tol: {type(exc).__name__}: {exc}")
        break
    sim = LISASimulator(
        orbit_path=cfg.orbit_path, n_ts=cfg.n_ts, dt=cfg.dt,
        tdi_channels=cfg.tdi_channel, params={p.name: p for p in cfg.params},
        waveform_model=wf, add_noise=False, signal_chunk_size=1,
    )
    # the previously failing vector, plus etas that spanned the threshold
    results = []
    for eta in (0.10077, 0.120, 0.140, 0.145, 0.200):
        th = base.copy()
        th[i_eta] = eta
        try:
            out = np.asarray(sim.forward(torch.tensor(th[None, :], dtype=torch.float64)))
            amax = np.abs(out).max()
            results.append(f"eta={eta}:OK(|x|max={amax:.2e})" if amax > 0 else f"eta={eta}:ZERO")
        except Exception:  # noqa: BLE001
            results.append(f"eta={eta}:FAIL")
    print(f"TOL atol=rtol={tol:.0e} -> " + "  ".join(results))
