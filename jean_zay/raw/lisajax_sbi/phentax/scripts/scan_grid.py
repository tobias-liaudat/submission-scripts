"""Find which analytical-corner-plot grid points break phentax's solver.

Reproduces validate_posterior.py's test-sample selection and its 1-D grid
sweeps (each parameter scanned across its full prior range with the others held
at the true value), evaluating one source at a time so a failure can be
attributed to a single parameter value instead of taking a whole vmapped chunk
down with it.
"""

import json
import os
import sys

import jax

jax.config.update("jax_enable_x64", True)

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from lisajax_sbi.simulator import LISASimulator  # noqa: E402
from lisajax_sbi.utils import DataGenerationConfig  # noqa: E402
from lisajax_sbi.waveforms import build_waveform_model  # noqa: E402

DATASET = sys.argv[1]
N_GRID_1D = int(sys.argv[2]) if len(sys.argv) > 2 else 25

with h5py.File(DATASET, "r") as f:
    cfg = DataGenerationConfig.model_validate(json.loads(f.attrs["config"]))
    n_total = f["xs"].shape[0]
    # Same split as validate_posterior.py: seed 0, 0.8 / 0.1 / 0.1.
    idx = np.arange(n_total)
    np.random.default_rng(0).shuffle(idx)
    test_idx = idx[int(n_total * 0.8) + int(n_total * 0.1):]
    selected = sorted(test_idx[:100])
    thetas_test = f["thetas"][selected]

theta_true = thetas_test[0].astype(np.float64)
names = [p.name for p in cfg.params]
limits = [(float(p.bounds[0]), float(p.bounds[1])) for p in cfg.params]

print("waveform:", cfg.waveform, "| higher_modes:", cfg.higher_modes)
print("theta_true:", dict(zip(names, np.round(theta_true, 4))))

waveform_model = build_waveform_model(
    cfg.waveform,
    delta_t=cfg.dt,
    f_min=cfg.f_min,
    f_ref=cfg.f_ref,
    higher_modes=cfg.higher_modes,
)
sim = LISASimulator(
    orbit_path=cfg.orbit_path,
    n_ts=cfg.n_ts,
    dt=cfg.dt,
    tdi_channels=cfg.tdi_channel,
    params={p.name: p for p in cfg.params},
    waveform_model=waveform_model,
    add_noise=False,
    signal_chunk_size=1,
)

failures = []
for i, (name, (lo, hi)) in enumerate(zip(names, limits)):
    grid = np.linspace(lo, hi, N_GRID_1D)
    bad = []
    for v in grid:
        theta = theta_true.copy()
        theta[i] = v
        try:
            out = sim.forward(torch.tensor(theta[None, :], dtype=torch.float64))
            if not np.isfinite(np.asarray(out)).all():
                bad.append((float(v), "non-finite"))
        except Exception as exc:  # noqa: BLE001
            bad.append((float(v), type(exc).__name__))
    status = "OK" if not bad else f"{len(bad)}/{N_GRID_1D} FAIL"
    print(f"SCAN {name:22s} [{lo:.4g}, {hi:.4g}] -> {status}")
    for v, kind in bad:
        print(f"     FAIL at {name}={v:.6g} ({kind})")
    if bad:
        failures.append((name, bad))

print("SUMMARY failing parameters:", [n for n, _ in failures] or "none")
