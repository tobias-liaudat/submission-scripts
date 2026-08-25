"""Isolate which higher mode breaks the solver at a fixed parameter vector.

Holds theta exactly fixed (the analytical-grid point that failed) and varies
only the mode content, so any difference is attributable to the modes alone.
"""

import json

import jax

jax.config.update("jax_enable_x64", True)

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from lisajax_sbi.simulator import LISASimulator  # noqa: E402
from lisajax_sbi.utils import DataGenerationConfig  # noqa: E402
from lisajax_sbi.waveforms import build_waveform_model  # noqa: E402

T = "/lustre/fswork/projects/rech/ney/ulx23va/projects/LISA/outputs/testing"
DATASET = f"{T}/data/test_jean_zay_dataset_phentax_hm.h5"
BAD_CHI1Z = 0.5775

with h5py.File(DATASET, "r") as f:
    cfg = DataGenerationConfig.model_validate(json.loads(f.attrs["config"]))
    n_total = f["xs"].shape[0]
    idx = np.arange(n_total)
    np.random.default_rng(0).shuffle(idx)
    test_idx = idx[int(n_total * 0.8) + int(n_total * 0.1):]
    thetas_test = f["thetas"][sorted(test_idx[:100])]

names = [p.name for p in cfg.params]
theta = thetas_test[0].astype(np.float64)
theta[names.index("chi1z")] = BAD_CHI1Z
print("theta:", dict(zip(names, np.round(theta, 5))))

for modes in (None, [21], [33], [44], [55], [21, 33], [21, 33, 44], [21, 33, 44, 55]):
    wf = build_waveform_model(
        "phentax", delta_t=cfg.dt, f_min=cfg.f_min, f_ref=cfg.f_ref, higher_modes=modes
    )
    sim = LISASimulator(
        orbit_path=cfg.orbit_path,
        n_ts=cfg.n_ts,
        dt=cfg.dt,
        tdi_channels=cfg.tdi_channel,
        params={p.name: p for p in cfg.params},
        waveform_model=wf,
        add_noise=False,
        signal_chunk_size=1,
    )
    try:
        out = np.asarray(sim.forward(torch.tensor(theta[None, :], dtype=torch.float64)))
        ok = np.isfinite(out).all()
        print(f"MODES {str(modes):22s} -> {'OK' if ok else 'NON-FINITE'} (|x|max={np.abs(out).max():.3e})")
    except Exception as exc:  # noqa: BLE001
        print(f"MODES {str(modes):22s} -> FAIL {type(exc).__name__}")
