"""At the failing (eta, chi1z) point, find the eta threshold where phentax recovers."""

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
with h5py.File(f"{T}/data/test_jean_zay_dataset_phentax_hm.h5", "r") as f:
    cfg = DataGenerationConfig.model_validate(json.loads(f.attrs["config"]))
    n_total = f["xs"].shape[0]
    idx = np.arange(n_total)
    np.random.default_rng(0).shuffle(idx)
    thetas_test = f["thetas"][sorted(idx[int(n_total * 0.9):][:100])]

names = [p.name for p in cfg.params]
base = thetas_test[0].astype(np.float64)
base[names.index("chi1z")] = 0.5775
i_eta = names.index("reduced_mass")

wf = build_waveform_model(
    "phentax", delta_t=cfg.dt, f_min=cfg.f_min, f_ref=cfg.f_ref,
    higher_modes=cfg.higher_modes,
)
sim = LISASimulator(
    orbit_path=cfg.orbit_path, n_ts=cfg.n_ts, dt=cfg.dt,
    tdi_channels=cfg.tdi_channel, params={p.name: p for p in cfg.params},
    waveform_model=wf, add_noise=False, signal_chunk_size=1,
)

for eta in np.round(np.arange(0.100, 0.161, 0.005), 4):
    theta = base.copy()
    theta[i_eta] = eta
    try:
        out = np.asarray(sim.forward(torch.tensor(theta[None, :], dtype=torch.float64)))
        print(f"ETA {eta:.4f} -> {'OK' if np.isfinite(out).all() else 'NON-FINITE'}")
    except Exception as exc:  # noqa: BLE001
        print(f"ETA {eta:.4f} -> FAIL {type(exc).__name__}")
