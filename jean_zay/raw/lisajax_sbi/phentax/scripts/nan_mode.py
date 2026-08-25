"""Does EQX_ON_ERROR=nan isolate the failing source, or poison the whole chunk?

Builds one chunk of 8 sources with exactly one known-bad vector at index 3 and
reports, per source, whether the returned signal is finite.
"""

import json
import os

import jax

jax.config.update("jax_enable_x64", True)

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from lisajax_sbi.simulator import LISASimulator  # noqa: E402
from lisajax_sbi.utils import DataGenerationConfig  # noqa: E402
from lisajax_sbi.waveforms import build_waveform_model  # noqa: E402

print("EQX_ON_ERROR =", os.environ.get("EQX_ON_ERROR", "<unset>"))

T = "/lustre/fswork/projects/rech/ney/ulx23va/projects/LISA/outputs/testing"
BAD_IDX, BAD_CHI1Z = 3, 0.5775

with h5py.File(f"{T}/data/test_jean_zay_dataset_phentax_hm.h5", "r") as f:
    cfg = DataGenerationConfig.model_validate(json.loads(f.attrs["config"]))
    n_total = f["xs"].shape[0]
    idx = np.arange(n_total)
    np.random.default_rng(0).shuffle(idx)
    thetas_test = f["thetas"][sorted(idx[int(n_total * 0.9):][:100])]

names = [p.name for p in cfg.params]
batch = thetas_test[:8].astype(np.float64).copy()
batch[BAD_IDX] = thetas_test[0].astype(np.float64)
batch[BAD_IDX, names.index("chi1z")] = BAD_CHI1Z

wf = build_waveform_model(
    "phentax", delta_t=cfg.dt, f_min=cfg.f_min, f_ref=cfg.f_ref,
    higher_modes=cfg.higher_modes,
)
sim = LISASimulator(
    orbit_path=cfg.orbit_path, n_ts=cfg.n_ts, dt=cfg.dt,
    tdi_channels=cfg.tdi_channel, params={p.name: p for p in cfg.params},
    waveform_model=wf, add_noise=False, signal_chunk_size=8,
)

try:
    out = np.asarray(sim.forward(torch.tensor(batch, dtype=torch.float64)))
    finite = np.isfinite(out).all(axis=(1, 2))
    amax = np.abs(out).max(axis=(1, 2))
    zeroed = amax == 0
    print("CHUNK8 returned without raising.")
    for i in range(len(out)):
        mark = "  <-- known-bad" if i == BAD_IDX else ""
        print(f"  source {i}: finite={bool(finite[i])} |x|max={amax[i]:.4e} "
              f"allzero={bool(zeroed[i])}{mark}")
    good = np.array([i for i in range(len(out)) if i != BAD_IDX])
    if zeroed[BAD_IDX] and not zeroed[good].any():
        print("VERDICT: per-source isolation -- bad source zeroed, good sources intact")
    elif zeroed.all():
        print("VERDICT: whole chunk poisoned -- every source zeroed")
    else:
        print("VERDICT: unexpected pattern")
except Exception as exc:  # noqa: BLE001
    print(f"CHUNK8 RAISED {type(exc).__name__} -- env var did not suppress it")
