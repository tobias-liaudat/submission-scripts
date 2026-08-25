"""Calibrate the accepted numerical noise floor.

CLAUDE.md states signals are not bit-reproducible across chunk sizes (2nd-gen
TDI amplifies reassociation by ~1e5).  Measure that accepted difference on the
same 24 reference sources, so the patch's difference can be judged against the
codebase's own floor rather than an arbitrary threshold.
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
with h5py.File(f"{T}/data/test_jean_zay_dataset_phentax_hm.h5", "r") as f:
    cfg = DataGenerationConfig.model_validate(json.loads(f.attrs["config"]))
    n_total = f["xs"].shape[0]
    idx = np.arange(n_total)
    np.random.default_rng(0).shuffle(idx)
    test_idx = idx[int(n_total * 0.8) + int(n_total * 0.1):]
    thetas = f["thetas"][sorted(test_idx[:100])][:24].astype(np.float64)


def run(chunk):
    wf = build_waveform_model("phentax", delta_t=cfg.dt, f_min=cfg.f_min,
                              f_ref=cfg.f_ref, higher_modes=cfg.higher_modes)
    sim = LISASimulator(orbit_path=cfg.orbit_path, n_ts=cfg.n_ts, dt=cfg.dt,
                        tdi_channels=cfg.tdi_channel,
                        params={p.name: p for p in cfg.params},
                        waveform_model=wf, add_noise=False,
                        signal_chunk_size=chunk)
    return np.asarray(sim.forward(torch.tensor(thetas, dtype=torch.float64)))


s1, s8 = run(1).astype(np.float64), run(8).astype(np.float64)  # float32 squares underflow
d = np.sqrt(((s1 - s8) ** 2).sum(axis=(1, 2))) / np.sqrt((s1**2).sum(axis=(1, 2)))
amax = np.abs(s1 - s8).max(axis=(1, 2)) / np.abs(s1).max(axis=(1, 2))
print("CHUNKSIZE 1 vs 8 (same code, accepted as benign by CLAUDE.md):")
print("  relative L2:    median %.3e  max %.3e" % (np.median(d), d.max()))
print("  max|dh|/max|h|: median %.3e  max %.3e" % (np.median(amax), amax.max()))
print("  bitwise identical: %d/24" % int((d == 0).sum()))
