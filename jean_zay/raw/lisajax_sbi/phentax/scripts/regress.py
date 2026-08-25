"""Baseline/after comparison for the phentax Bisection bracket patch.

Run once before patching (tag "before") and once after (tag "after"):

* signals for a set of currently-working test sources, to prove the patch is
  inert where the solver already converged;
* the known-failing parameter vector;
* an eta sweep at the failing chi1z.
"""

import json
import sys

import jax

jax.config.update("jax_enable_x64", True)

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from lisajax_sbi.simulator import LISASimulator  # noqa: E402
from lisajax_sbi.utils import DataGenerationConfig  # noqa: E402
from lisajax_sbi.waveforms import build_waveform_model  # noqa: E402

TAG = sys.argv[1]
T = "/lustre/fswork/projects/rech/ney/ulx23va/projects/LISA/outputs/testing"
N_REF = 24
BAD_CHI1Z = 0.5775

with h5py.File(f"{T}/data/test_jean_zay_dataset_phentax_hm.h5", "r") as f:
    cfg = DataGenerationConfig.model_validate(json.loads(f.attrs["config"]))
    n_total = f["xs"].shape[0]
    idx = np.arange(n_total)
    np.random.default_rng(0).shuffle(idx)
    thetas_test = f["thetas"][sorted(idx[int(n_total * 0.9):][:100])]

names = [p.name for p in cfg.params]
print("higher_modes:", cfg.higher_modes)

wf = build_waveform_model(
    "phentax", delta_t=cfg.dt, f_min=cfg.f_min, f_ref=cfg.f_ref,
    higher_modes=cfg.higher_modes,
)
sim = LISASimulator(
    orbit_path=cfg.orbit_path, n_ts=cfg.n_ts, dt=cfg.dt,
    tdi_channels=cfg.tdi_channel, params={p.name: p for p in cfg.params},
    waveform_model=wf, add_noise=False, signal_chunk_size=1,
)


def evaluate(theta):
    try:
        out = np.asarray(
            sim.forward(torch.tensor(theta[None, :], dtype=torch.float64))
        )
        return out[0], True
    except Exception:  # noqa: BLE001
        return np.zeros((3, cfg.n_ts)), False


# 1. Reference sources that already work -> must be unchanged by the patch.
ref = np.zeros((N_REF, 3, cfg.n_ts))
ref_ok = np.zeros(N_REF, dtype=bool)
for i in range(N_REF):
    ref[i], ref_ok[i] = evaluate(thetas_test[i].astype(np.float64))
print(f"REF sources evaluated: {ref_ok.sum()}/{N_REF} ok")

# 2. The known-failing vector.
bad = thetas_test[0].astype(np.float64).copy()
bad[names.index("chi1z")] = BAD_CHI1Z
bad_sig, bad_ok = evaluate(bad)
print(f"BAD vector (chi1z={BAD_CHI1Z}): {'OK' if bad_ok else 'FAIL'}")

# 3. Eta sweep at the failing chi1z.
etas = np.round(np.arange(0.100, 0.161, 0.005), 4)
sweep_ok = np.zeros(len(etas), dtype=bool)
sweep_sig = np.zeros((len(etas), 3, cfg.n_ts))
for j, eta in enumerate(etas):
    th = bad.copy()
    th[names.index("reduced_mass")] = eta
    sweep_sig[j], sweep_ok[j] = evaluate(th)
print("ETA sweep ok:", dict(zip(etas.tolist(), sweep_ok.tolist())))

np.savez_compressed(
    f"{T}/hm_diag/regress_{TAG}.npz",
    ref=ref, ref_ok=ref_ok, bad=bad_sig, bad_ok=bad_ok,
    etas=etas, sweep_sig=sweep_sig, sweep_ok=sweep_ok,
)
print("SAVED", TAG)
