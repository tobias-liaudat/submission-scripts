"""Is the EQX_ON_ERROR=nan output for the failing vector silently wrong?

Compares the bad vector against a good neighbour (same params, eta nudged into
the converging region) to see whether the returned signal is plausible or junk.
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

with h5py.File(f"{T}/data/test_jean_zay_dataset_phentax_hm.h5", "r") as f:
    cfg = DataGenerationConfig.model_validate(json.loads(f.attrs["config"]))
    n_total = f["xs"].shape[0]
    idx = np.arange(n_total)
    np.random.default_rng(0).shuffle(idx)
    thetas_test = f["thetas"][sorted(idx[int(n_total * 0.9):][:100])]

names = [p.name for p in cfg.params]
i_eta, i_chi = names.index("reduced_mass"), names.index("chi1z")

wf = build_waveform_model(
    "phentax", delta_t=cfg.dt, f_min=cfg.f_min, f_ref=cfg.f_ref,
    higher_modes=cfg.higher_modes,
)
sim = LISASimulator(
    orbit_path=cfg.orbit_path, n_ts=cfg.n_ts, dt=cfg.dt,
    tdi_channels=cfg.tdi_channel, params={p.name: p for p in cfg.params},
    waveform_model=wf, add_noise=False, signal_chunk_size=1,
)

base = thetas_test[0].astype(np.float64).copy()
base[i_chi] = 0.5775
cases = {
    "BAD  (eta=0.1008, solver fails)": base.copy(),
    "GOOD (eta=0.1450, solver converges)": None,
    "GOOD (eta=0.2000, solver converges)": None,
}
g1 = base.copy(); g1[i_eta] = 0.145
g2 = base.copy(); g2[i_eta] = 0.200
cases["GOOD (eta=0.1450, solver converges)"] = g1
cases["GOOD (eta=0.2000, solver converges)"] = g2

for label, th in cases.items():
    out = np.asarray(sim.forward(torch.tensor(th[None, :], dtype=torch.float64)))[0]
    nz = np.abs(out) > 0
    print(
        f"{label}: |x|max={np.abs(out).max():.4e} "
        f"rms={np.sqrt((out**2).mean()):.4e} "
        f"nonzero_frac={nz.mean():.4f} allzero={not nz.any()}"
    )
