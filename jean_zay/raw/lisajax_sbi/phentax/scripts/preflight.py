"""Pre-flight checks that decide details of the get_time_of_frequency rewrite.

1. flip orientation  -- is g decreasing in t (=> Bisection(flip=True))?
2. segment continuity -- is tEarly the ONLY jump, or do the imr_omega ansatz
   cuts (inspiral_cut / ringdown_cut) add more?  Decides 2 branches vs ~4.
3. t_low_fit=False    -- documented knob widening the bracket to -1e9; does it
   fix the failing vector on its own (zero-code alternative)?
"""

import json
import os

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import phentax.core.phase as phase  # noqa: E402

_REAL = phase.get_time_of_frequency
phase.get_time_of_frequency = (
    lambda freq, eta, phase_coeffs, t_low=0.0, **kw: jnp.asarray(-1.0)
)

from phentax.core.phase import compute_phase_coeffs_22, imr_omega  # noqa: E402
from phentax.waveform import IMRPhenomTHM  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))  # tracked repo dir
DATA_IN = os.path.join(HERE, "..", "data")         # tracked inputs
T = "/lustre/fswork/projects/rech/ney/ulx23va/projects/LISA/outputs/testing"  # outputs
with open(os.path.join(DATA_IN, "failing_theta.json")) as f:
    TH = json.load(f)
BAD_CHI1Z = 0.5775


def build(eta, chi1z=BAD_CHI1Z, imr=None):
    imr = imr or IMRPhenomTHM(higher_modes=[21, 33, 44, 55], include_negative_modes=True)
    delta = np.sqrt(max(1.0 - 4.0 * eta, 0.0))
    wp = imr._process_parameters(
        m1=jnp.asarray(TH["total_mass"] * (1 + delta) / 2),
        m2=jnp.asarray(TH["total_mass"] * (1 - delta) / 2),
        chi1z=jnp.asarray(chi1z),
        chi2z=jnp.asarray(TH["chi2z"]),
        distance=jnp.asarray(1.0e3),
        phi_ref=jnp.asarray(TH["initial_phase"]),
        inclination=jnp.asarray(TH["inclination"]),
        psi=jnp.asarray(TH["psi"]),
        delta_t=2.0,
        f_min=1.0e-4,
        f_ref=1.0e-4,
    )
    wp, co = jax.vmap(compute_phase_coeffs_22)(wp)
    first = lambda x: x[0] if getattr(x, "ndim", 0) else x  # noqa: E731
    return jax.tree.map(first, wp), jax.tree.map(first, co)


print("=" * 62)
print("CHECK 1: flip orientation  (need g(lower) > g(upper) => flip=True)")
print("=" * 62)
for label, eta, chi in (
    ("failing", TH["reduced_mass"], BAD_CHI1Z),
    ("working", 0.145, BAD_CHI1Z),
    ("typical", 0.20, 0.0),
    ("neg spin", 0.18, -0.7),
):
    wp, co = build(eta, chi)
    freq, eta_v = float(wp.Mf_min), float(wp.eta)
    tE, tt0 = float(co.tEarly), float(co.tt0)
    t_low = -0.015 * freq ** (-2.7)

    def g(t):
        time = jnp.where(t < tE, t - tt0, t)
        return 2 * jnp.pi * freq - imr_omega(time, eta_v, co)

    glo, ghi = float(g(t_low)), float(g(500.0))
    print(
        f"  {label:9s} eta={eta_v:.5f} chi1z={chi:+.2f}: "
        f"g(lo)={glo:+.4e} g(hi)={ghi:+.4e} -> flip={glo > ghi}"
    )

print()
print("=" * 62)
print("CHECK 2: segment continuity  (is tEarly the only jump?)")
print("=" * 62)
for label, eta, chi in (
    ("failing", TH["reduced_mass"], BAD_CHI1Z),
    ("typical", 0.20, 0.0),
):
    wp, co = build(eta, chi)
    freq, eta_v = float(wp.Mf_min), float(wp.eta)
    tE, tt0 = float(co.tEarly), float(co.tt0)
    icut, rcut = float(co.inspiral_cut), float(co.ringdown_cut)
    t_low = -0.015 * freq ** (-2.7)
    target = 2 * jnp.pi * freq

    def g_early(t):
        return target - imr_omega(t - tt0, eta_v, co)

    def g_late(t):
        return target - imr_omega(t, eta_v, co)

    print(f"\n  --- {label} (eta={eta_v:.5f}) ---")
    print(f"      tEarly={tE:.6e} inspiral_cut={icut:.6e} ringdown_cut={rcut:.6e}")
    for name, fn, lo, hi in (
        ("early [t_low, tEarly]", g_early, t_low, tE),
        ("late  [tEarly, 500]", g_late, tE, 500.0),
    ):
        ts = jnp.linspace(lo, hi, 20000)
        v = np.asarray(jax.vmap(fn)(ts))
        tsn = np.asarray(ts)
        d = np.abs(np.diff(v))
        # a jump = a step far larger than its neighbours
        med = np.median(d[d > 0]) if (d > 0).any() else 0.0
        jumps = np.where(d > max(200 * med, 1e-12))[0]
        print(
            f"      {name}: finite={np.isfinite(v).all()} "
            f"sign_changes={int((np.diff(np.sign(v)) != 0).sum())} "
            f"suspect_jumps={len(jumps)}"
        )
        for k in jumps[:4]:
            near = []
            for cname, cval in (("tEarly", tE), ("inspiral_cut", icut), ("ringdown_cut", rcut)):
                if abs(tsn[k] - cval) < abs(hi - lo) / 1000:
                    near.append(cname)
            print(
                f"        jump at t={tsn[k]:.6e} size={d[k]:.3e}"
                f"{'  near ' + ','.join(near) if near else ''}"
            )

print()
print("=" * 62)
print("CHECK 3: t_low_fit=False  (bracket -1e9; zero-code alternative?)")
print("=" * 62)
phase.get_time_of_frequency = _REAL  # restore the real solver for this check

from lisajax_sbi.simulator import LISASimulator  # noqa: E402
from lisajax_sbi.utils import DataGenerationConfig  # noqa: E402
import h5py  # noqa: E402

with h5py.File(f"{T}/data/test_jean_zay_dataset_phentax_hm.h5", "r") as f:
    cfg = DataGenerationConfig.model_validate(json.loads(f.attrs["config"]))
    n_total = f["xs"].shape[0]
    idx = np.arange(n_total)
    np.random.default_rng(0).shuffle(idx)
    thetas_test = f["thetas"][sorted(idx[int(n_total * 0.9) :][:100])]

names = [p.name for p in cfg.params]
bad = thetas_test[0].astype(np.float64).copy()
bad[names.index("chi1z")] = BAD_CHI1Z

for t_low_fit in (True, False):
    from lisajax_sbi.waveforms import build_waveform_model

    wf = build_waveform_model(
        "phentax", delta_t=cfg.dt, f_min=cfg.f_min, f_ref=cfg.f_ref,
        higher_modes=cfg.higher_modes,
    )
    object.__setattr__(wf._imr, "t_low", 0.0 if t_low_fit else -1e9)
    sim = LISASimulator(
        orbit_path=cfg.orbit_path, n_ts=cfg.n_ts, dt=cfg.dt,
        tdi_channels=cfg.tdi_channel, params={p.name: p for p in cfg.params},
        waveform_model=wf, add_noise=False, signal_chunk_size=1,
    )
    try:
        out = np.asarray(sim.forward(torch.tensor(bad[None, :], dtype=torch.float64)))
        print(f"  t_low_fit={t_low_fit} (t_low={wf._imr.t_low:.3e}): "
              f"OK |x|max={np.abs(out).max():.3e}")
    except Exception as exc:  # noqa: BLE001
        print(f"  t_low_fit={t_low_fit} (t_low={wf._imr.t_low:.3e}): "
              f"FAIL {type(exc).__name__}")
