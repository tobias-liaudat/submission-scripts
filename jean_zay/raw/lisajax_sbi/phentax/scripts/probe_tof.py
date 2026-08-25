"""Instrument phentax's time_of_freq at the failing parameters.

Rebuilds the exact input to the only optx.root_find in phentax
(get_time_of_frequency) and evaluates the residual it bisects,
    g(t) = 2*pi*Mf_min - imr_omega(t, eta, coeffs),
densely across the bracket, for a failing and a working eta.

get_time_of_frequency is stubbed out first so the coefficients can be built
without the root find raising.
"""

import json
import os

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import phentax.core.phase as phase  # noqa: E402

# Stub the root find so compute_phase_coeffs_22 cannot raise while we build
# the coefficients we want to inspect.
_REAL_ROOT_FIND = phase.get_time_of_frequency
phase.get_time_of_frequency = lambda freq, eta, phase_coeffs, t_low=0.0, **kw: jnp.asarray(
    -1.0
)

from phentax.core.phase import compute_phase_coeffs_22, imr_omega  # noqa: E402
from phentax.waveform import IMRPhenomTHM  # noqa: E402

MSUN_S = 4.925490947641267e-6
HERE = os.path.dirname(os.path.abspath(__file__))  # tracked repo dir
DATA_IN = os.path.join(HERE, "..", "data")         # tracked inputs
T = "/lustre/fswork/projects/rech/ney/ulx23va/projects/LISA/outputs/testing"  # outputs
BAD_CHI1Z = 0.5775

with open(os.path.join(DATA_IN, "failing_theta.json")) as f:
    th = json.load(f)

imr = IMRPhenomTHM(higher_modes=[21, 33, 44, 55], include_negative_modes=True)


def m1_m2(total_mass, eta):
    delta = np.sqrt(max(1.0 - 4.0 * eta, 0.0))
    return total_mass * (1 + delta) / 2, total_mass * (1 - delta) / 2


for label, eta in (("FAILING eta=0.1008", th["reduced_mass"]), ("WORKING eta=0.1450", 0.145)):
    m1, m2 = m1_m2(th["total_mass"], eta)
    wf_params = imr._process_parameters(
        m1=jnp.asarray(m1), m2=jnp.asarray(m2),
        chi1z=jnp.asarray(BAD_CHI1Z), chi2z=jnp.asarray(th["chi2z"]),
        distance=jnp.asarray(1.0e3), phi_ref=jnp.asarray(th["initial_phase"]),
        inclination=jnp.asarray(th["inclination"]), psi=jnp.asarray(th["psi"]),
        delta_t=2.0, f_min=1.0e-4, f_ref=1.0e-4,
    )
    wf_params, coeffs = jax.vmap(compute_phase_coeffs_22)(wf_params)
    wf_params = jax.tree.map(lambda x: x[0] if getattr(x, "ndim", 0) else x, wf_params)
    coeffs = jax.tree.map(lambda x: x[0] if getattr(x, "ndim", 0) else x, coeffs)

    freq = float(wf_params.Mf_min)
    eta_v = float(wf_params.eta)
    t_low_cfg = float(wf_params.t_low)
    t_low = -0.015 * freq ** (-2.7) if t_low_cfg == 0 else t_low_cfg
    t_high = 500.0

    def g(t):
        time = jnp.where(t < coeffs.tEarly, t - coeffs.tt0, t)
        return 2 * jnp.pi * freq - imr_omega(time, eta_v, coeffs)

    ts = jnp.concatenate([
        -jnp.geomspace(abs(t_low), 1e-3, 4000)[::-1] if t_low < 0 else jnp.linspace(t_low, 0, 4000),
        jnp.linspace(0.0, t_high, 2000),
    ])
    vals = np.asarray(jax.vmap(g)(ts))
    ts_np = np.asarray(ts)
    finite = np.isfinite(vals)

    print(f"\n=== {label} ===")
    print(f"  Mf_min={freq:.6e}  eta={eta_v:.5f}  bracket=[{t_low:.6e}, {t_high:.1f}]")
    print(f"  g finite: {finite.sum()}/{len(vals)}  ({100*(~finite).mean():.2f}% NaN/inf)")
    if finite.any():
        print(f"  g(t_low)={vals[0]:.6e}   g(t_high)={vals[-1]:.6e}")
        print(f"  g range: [{np.nanmin(vals):.6e}, {np.nanmax(vals):.6e}]")
        sgn = np.sign(vals[finite])
        crossings = int((np.diff(sgn) != 0).sum())
        print(f"  sign changes across sampled bracket: {crossings}")
        print(f"  ENDPOINT SIGNS: {'OPPOSITE (root bracketed)' if vals[0]*vals[-1] < 0 else 'SAME (no bracketed root)'}")
    # Locate the sign changes and the positive excursion.
    sgn = np.sign(vals)
    idx = np.where(np.diff(sgn) != 0)[0]
    for k in idx:
        print(f"  sign change between t={ts_np[k]:.6e} and t={ts_np[k+1]:.6e}"
              f"  (g: {vals[k]:.4e} -> {vals[k+1]:.4e})")
    pos = vals > 0
    if pos.any():
        tp = ts_np[pos]
        print(f"  POSITIVE excursion: t in [{tp.min():.6e}, {tp.max():.6e}] "
              f"width={tp.max()-tp.min():.6e}, peak g={vals.max():.4e}")
        print(f"  fraction of sampled bracket where g>0: {pos.mean()*100:.4f}%")
    # What bisection actually does: midpoint of [lower, upper], repeatedly.
    lo, hi = t_low, t_high
    for step in range(60):
        mid = 0.5 * (lo + hi)
        gm = float(g(mid))
        if step < 6:
            print(f"  bisect step {step}: mid={mid:.6e} g(mid)={gm:.4e}")
        if gm > 0:
            print(f"  -> midpoint sequence REACHED the positive region at step {step}")
            break
        lo = mid
    else:
        print("  -> midpoint sequence NEVER reached the positive region (bisection cannot converge)")
