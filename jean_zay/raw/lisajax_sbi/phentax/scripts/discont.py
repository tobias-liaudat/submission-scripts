"""Is the f_min crossing a genuine root, or the tEarly shift discontinuity?"""

import json
import os

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import phentax.core.phase as phase  # noqa: E402

_REAL = phase.get_time_of_frequency
phase.get_time_of_frequency = lambda freq, eta, phase_coeffs, t_low=0.0, **kw: jnp.asarray(-1.0)

from phentax.core.phase import compute_phase_coeffs_22, imr_omega  # noqa: E402
from phentax.waveform import IMRPhenomTHM  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))  # tracked repo dir
DATA_IN = os.path.join(HERE, "..", "data")         # tracked inputs
T = "/lustre/fswork/projects/rech/ney/ulx23va/projects/LISA/outputs/testing"  # outputs
with open(os.path.join(DATA_IN, "failing_theta.json")) as f:
    th = json.load(f)

imr = IMRPhenomTHM(higher_modes=[21, 33, 44, 55], include_negative_modes=True)


def build(eta):
    delta = np.sqrt(max(1.0 - 4.0 * eta, 0.0))
    m1 = th["total_mass"] * (1 + delta) / 2
    m2 = th["total_mass"] * (1 - delta) / 2
    wp = imr._process_parameters(
        m1=jnp.asarray(m1), m2=jnp.asarray(m2),
        chi1z=jnp.asarray(0.5775), chi2z=jnp.asarray(th["chi2z"]),
        distance=jnp.asarray(1.0e3), phi_ref=jnp.asarray(th["initial_phase"]),
        inclination=jnp.asarray(th["inclination"]), psi=jnp.asarray(th["psi"]),
        delta_t=2.0, f_min=1.0e-4, f_ref=1.0e-4,
    )
    wp, co = jax.vmap(compute_phase_coeffs_22)(wp)
    first = lambda x: x[0] if getattr(x, "ndim", 0) else x  # noqa: E731
    return jax.tree.map(first, wp), jax.tree.map(first, co)


for label, eta in (("FAILING eta=0.10077", th["reduced_mass"]), ("WORKING eta=0.145", 0.145)):
    wp, co = build(eta)
    freq, eta_v = float(wp.Mf_min), float(wp.eta)
    tEarly, tt0 = float(co.tEarly), float(co.tt0)

    def g(t):
        time = jnp.where(t < tEarly, t - tt0, t)
        return 2 * jnp.pi * freq - imr_omega(time, eta_v, co)

    # Bisect for the sign change on the true bracket.
    lo, hi = -0.015 * freq ** (-2.7), 500.0
    glo = float(g(lo))
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if (float(g(mid)) > 0) == (glo > 0):
            lo = mid
        else:
            hi = mid
    root = 0.5 * (lo + hi)

    # Continuity across the located crossing.
    eps = max(abs(root) * 1e-9, 1e-6)
    gl, gr = float(g(root - eps)), float(g(root + eps))

    print(f"\n=== {label} ===")
    print(f"  tEarly = {tEarly:.6e}   tt0 = {tt0:.6e}")
    print(f"  located crossing t* = {root:.6e}")
    print(f"  |t* - tEarly| = {abs(root - tEarly):.6e}   "
          f"crossing IS the tEarly jump: {abs(root - tEarly) < max(abs(tEarly)*1e-6, 1e-3)}")
    print(f"  g(t*-eps) = {gl:.6e}   g(t*+eps) = {gr:.6e}   jump = {abs(gr - gl):.6e}")
    print(f"  min |g| attainable near crossing = {min(abs(gl), abs(gr)):.6e}")
    print(f"  -> can |g| < atol=1e-12 ever hold? {min(abs(gl), abs(gr)) < 1e-12}")
