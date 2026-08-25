"""Does overriding IMRPhenomTHM.atol actually reach the root finder?"""

import json
import os

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from lisajax_sbi.waveforms import build_waveform_model  # noqa: E402

wf = build_waveform_model("phentax", delta_t=2.0, f_min=1e-4, f_ref=1e-4,
                          higher_modes=[21, 33, 44, 55])
imr = wf._imr
print("is eqx.Module:", hasattr(imr, "__dataclass_fields__"))
print("default atol/rtol:", imr.atol, imr.rtol)

try:
    object.__setattr__(imr, "atol", 1e-6)
    object.__setattr__(imr, "rtol", 1e-6)
    print("after override:  ", imr.atol, imr.rtol)
except Exception as exc:  # noqa: BLE001
    print("override failed:", type(exc).__name__, exc)

HERE = os.path.dirname(os.path.abspath(__file__))  # tracked repo dir
DATA_IN = os.path.join(HERE, "..", "data")         # tracked inputs
with open(os.path.join(DATA_IN, "failing_theta.json")) as f:
    th = json.load(f)

eta = th["reduced_mass"]
delta = np.sqrt(max(1.0 - 4.0 * eta, 0.0))
m1 = th["total_mass"] * (1 + delta) / 2
m2 = th["total_mass"] * (1 - delta) / 2

wf_params = imr._process_parameters(
    m1=jnp.asarray(m1), m2=jnp.asarray(m2),
    chi1z=jnp.asarray(0.5775), chi2z=jnp.asarray(th["chi2z"]),
    distance=jnp.asarray(1.0e3), phi_ref=jnp.asarray(th["initial_phase"]),
    inclination=jnp.asarray(th["inclination"]), psi=jnp.asarray(th["psi"]),
    delta_t=2.0, f_min=1.0e-4, f_ref=1.0e-4,
)
print("wf_params.atol =", np.asarray(wf_params.atol),
      " wf_params.rtol =", np.asarray(wf_params.rtol))
print("PROPAGATES:", bool(np.allclose(np.asarray(wf_params.atol), 1e-6)))
