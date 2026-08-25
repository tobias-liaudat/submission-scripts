"""Record the root values and TDI signals that the phentax fix could perturb.

Run once before patching (tag "base") and once after (tag "fix"), then diff with
``compare_roots.py --diff base fix``.

``Mt_min`` / ``Mt_ref`` are the direct output of ``get_time_of_frequency``, so
they are far more sensitive to the change than the downstream waveform; the TDI
signals are recorded too because that is what actually reaches the datasets.
"""

import argparse
import json
import os

import jax

jax.config.update("jax_enable_x64", True)

import h5py  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from phentax.core.phase import compute_phase_coeffs_22  # noqa: E402

from lisajax_sbi.simulator import LISASimulator  # noqa: E402
from lisajax_sbi.utils import DataGenerationConfig  # noqa: E402
from lisajax_sbi.waveforms import build_waveform_model  # noqa: E402

T = "/lustre/fswork/projects/rech/ney/ulx23va/projects/LISA/outputs/testing"
DATASET = f"{T}/data/test_jean_zay_dataset_phentax_hm.h5"
N_REF = 24


def _load_reference_thetas():
    with h5py.File(DATASET, "r") as f:
        cfg = DataGenerationConfig.model_validate(json.loads(f.attrs["config"]))
        n_total = f["xs"].shape[0]
        idx = np.arange(n_total)
        np.random.default_rng(0).shuffle(idx)
        test_idx = idx[int(n_total * 0.8) + int(n_total * 0.1) :]
        thetas = f["thetas"][sorted(test_idx[:100])]
    return cfg, thetas[:N_REF].astype(np.float64)


def capture(tag):
    cfg, thetas = _load_reference_thetas()
    names = [p.name for p in cfg.params]
    i_M, i_eta = names.index("total_mass"), names.index("reduced_mass")
    i_c1, i_c2 = names.index("chi1z"), names.index("chi2z")
    i_inc, i_psi = names.index("inclination"), names.index("psi")
    i_phi = names.index("initial_phase")

    wf = build_waveform_model(
        "phentax",
        delta_t=cfg.dt,
        f_min=cfg.f_min,
        f_ref=cfg.f_ref,
        higher_modes=cfg.higher_modes,
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

    # ---- root values: Mt_min / Mt_ref straight out of the root finder ----
    mt_min = np.zeros(N_REF)
    mt_ref = np.zeros(N_REF)
    root_ok = np.zeros(N_REF, dtype=bool)
    for i, th in enumerate(thetas):
        eta = th[i_eta]
        delta = np.sqrt(max(1.0 - 4.0 * eta, 0.0))
        m1 = th[i_M] * (1 + delta) / 2
        m2 = th[i_M] * (1 - delta) / 2
        try:
            wp = wf._imr._process_parameters(
                m1=jnp.asarray(m1),
                m2=jnp.asarray(m2),
                chi1z=jnp.asarray(th[i_c1]),
                chi2z=jnp.asarray(th[i_c2]),
                distance=jnp.asarray(1.0e3),  # amplitude only; irrelevant to the root
                phi_ref=jnp.asarray(th[i_phi]),
                inclination=jnp.asarray(th[i_inc]),
                psi=jnp.asarray(th[i_psi]),
                delta_t=cfg.dt,
                f_min=cfg.f_min,
                f_ref=cfg.f_ref,
            )
            wp, _ = jax.vmap(compute_phase_coeffs_22)(wp)
            mt_min[i] = float(np.asarray(wp.Mt_min).ravel()[0])
            mt_ref[i] = float(np.asarray(wp.Mt_ref).ravel()[0])
            root_ok[i] = True
        except Exception as exc:  # noqa: BLE001
            print(f"  source {i}: root FAILED ({type(exc).__name__})")

    print(f"ROOTS captured: {root_ok.sum()}/{N_REF}")

    # ---- TDI signals -----------------------------------------------------
    sigs = np.zeros((N_REF, 3, cfg.n_ts))
    sig_ok = np.zeros(N_REF, dtype=bool)
    for i, th in enumerate(thetas):
        try:
            sigs[i] = np.asarray(
                sim.forward(torch.tensor(th[None, :], dtype=torch.float64))
            )[0]
            sig_ok[i] = True
        except Exception as exc:  # noqa: BLE001
            print(f"  source {i}: signal FAILED ({type(exc).__name__})")
    print(f"SIGNALS captured: {sig_ok.sum()}/{N_REF}")

    out = f"{T}/hm_diag/roots_{tag}.npz"
    np.savez_compressed(
        out,
        thetas=thetas,
        mt_min=mt_min,
        mt_ref=mt_ref,
        root_ok=root_ok,
        sigs=sigs,
        sig_ok=sig_ok,
    )
    print("SAVED", out)


def diff(tag_a, tag_b):
    a = np.load(f"{T}/hm_diag/roots_{tag_a}.npz")
    b = np.load(f"{T}/hm_diag/roots_{tag_b}.npz")

    both = a["root_ok"] & b["root_ok"]
    print(f"\n=== ROOT VALUES ({both.sum()} sources solvable in both) ===")
    for key in ("mt_min", "mt_ref"):
        d = np.abs(a[key][both] - b[key][both])
        scale = np.maximum(np.abs(a[key][both]), 1e-30)
        print(
            f"  {key}: max|delta| = {d.max():.3e}   "
            f"max relative = {(d / scale).max():.3e}"
        )

    print(f"\n=== NEWLY SOLVABLE (was failing, now works) ===")
    gained = (~a["root_ok"]) & b["root_ok"]
    lost = a["root_ok"] & (~b["root_ok"])
    print(f"  gained: {int(gained.sum())}   lost: {int(lost.sum())}")

    bs = a["sig_ok"] & b["sig_ok"]
    print(f"\n=== TDI SIGNALS ({bs.sum()} sources) ===")
    sa, sb = a["sigs"][bs], b["sigs"][bs]
    dl2 = np.sqrt(((sa - sb) ** 2).sum(axis=(1, 2)))
    nl2 = np.sqrt((sa**2).sum(axis=(1, 2)))
    rel_l2 = dl2 / np.maximum(nl2, 1e-300)
    amax = np.abs(sa - sb).max(axis=(1, 2)) / np.maximum(
        np.abs(sa).max(axis=(1, 2)), 1e-300
    )
    # Thresholds are the measured reproducibility floor of this codebase, not
    # arbitrary values.  CLAUDE.md notes signals are not bit-reproducible across
    # chunk sizes (2nd-gen TDI amplifies reassociation by ~1e5); noise_floor.py
    # quantifies that accepted difference on these same 24 sources as
    # relative L2 max 2.038e-04 and max|dh|/max|h| max 3.009e-03.  A change is
    # acceptable if it stays within that floor.
    FLOOR_L2, FLOOR_AMAX = 2.04e-4, 3.01e-3
    print(f"  relative L2:        max = {rel_l2.max():.3e}   (floor {FLOOR_L2:.2e})")
    print(f"  max|dh|/max|h|:     max = {amax.max():.3e}   (floor {FLOOR_AMAX:.2e})")
    print(f"  bitwise identical:  {int((dl2 == 0).sum())}/{int(bs.sum())}")
    print(f"  headroom vs floor:  {FLOOR_L2 / max(rel_l2.max(), 1e-300):.0f}x (L2), "
          f"{FLOOR_AMAX / max(amax.max(), 1e-300):.0f}x (amax)")

    ok = (rel_l2.max() <= FLOOR_L2) and (amax.max() <= FLOOR_AMAX) and lost.sum() == 0
    print(f"\nVERDICT: {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("tag", nargs="?", help="capture under this tag")
    p.add_argument("--diff", nargs=2, metavar=("A", "B"), help="diff two tags")
    args = p.parse_args()
    if args.diff:
        diff(*args.diff)
    else:
        capture(args.tag)
