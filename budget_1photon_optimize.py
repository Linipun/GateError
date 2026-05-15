"""
Bounded optimizer for the 1-photon Rydberg gate infidelity budget.

This is a refactor of budget_1photon_scan.py.  It can:
  1. optimize the parameters from the original "### parameters ####" section
  2. rerun a full Rabi scan at the best point
  3. save the same JSON-style output plus a contribution bar plot at the minimum

Examples
--------
Fast bounded optimization:
    python budget_1photon_optimize.py --mode optimize --opt-samples 500 --maxiter 50

Full final scan using the best parameters printed/saved by the optimizer:
    python budget_1photon_optimize.py --mode scan --n 60 --atom-d 5.0 --omega-rabi-mhz 10 --Bz 10 --w0-rydberg 20 --trap-depth 1000 --num-samples 10000

Optimize, then automatically do a final full scan:
    python budget_1photon_optimize.py --mode optimize_and_scan --opt-samples 500 --num-samples 10000 --maxiter 50
"""

from budget_monte_carlo import *
import phase_noise  # kept for compatibility with the original script
from linear_response import build_Oseq, response_G13, isometry_haar_full
import numpy as np
import matplotlib.pyplot as plt
from arc import *
import json
import scipy.optimize as opt
import pandas as pd
import sys
import os
import argparse
from functools import lru_cache

# -----------------------------
# Constants
# -----------------------------
h = 6.626e-34
e = 1.602e-19
a0 = 5.291e-11
hbar = h / 2 / np.pi
EH = 4.359744e-18
c = 299792458
kb = 1.380649e-23
me = 9.1093837e-31
epi0 = 8.854e-12
bohr_r = 5.291e-11


# -----------------------------
# Default fixed config
# -----------------------------
DEFAULT_CONFIG = dict(
    atom_name="Cs",
    l=1,
    j=3 / 2,
    mj=3 / 2,
    pulse_time=7.65,
    resolution=200,
    lambda_rydberg=0.319,  # um
    HF_split=2000 * np.pi * 2,  # MHz angular units, as in original script
    alpha_dc=None,  # MHz (V/cm)^-2; None means compute with ARC StarkMap
    T_atom=1.0,  # uK
    lambda_trap=1.064,  # um
    w0_trap=1.0,  # um
    Bz = 10,
    edc_fluc=1e-3,  # V/cm
    edc_zero=0.0,  # original variable name says V/m; original formula uses it directly
    bdc_fluc=1e-3,  # G
    rin_strength=1e-4,
    f_hz_hz2=220,
    f_range=1e5,  # Hz
)

# Parameters to optimize.  These are the variables from the original
# "### parameters ####" section.  Edit bounds here if your physical range changes.
PARAMETER_NAMES = [
    "n",
    "atom_d_um",
    "Omega_Rabi_MHz",
    # "Bz_G",
    "w0_rydberg_um",
    "trap_depth_uK",
]

DEFAULT_X0 = np.array([60, 5.0, 10.0, 10.0, 20.0, 1000.0], dtype=float)

DEFAULT_BOUNDS = [
    (40, 100),       # n, rounded to integer during evaluation
    (2.0, 15.0),     # atom_d_um
    (2.0, 40.0),     # Omega_Rabi_MHz
    # (0.1, 100.0),    # Bz_G
    (3.0, 80.0),     # w0_rydberg_um
    (20.0, 5000.0),  # trap_depth_uK
]


# -----------------------------
# Helpers
# -----------------------------
def _to_jsonable(x):
    """Convert numpy/pandas types to JSON-serializable python types."""
    try:
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return float(x)
        if isinstance(x, (np.ndarray,)):
            return x.tolist()
    except Exception:
        pass
    try:
        if isinstance(x, (pd.Timestamp,)):
            return x.isoformat()
    except Exception:
        pass
    return x


def vector_to_params(x):
    """Map optimizer vector to named physical parameters."""
    x = np.asarray(x, dtype=float)
    return dict(
        n=int(round(x[0])),
        atom_d=float(x[1]),
        Omega_Rabi=float(x[2]) * 2 * np.pi,  # MHz -> angular MHz units used by original code
        Omega_Rabi_MHz=float(x[2]),
        # Bz=float(x[3]),
        w0_rydberg=float(x[3]),
        trap_depth=float(x[4]),
    )


def params_to_vector(n, atom_d, omega_rabi_mhz, w0_rydberg, trap_depth):
    return np.array([n, atom_d, omega_rabi_mhz, w0_rydberg, trap_depth], dtype=float)


@lru_cache(maxsize=100000)
def cached_blockade_mrad(atom_name, n, distance_um_rounded):
    """Cache expensive blockade calculations.

    distance_um_rounded should be a rounded float to make cache hits possible.
    """
    return find_blockade_Mrad(atom_name, int(n), float(distance_um_rounded))


@lru_cache(maxsize=1024)
def cached_atom_quantities(atom_name, n, l, j, Bz_rounded, alpha_dc_input):
    """Cache atom object derived quantities for repeated optimizer calls."""
    if atom_name == "Rb":
        atom = Rubidium()
    elif atom_name == "Cs":
        atom = Caesium()
    else:
        raise ValueError(f"Unsupported atom_name={atom_name!r}; use 'Rb' or 'Cs'.")

    Bz = float(Bz_rounded)
    R_lifetime = atom.getStateLifetime(
        n=int(n), l=int(l), j=float(j), temperature=300, includeLevelsUpTo=int(n) + 20, s=0.5
    ) * 1e6
    m_atom = atom.mass

    if alpha_dc_input is None:
        calc = StarkMap(atom)
        calc.defineBasis(n=int(n), l=1, j=1.5, mj=1.5, nMin=int(n) - 20, nMax=int(n) + 30, maxL=5, Bz=Bz / 10000)
        calc.diagonalise(np.linspace(0, 60, 600))
        alpha_dc = calc.getPolarizability(debugOutput=False)
    else:
        alpha_dc = float(alpha_dc_input)

    return R_lifetime, m_atom, alpha_dc


def make_rng(seed):
    return np.random.default_rng(seed) if seed is not None else None


# -----------------------------
# Core simulation
# -----------------------------
def evaluate_single_point(
    x,
    config=None,
    num_samples=500,
    seed=1234,
    optimize_phase=True,
    verbose=False,
    return_details=False,
):
    """Return total infidelity for one parameter vector.

    This is the objective function used by scipy.optimize.minimize.
    It computes the same terms as the original script, but for one Omega_Rabi value.
    """
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)

    p = vector_to_params(x)
    atom_name = cfg["atom_name"]
    n = p["n"]
    atom_d = p["atom_d"]
    Omega_Rabi = p["Omega_Rabi"]
    # Bz = p["Bz"]
    w0_rydberg = p["w0_rydberg"]
    trap_depth = p["trap_depth"]

    # Penalize invalid/unphysical points defensively.
    if n < 1 or atom_d <= 0 or Omega_Rabi <= 0 or w0_rydberg <= 0 or trap_depth <= 0:
        return (1e9, {}) if return_details else 1e9

    if atom_name == "Rb":
        atom = Rubidium()
    elif atom_name == "Cs":
        atom = Caesium()
    else:
        raise ValueError(f"Unsupported atom_name={atom_name!r}; use 'Rb' or 'Cs'.")

    l = cfg["l"]
    j = cfg["j"]
    resolution = cfg["resolution"]
    pulse_time = cfg["pulse_time"]
    lambda_rydberg = cfg["lambda_rydberg"]
    lambda_trap = cfg["lambda_trap"]
    w0_trap = cfg["w0_trap"]
    T_atom = cfg["T_atom"]
    HF_split = cfg["HF_split"]
    Bz = cfg["Bz"]

    blockade_mrad = cached_blockade_mrad(atom_name, n, round(atom_d, 6))
    R_lifetime, m_atom, alpha_dc = cached_atom_quantities(
        atom_name, n, l, j, round(Bz, 6), cfg["alpha_dc"]
    )

    if HF_split is None:
        HF_split = (
            atom.getZeemanEnergyShift(l=1, j=3 / 2, mj=3 / 2, magneticFieldBz=Bz / 10000)
            - atom.getZeemanEnergyShift(l=1, j=3 / 2, mj=1 / 2, magneticFieldBz=Bz / 10000)
        ) / hbar / 1e6

    rng = make_rng(seed)
    sigma_r = sigma_r_um(T_atom, trap_depth, w0_trap)
    sigma_z = sigma_z_um(T_atom, trap_depth, w0_trap, lambda_trap)
    ds, c1, c2 = sample_pair_distances(
        n_samples=int(num_samples),
        sigma_r=sigma_r,
        sigma_z=sigma_z,
        x_offset=atom_d,
        rng=rng,
    )

    x1, y1, z1 = c1["x"], c1["y"], c1["z"]
    x2, y2, z2 = c2["x"], c2["y"], c2["z"]

    rabi_center = np.sqrt(relative_gaussian_beam_intensity(0, 0, -atom_d / 2, w0_rydberg, lambda_rydberg))
    rabis1 = np.sqrt(relative_gaussian_beam_intensity(x1, z1, y1 - atom_d / 2, w0_rydberg, lambda_rydberg)) / rabi_center
    rabis2 = np.sqrt(relative_gaussian_beam_intensity(x2 - atom_d, z2, y2 + atom_d / 2, w0_rydberg, lambda_rydberg)) / rabi_center

    # Phase optimization at this point.
    H_gen1 = Hamiltonians(
        Omega_Rabi1=Omega_Rabi,
        blockade_inf=False,
        blockade=blockade_mrad,
        r_lifetime=R_lifetime,
        Delta1=0,
        Stark1=0,
        Stark2=0,
        resolution=resolution,
        r_lifetime2=R_lifetime,
        pulse_time=pulse_time,
    )
    PhaseGuess = [2 * np.pi * 0.1122, 1.0431, -0.7318, 0]

    if optimize_phase:
        opt_out = opt.minimize(fun=fid_optimize, x0=PhaseGuess, args=(H_gen1,), method="Nelder-Mead")
        phase_params = opt_out.x
    else:
        phase_params = PhaseGuess

    H_gen1 = Hamiltonians(
        Omega_Rabi1=Omega_Rabi,
        blockade_inf=False,
        blockade=blockade_mrad,
        r_lifetime=1e10,
        Delta1=0,
        Stark1=0,
        Stark2=0,
        resolution=resolution,
        r_lifetime2=1e10,
        pulse_time=pulse_time,
    )
    time, phase, dt = phase_cosine_generate(*phase_params, H_gen1.pulse_time, H_gen1.resolution)
    fid1, global_phi = H_gen1.return_fidel(phases=phase, dt=dt)
    infid_TO1 = 1 - fid1

    # Linear response terms.
    S_haar = isometry_haar_full()
    o_f = build_Oseq(phases=phase, dt=dt, B=blockade_mrad, is_intensity=False)
    o_I = build_Oseq(phases=phase, dt=dt, B=blockade_mrad, is_intensity=True)

    Ii = response_G13(o_I, S_haar, 0, dt=dt)
    RIN_contribution = Ii * cfg["rin_strength"]

    If_1 = response_G13(o_f, S_haar, 0, dt=dt) / Omega_Rabi**2
    v_contribution = If_1 * cfg["f_hz_hz2"] * cfg["f_range"] / 1e6 / 1e6

    infids_decay = (2.95 / Omega_Rabi) / R_lifetime

    # Detuning contribution.
    doppler_shift = 1 / lambda_trap / 1e-6 * np.sqrt(kb * (T_atom * 1e-6) / m_atom)
    delta_edc = abs(-0.5 * alpha_dc * 1e6 * ((cfg["edc_zero"] + cfg["edc_fluc"]) ** 2 - cfg["edc_zero"] ** 2)) * 2 * np.pi
    delta_bdc = atom.getZeemanEnergyShift(l=1, j=3 / 2, mj=3 / 2, magneticFieldBz=cfg["bdc_fluc"] / 10000) / hbar
    total_shift = np.sqrt(delta_bdc**2 + delta_edc**2 + doppler_shift**2)
    detunings = total_shift / 1e6 / Omega_Rabi

    infids_s = []
    for _ in range(int(num_samples)):
        d = sample_gaussian(detunings)
        H_gen = Hamiltonians(
            Omega_Rabi1=Omega_Rabi,
            blockade_inf=False,
            blockade=blockade_mrad,
            r_lifetime=10e9,
            Delta1=d,
            Stark1=0,
            Stark2=0,
            resolution=resolution,
            r_lifetime2=10e9,
            pulse_time=pulse_time,
        )
        fid, global_phi = H_gen.return_fidel(phases=phase, dt=dt)
        infids_s.append(1 - fid)
    infids_s = np.asarray(infids_s)
    infids_detuning = np.mean(infids_s) - infid_TO1

    if total_shift > 0:
        infids_bdc = delta_bdc**2 / total_shift**2 * infids_detuning
        infids_edc = delta_edc**2 / total_shift**2 * infids_detuning
        infids_doppler = doppler_shift**2 / total_shift**2 * infids_detuning
    else:
        infids_bdc = infids_edc = infids_doppler = 0.0

    # Motion: blockade and Rabi inhomogeneity.
    infids_blockade = []
    infids_rabi = []
    for rabi1, rabi2, d in zip(rabis1, rabis2, ds):
        blockade = cached_blockade_mrad(atom_name, n, round(float(d), 6))

        H_gen_block = Hamiltonians(
            Omega_Rabi1=Omega_Rabi,
            blockade_inf=False,
            blockade=blockade,
            r_lifetime=10e9,
            Delta1=0,
            Stark1=0,
            Stark2=0,
            resolution=resolution,
            r_lifetime2=10e9,
            pulse_time=pulse_time,
        )
        fid, global_phi = H_gen_block.asym_return_fidel(phases=phase, dt=dt, omega1_scale=1, omega2_scale=1)
        infids_blockade.append(1 - fid)

        H_gen_rabi = Hamiltonians(
            Omega_Rabi1=Omega_Rabi,
            blockade_inf=False,
            blockade=blockade_mrad,
            r_lifetime=10e9,
            Delta1=0,
            Stark1=0,
            Stark2=0,
            resolution=resolution,
            r_lifetime2=10e9,
            pulse_time=pulse_time,
        )
        fid, global_phi = H_gen_rabi.asym_return_fidel(phases=phase, dt=dt, omega1_scale=rabi1, omega2_scale=rabi2)
        infids_rabi.append(1 - fid)

    infids_motion_blockade = np.mean(np.asarray(infids_blockade)) - infid_TO1
    infids_motion_rabi = np.mean(np.asarray(infids_rabi)) - infid_TO1

    # Leakage.
    H_leak = LeakageHamiltonians(
        Omega_Rabi1=Omega_Rabi,
        blockade_inf=False,
        blockade=blockade_mrad,
        r_lifetime=10e9,
        Delta1=0,
        Stark1=0,
        Stark2=0,
        resolution=resolution,
        r_lifetime2=10e9,
        pulse_time=pulse_time,
        mj12_split=HF_split,
    )
    fid_with_leak, global_phi = H_leak.return_fidel(phases=phase, dt=dt)
    leakage_mj = (1 - fid_with_leak) - infid_TO1

    scattering = 0.0
    total = (
        infid_TO1
        + infids_motion_blockade
        + infids_motion_rabi
        + leakage_mj
        + infids_detuning
        + v_contribution
        + RIN_contribution
        + infids_decay
    )

    details = dict(
        TO=float(infid_TO1),
        blockade=float(infids_motion_blockade),
        rabi=float(infids_motion_rabi),
        leakage=float(leakage_mj),
        detuning=float(infids_detuning),
        efield=float(infids_edc),
        bfield=float(infids_bdc),
        doppler=float(infids_doppler),
        vnoise=float(v_contribution),
        RIN=float(RIN_contribution),
        decay=float(infids_decay),
        scattering=float(scattering),
        total=float(total),
        phase_params=_to_jsonable(np.asarray(phase_params)),
        blockade_mrad=float(blockade_mrad),
        blockade_MHz=float(blockade_mrad / 2 / np.pi),
        R_lifetime_us=float(R_lifetime),
        alpha_dc=float(alpha_dc),
    )

    if verbose:
        print(
            f"n={n}, d={atom_d:.4g} um, Omega/2pi={p['Omega_Rabi_MHz']:.4g} MHz, "
            f"Bz={Bz:.4g} G, w0={w0_rydberg:.4g} um, U={trap_depth:.4g} uK -> total={total:.6e}"
        )

    return (float(total), details) if return_details else float(total)


def run_scan(
    x,
    config=None,
    num_samples=10000,
    seed=1234,
    result_dir="result",
    f_rabis=None,
    optimize_phase=True,
    make_plots=True,
    verbose=True,
):
    """Run an Omega_Rabi scan at fixed non-Omega parameters and save JSON/plots."""
    os.makedirs(result_dir, exist_ok=True)
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)

    p0 = vector_to_params(x)
    if f_rabis is None:
        f_rabis = np.linspace(2, 40, 20)
    f_rabis = np.asarray(f_rabis, dtype=float)

    records = []
    for f in f_rabis:
        xx = np.asarray(x, dtype=float).copy()
        xx[2] = f
        total, details = evaluate_single_point(
            xx,
            config=cfg,
            num_samples=num_samples,
            seed=seed,
            optimize_phase=optimize_phase,
            verbose=verbose,
            return_details=True,
        )
        details["Omega_Rabi_MHz"] = float(f)
        records.append(details)

    def arr(key):
        return np.asarray([r[key] for r in records], dtype=float)

    y = dict(
        TO=arr("TO"),
        vnoise=arr("vnoise"),
        RIN=arr("RIN"),
        blockade=arr("blockade"),
        efield=arr("efield"),
        bfield=arr("bfield"),
        doppler=arr("doppler"),
        rabi=arr("rabi"),
        decay=arr("decay"),
        leakage=arr("leakage"),
        scattering=arr("scattering"),
        total=arr("total"),
    )

    min_idx = int(np.argmin(y["total"]))
    min_total = float(y["total"][min_idx])

    if make_plots:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(f_rabis, y["vnoise"], linewidth=2, label="$v$")
        ax.plot(f_rabis, y["RIN"], linewidth=2, label="RIN")
        ax.plot(f_rabis, y["blockade"], linewidth=2, label="$\\delta V$")
        ax.plot(f_rabis, y["rabi"], linewidth=2, label="$\\delta \\Omega$")
        ax.plot(f_rabis, y["decay"], linewidth=2, label="$\\gamma$")
        ax.plot(f_rabis, y["total"], linewidth=4, label="$\\Sigma$")
        ax.plot(f_rabis, y["leakage"], linewidth=2, label="leakage")
        ax.plot(f_rabis, y["efield"], linewidth=2, label="E")
        ax.plot(f_rabis, y["bfield"], linewidth=2, label="B")
        ax.plot(f_rabis, y["doppler"], linewidth=2, label="doppler")
        ax.axhline(1e-3, linestyle=":")
        ax.set_ylabel("Infidelity", fontsize=14)
        ax.set_xlabel("$\\Omega/2\\pi$ [MHz]", fontsize=14)
        ax.tick_params(labelsize=12)
        ax.set_yscale("log")
        ax.set_title("$1 \\gamma$ gate", fontsize=16)
        ax.set_ylim([1e-9, 1e-3])
        ax.legend(fontsize=12)
        fig.savefig(os.path.join(result_dir, "Error_vs_rabi.pdf"), bbox_inches="tight")
        plt.close(fig)

        bar_keys = [k for k in y.keys() if k != "total"]
        bar_vals = [float(y[k][min_idx]) for k in bar_keys]
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(bar_keys, bar_vals)
        ax.set_ylabel("Infidelity contribution")
        ax.set_title(f"Contributions at minimum total infidelity\n(total = {min_total:.3e})")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        fig.savefig(os.path.join(result_dir, "contributions_at_min.pdf"), bbox_inches="tight")
        plt.close(fig)

    config_out = dict(
        atom_name=cfg["atom_name"],
        n=int(p0["n"]),
        l=cfg["l"],
        j=float(cfg["j"]),
        mj=float(cfg["mj"]),
        atom_d_um=float(p0["atom_d"]),
        Bz_G=float(p0["Bz"]),
        pulse_time=float(cfg["pulse_time"]),
        resolution=int(cfg["resolution"]),
        w0_rydberg_um=float(p0["w0_rydberg"]),
        lambda_rydberg_um=float(cfg["lambda_rydberg"]),
        HF_split_MHz=float(cfg["HF_split"]) if cfg["HF_split"] is not None else None,
        alpha_dc_MHz_per_Vcm2=None if cfg["alpha_dc"] is None else float(cfg["alpha_dc"]),
        T_atom_uK=float(cfg["T_atom"]),
        trap_depth_uK=float(p0["trap_depth"]),
        lambda_trap_um=float(cfg["lambda_trap"]),
        w0_trap_um=float(cfg["w0_trap"]),
        edc_fluc_V_per_cm=float(cfg["edc_fluc"]),
        edc_zero_V_per_m=float(cfg["edc_zero"]),
        bdc_fluc_G=float(cfg["bdc_fluc"]),
        num_samples=int(num_samples),
        phase_noise_model=cfg["f_hz_hz2"],
        RIN_strength=cfg["rin_strength"],
        f_Rabi_scan_MHz=dict(start=float(f_rabis[0]), stop=float(f_rabis[-1]), num=int(len(f_rabis))),
        derived=dict(
            blockade_mrad=float(records[min_idx]["blockade_mrad"]),
            blockade_MHz=float(records[min_idx]["blockade_MHz"]),
            R_lifetime_us=float(records[min_idx]["R_lifetime_us"]),
        ),
        minimum=dict(
            index=min_idx,
            Omega_Rabi_MHz=float(f_rabis[min_idx]),
            total=float(min_total),
        ),
    )

    raw_fig2 = dict(
        x=dict(name="Omega_over_2pi_MHz", values=_to_jsonable(f_rabis)),
        y={k: _to_jsonable(v) for k, v in y.items()},
    )

    out = dict(
        meta=dict(script=os.path.basename(__file__) if "__file__" in globals() else "budget_1photon_optimize.py"),
        config={k: _to_jsonable(v) for k, v in config_out.items()},
        raw_fig2=raw_fig2,
        records=[{k: _to_jsonable(v) for k, v in r.items()} for r in records],
    )

    out_json = os.path.join(result_dir, f"scan_1photon_n{int(p0['n'])}_optimized_config_and_raw.json")
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)

    if verbose:
        print(f"min infid: {min_total:.6e} at Omega/2pi = {f_rabis[min_idx]:.6g} MHz")
        print(f"Saved config + raw scan data to: {out_json}")

    return out, out_json


# -----------------------------
# Optimization driver
# -----------------------------
def run_optimization(
    x0=DEFAULT_X0,
    bounds=DEFAULT_BOUNDS,
    config=None,
    opt_samples=500,
    seed=1234,
    maxiter=50,
    result_dir="result",
    optimize_phase=True,
):
    os.makedirs(result_dir, exist_ok=True)
    history = []

    def objective(x):
        # Powell respects bounds, but this makes accidental out-of-bound calls safe.
        x = np.asarray(x, dtype=float)
        for i, (lo, hi) in enumerate(bounds):
            if x[i] < lo or x[i] > hi:
                return 1e9 + np.sum((np.maximum(lo - x, 0) + np.maximum(x - hi, 0)) ** 2)

        val = evaluate_single_point(
            x,
            config=config,
            num_samples=opt_samples,
            seed=seed,
            optimize_phase=optimize_phase,
            verbose=True,
            return_details=False,
        )
        history.append(dict(x=_to_jsonable(np.asarray(x)), total=float(val)))
        return float(val)

    result = opt.minimize(
        objective,
        np.asarray(x0, dtype=float),
        method="Powell",
        bounds=bounds,
        options=dict(maxiter=int(maxiter), disp=True, xtol=1e-3, ftol=1e-6),
    )

    best_x = np.asarray(result.x, dtype=float)
    best_params = vector_to_params(best_x)

    out = dict(
        best_x=_to_jsonable(best_x),
        best_params={k: _to_jsonable(v) for k, v in best_params.items()},
        best_infidelity=float(result.fun),
        success=bool(result.success),
        message=str(result.message),
        bounds=[list(b) for b in bounds],
        opt_samples=int(opt_samples),
        maxiter=int(maxiter),
        history=history,
    )

    out_json = os.path.join(result_dir, "optimization_result.json")
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)

    print("\n===== OPTIMIZATION RESULT =====")
    print("success:", result.success)
    print("message:", result.message)
    print("best_x:", best_x)
    print("best parameters:")
    for name in PARAMETER_NAMES:
        print(f"  {name}: {best_params.get(name.replace('_um', '').replace('_G', '').replace('_uK', ''), None)}")
    print("  n:", best_params["n"])
    print("  atom_d_um:", best_params["atom_d"])
    print("  Omega_Rabi_MHz:", best_params["Omega_Rabi_MHz"])
    # print("  Bz_G:", best_params["Bz"])
    print("  w0_rydberg_um:", best_params["w0_rydberg"])
    print("  trap_depth_uK:", best_params["trap_depth"])
    print("best infidelity:", result.fun)
    print(f"Saved optimization result to: {out_json}")

    return result, best_x, out_json


def parse_args():
    parser = argparse.ArgumentParser(description="Bounded optimization for 1-photon gate infidelity.")
    parser.add_argument("--mode", choices=["optimize", "scan", "optimize_and_scan"], default="optimize")
    parser.add_argument("--result-dir", default="result")
    parser.add_argument("--seed", type=int, default=1234)

    # Initial/current parameter values.
    parser.add_argument("--n", type=float, default=DEFAULT_X0[0])
    parser.add_argument("--atom-d", type=float, default=DEFAULT_X0[1])
    parser.add_argument("--omega-rabi-mhz", type=float, default=DEFAULT_X0[2])
    # parser.add_argument("--Bz", type=float, default=DEFAULT_X0[3])
    parser.add_argument("--w0-rydberg", type=float, default=DEFAULT_X0[3])
    parser.add_argument("--trap-depth", type=float, default=DEFAULT_X0[4])

    # Bounds.  These default to DEFAULT_BOUNDS.
    parser.add_argument("--n-bounds", type=float, nargs=2, default=DEFAULT_BOUNDS[0])
    parser.add_argument("--atom-d-bounds", type=float, nargs=2, default=DEFAULT_BOUNDS[1])
    parser.add_argument("--omega-rabi-bounds", type=float, nargs=2, default=DEFAULT_BOUNDS[2])
    # parser.add_argument("--Bz-bounds", type=float, nargs=2, default=DEFAULT_BOUNDS[3])
    parser.add_argument("--w0-rydberg-bounds", type=float, nargs=2, default=DEFAULT_BOUNDS[3])
    parser.add_argument("--trap-depth-bounds", type=float, nargs=2, default=DEFAULT_BOUNDS[4])

    parser.add_argument("--opt-samples", type=int, default=500, help="Monte Carlo samples during optimization.")
    parser.add_argument("--num-samples", type=int, default=10000, help="Monte Carlo samples for final scan.")
    parser.add_argument("--maxiter", type=int, default=50)
    parser.add_argument("--no-phase-opt", action="store_true", help="Use fixed PhaseGuess instead of nested phase optimization.")
    parser.add_argument("--scan-points", type=int, default=20)
    parser.add_argument("--scan-min-mhz", type=float, default=2.0)
    parser.add_argument("--scan-max-mhz", type=float, default=40.0)
    return parser.parse_args()


def main():
    args = parse_args()
    x0 = params_to_vector(args.n, args.atom_d, args.omega_rabi_mhz, args.w0_rydberg, args.trap_depth)
    bounds = [
        tuple(args.n_bounds),
        tuple(args.atom_d_bounds),
        tuple(args.omega_rabi_bounds),
        # tuple(args.Bz_bounds),
        tuple(args.w0_rydberg_bounds),
        tuple(args.trap_depth_bounds),
    ]
    f_rabis = np.linspace(args.scan_min_mhz, args.scan_max_mhz, args.scan_points)
    optimize_phase = not args.no_phase_opt

    if args.mode == "optimize":
        run_optimization(
            x0=x0,
            bounds=bounds,
            opt_samples=args.opt_samples,
            seed=args.seed,
            maxiter=args.maxiter,
            result_dir=args.result_dir,
            optimize_phase=optimize_phase,
        )
    elif args.mode == "scan":
        run_scan(
            x0,
            num_samples=args.num_samples,
            seed=args.seed,
            result_dir=args.result_dir,
            f_rabis=f_rabis,
            optimize_phase=optimize_phase,
        )
    elif args.mode == "optimize_and_scan":
        result, best_x, opt_json = run_optimization(
            x0=x0,
            bounds=bounds,
            opt_samples=args.opt_samples,
            seed=args.seed,
            maxiter=args.maxiter,
            result_dir=args.result_dir,
            optimize_phase=optimize_phase,
        )
        # Final scan uses the optimized non-Omega parameters.  Omega itself is scanned.
        run_scan(
            best_x,
            num_samples=args.num_samples,
            seed=args.seed,
            result_dir=args.result_dir,
            f_rabis=f_rabis,
            optimize_phase=optimize_phase,
        )


if __name__ == "__main__":
    main()
