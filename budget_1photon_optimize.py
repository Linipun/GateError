"""
Bounded optimizer for the 1-photon Rydberg gate infidelity budget.

Same physics as budget_1photon_scan.py, channel for channel:

    total = TO + dV(motion) + dOmega(motion) + leakage + detuning(E,B,Doppler)
            + laser frequency noise + RIN + decay + fluctuating-E-field (optional)

("scattering" is not modelled in either script and is reported as NaN, excluded
from the total.)

Modes
-----
optimize          bounded minimisation of the total infidelity over
                  (n, atom_d, Omega_Rabi, w0_rydberg, trap_depth)
scan              reproduce the budget_1photon_scan.py Omega scan at a fixed
                  parameter point (JSON + Error_vs_rabi.pdf + contribution bars)
optimize_and_scan optimize, then scan Omega at the optimum

Examples
--------
    python budget_1photon_optimize.py --mode optimize --opt-samples 500 --maxiter 30
    python budget_1photon_optimize.py --mode scan --n 70 --atom-d 5 --w0-rydberg 7.5 \
        --trap-depth 1000 --num-samples 10000
    python budget_1photon_optimize.py --mode optimize_and_scan --opt-samples 500 \
        --num-samples 10000 --maxiter 30
    python budget_1photon_optimize.py --self-test        # batched == reference propagator

Notes
-----
* The Monte-Carlo channels use a fixed seed, so the objective is deterministic
  (common random numbers): the optimizer sees a smooth-ish surface instead of
  MC noise.  Increase --opt-samples if the MC error is comparable to the
  differences you care about.
* n is an integer; it is rounded inside the objective, so the surface is a step
  function along that axis.  --n-grid runs the continuous optimisation once per
  integer n and keeps the best, which is more reliable when n matters.
* atom_d is bounded by the tabulated blockade (1.5-7 um): np.interp clamps
  silently outside it, which would let the optimizer "win" by walking off the
  table.  Points outside are rejected.
* There is no laser-power or trap-power cost in the model, so w0_rydberg and
  trap_depth run to their upper bounds (a bigger waist at fixed Omega and a
  deeper trap are both free here).  Bound them at what your setup can deliver,
  or read them as "as large as you can afford".
* Runtime: ~1.5-2 s per objective evaluation at --opt-samples 500, plus ~35 s
  per new n for the ARC StarkMap polarizability (memoised in
  .alpha_dc_cache.json; pass --alpha-dc to skip it entirely).
"""

from budget_monte_carlo import *  # noqa: F401,F403  (same import surface as the scan script)
from budget_monte_carlo import blockade_dict
import phase_noise  # noqa: F401  (kept for parity with the scan script)
from linear_response import build_Oseq, response_G13, isometry_haar_full
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from arc import *  # noqa: F401,F403
import json
import scipy.linalg
import scipy.optimize as opt
import pandas as pd
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
# Default fixed config (mirrors the "#### config ####" block of the scan script)
# -----------------------------
DEFAULT_CONFIG = dict(
    atom_name="Cs",
    l=1,
    j=3 / 2,
    mj=3 / 2,
    pulse_time=7.65,          # in units of 1/Omega (the Hamiltonians are Rabi-normalised)
    resolution=200,           # number of phase steps in the pulse
    lambda_rydberg=0.319,     # um
    # Effective detuning of the nP3/2 mj=1/2 leakage state from the mj=3/2 gate state, rad/us.
    # None -> bare Zeeman splitting at Bz (undressed limit).  See the scan script for the
    # microwave-dressing caveat: this must be the *effective* Delta_eff, not Omega_mw/2.
    HF_split=2000 * np.pi * 2,  # rad/us
    alpha_dc=None,            # MHz (V/cm)^-2; None -> ARC StarkMap at this n
    T_atom=1.0,               # uK
    lambda_trap=1.064,        # um
    w0_trap=1.064,            # um
    Bz=10.0,                  # G
    edc_fluc=1e-3,            # V/cm, quasi-static E-field uncertainty
    edc_zero=0.0,             # V/cm bias (the scan comment says V/m, the formula uses V/cm)
    bdc_fluc=1e-3,            # G
    rin_strength=1e-4,        # integrated relative intensity noise
    f_hz_hz2=220.0,           # white frequency-noise PSD [Hz^2/Hz]
    f_range=1e5,              # Hz, bandwidth of the frequency noise
    efield_fluc_on=False,     # broadband (no DC bias) E-field channel
    edc_fluc_fast=12.5e-3,    # V/cm RMS of the broadband E-field noise
    efield_range=2e6,         # Hz bandwidth of that noise
)

# Parameters that are optimized (the "### parameters ####" block of the scan script).
PARAMETER_NAMES = [
    "n",
    "atom_d_um",
    "Omega_Rabi_MHz",
    "w0_rydberg_um",
    "trap_depth_uK",
]

#                      n     d[um]  Om/2pi  w0[um]  U[uK]
DEFAULT_X0 = np.array([70.0, 5.0, 10.0, 7.5, 1000.0], dtype=float)

DEFAULT_BOUNDS = [
    (40.0, 99.0),    # n, rounded to an integer during evaluation (see blockade table)
    (1.5, 7.0),      # atom_d_um -- the blockade table only covers 1.5-7 um, np.interp
                     # would silently clamp outside it
    (2.0, 40.0),     # Omega_Rabi_MHz
    (3.0, 80.0),     # w0_rydberg_um
    (20.0, 5000.0),  # trap_depth_uK
]

# Channels that add up to `total`.  "detuning" is deliberately absent: it is the sum of
# efield + bfield + doppler and would double count.  "scattering" is not modelled.
TOTAL_CHANNELS = [
    "TO",
    "blockade",
    "rabi",
    "leakage",
    "efield",
    "bfield",
    "doppler",
    "vnoise",
    "RIN",
    "decay",
    "efield_fluc",
]

_BIG = 1e9  # objective value for invalid parameter points


# -----------------------------
# Helpers
# -----------------------------
def _to_jsonable(x):
    """Convert numpy/pandas types to JSON-serializable python types."""
    try:
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return None if not np.isfinite(x) else float(x)
        if isinstance(x, (np.ndarray,)):
            return [_to_jsonable(v) for v in x]
    except Exception:
        pass
    if isinstance(x, float) and not np.isfinite(x):
        return None
    try:
        if isinstance(x, (pd.Timestamp,)):
            return x.isoformat()
    except Exception:
        pass
    return x


def vector_to_params(x):
    """Map the optimizer vector to named physical parameters."""
    x = np.asarray(x, dtype=float)
    return dict(
        n=int(round(x[0])),
        atom_d=float(x[1]),
        Omega_Rabi_MHz=float(x[2]),
        Omega_Rabi=float(x[2]) * 2 * np.pi,  # rad/us, the unit the Hamiltonians expect
        w0_rydberg=float(x[3]),
        trap_depth=float(x[4]),
    )


def params_to_vector(n, atom_d, omega_rabi_mhz, w0_rydberg, trap_depth):
    return np.array([n, atom_d, omega_rabi_mhz, w0_rydberg, trap_depth], dtype=float)


@lru_cache(maxsize=4)
def get_atom(atom_name):
    if atom_name == "Rb":
        return Rubidium()
    if atom_name == "Cs":
        return Caesium()
    raise ValueError(f"Unsupported atom_name={atom_name!r}; use 'Rb' or 'Cs'.")


@lru_cache(maxsize=256)
def blockade_table_range(atom_name, n):
    """(r_min, r_max) in um covered by the tabulated blockade for this n, or None."""
    try:
        rows = blockade_dict[atom_name][str(int(n))]
    except KeyError:
        return None
    rs = [row[0] for row in rows]
    return float(min(rs)), float(max(rs))


def available_n(atom_name):
    return sorted(int(k) for k in blockade_dict[atom_name].keys())


_ALPHA_CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".alpha_dc_cache.json")


def _load_alpha_cache():
    try:
        with open(_ALPHA_CACHE_FILE) as fh:
            return json.load(fh)
    except Exception:
        return {}


def _save_alpha_cache(cache):
    try:
        with open(_ALPHA_CACHE_FILE, "w") as fh:
            json.dump(cache, fh, indent=1)
    except Exception:
        pass


@lru_cache(maxsize=4096)
def atom_quantities(atom_name, n, l, j, Bz, alpha_dc_input):
    """Rydberg lifetime [us], atom mass [kg] and DC polarizability [MHz/(V/cm)^2].

    The StarkMap diagonalisation costs ~30 s per n, so results are memoised in
    process and on disk (.alpha_dc_cache.json next to this file).
    """
    atom = get_atom(atom_name)
    R_lifetime = atom.getStateLifetime(
        n=int(n), l=int(l), j=float(j), temperature=300, includeLevelsUpTo=int(n) + 20, s=0.5
    ) * 1e6
    m_atom = atom.mass

    if alpha_dc_input is not None:
        return R_lifetime, m_atom, float(alpha_dc_input)

    key = f"{atom_name}|n={int(n)}|Bz={float(Bz):.6g}"
    cache = _load_alpha_cache()
    if key in cache:
        return R_lifetime, m_atom, float(cache[key])

    calc = StarkMap(atom)
    calc.defineBasis(n=int(n), l=1, j=1.5, mj=1.5, nMin=int(n) - 20, nMax=int(n) + 30, maxL=5,
                     Bz=float(Bz) / 10000)
    calc.diagonalise(np.linspace(0, 60, 600))
    alpha_dc = float(calc.getPolarizability(debugOutput=False))

    cache[key] = alpha_dc
    _save_alpha_cache(cache)
    return R_lifetime, m_atom, alpha_dc


# -----------------------------
# Batched propagator
# -----------------------------
# Vectorised equivalent of Hamiltonians.asym_return_fidel / return_fidel: the Monte-Carlo
# samples differ only through scalars (blockade, detuning, Rabi scaling), so all samples are
# propagated in one stacked expm per time step instead of one Python call per sample.
# `--self-test` checks it against the reference class to ~1e-13.
def batch_infidelity(phases, dt, Omega_Rabi, blockade, delta, w1, w2, r_lifetime, r_lifetime2):
    """Bell-state infidelity for a batch of samples.

    blockade, delta [rad/us] and w1, w2 [dimensionless Rabi scalings] broadcast to (S,).
    """
    blockade, delta, w1, w2 = np.broadcast_arrays(
        np.asarray(blockade, dtype=float),
        np.asarray(delta, dtype=float),
        np.asarray(w1, dtype=float),
        np.asarray(w2, dtype=float),
    )
    S = blockade.size
    B = (blockade / Omega_Rabi).ravel()
    D = (delta / Omega_Rabi).ravel()
    w1 = w1.ravel()
    w2 = w2.ravel()

    g1 = (1.0 / r_lifetime) / Omega_Rabi
    g2 = (1.0 / r_lifetime2) / Omega_Rabi

    psi01 = np.zeros((S, 2), dtype=complex)
    psi10 = np.zeros((S, 2), dtype=complex)
    psi11 = np.zeros((S, 4), dtype=complex)
    psi01[:, 0] = 1.0
    psi10[:, 0] = 1.0
    psi11[:, 0] = 1.0

    H1 = np.zeros((S, 2, 2), dtype=complex)
    H2 = np.zeros((S, 2, 2), dtype=complex)
    H = np.zeros((S, 4, 4), dtype=complex)

    # H01 carries only atom 1's decay rate in the reference implementation (both single-atom
    # propagators use self.H01), so g1 is used for psi10 as well.
    H1[:, 1, 1] = D - 1j * g1 / 2
    H2[:, 1, 1] = D - 1j * g1 / 2
    H[:, 1, 1] = D - 1j * g1 / 2
    H[:, 2, 2] = D - 1j * g2 / 2
    H[:, 3, 3] = 2 * D + B - 1j * (g1 + g2) / 2

    for phi in phases:
        ph = np.exp(1j * phi)
        O1 = w1 * ph / 2
        O2 = w2 * ph / 2
        O1c = np.conj(O1)
        O2c = np.conj(O2)

        H1[:, 0, 1] = O1
        H1[:, 1, 0] = O1c
        H2[:, 0, 1] = O2
        H2[:, 1, 0] = O2c

        H[:, 0, 1] = O1
        H[:, 0, 2] = O2
        H[:, 1, 0] = O1c
        H[:, 1, 3] = O2
        H[:, 2, 0] = O2c
        H[:, 2, 3] = O1
        H[:, 3, 1] = O2c
        H[:, 3, 2] = O1c

        U1 = scipy.linalg.expm(-1j * H1 * dt)
        U2 = scipy.linalg.expm(-1j * H2 * dt)
        U11 = scipy.linalg.expm(-1j * H * dt)

        psi01 = np.einsum("sij,sj->si", U1, psi01)
        psi10 = np.einsum("sij,sj->si", U2, psi10)
        psi11 = np.einsum("sij,sj->si", U11, psi11)

    gp1 = psi01[:, 0] / np.abs(psi01[:, 0])
    gp2 = psi10[:, 0] / np.abs(psi10[:, 0])
    a01 = psi01[:, 0] / gp1
    a10 = psi10[:, 0] / gp2
    a11 = psi11[:, 0] / (gp1 * gp2)

    fid = np.abs(1 + a01 + a10 - a11) ** 2 / 16
    return 1.0 - fid


def self_test(verbose=True):
    """Check the batched propagator against Hamiltonians.asym_return_fidel/return_fidel."""
    rng = np.random.default_rng(0)
    Omega = 2 * np.pi * 12.3
    pulse_time, resolution = 7.65, 60
    _, phases, dt = phase_cosine_generate(0.7, 1.05, -0.73, 0.02, pulse_time, resolution)

    blockades = rng.uniform(50, 900, 7)
    deltas = rng.normal(0, 0.05, 7)
    w1s = rng.uniform(0.9, 1.0, 7)
    w2s = rng.uniform(0.9, 1.0, 7)
    R1, R2 = 214.0, 214.0

    got = batch_infidelity(phases, dt, Omega, blockades, deltas, w1s, w2s, R1, R2)
    ref = []
    for b, d, a, c_ in zip(blockades, deltas, w1s, w2s):
        H = Hamiltonians(Omega_Rabi1=Omega, blockade_inf=False, blockade=b, r_lifetime=R1,
                         Delta1=d, Stark1=0, Stark2=0, resolution=resolution, r_lifetime2=R2,
                         pulse_time=pulse_time)
        fid, _ = H.asym_return_fidel(phases=phases, dt=dt, omega1_scale=a, omega2_scale=c_)
        ref.append(1 - fid)
    err_asym = np.max(np.abs(got - np.asarray(ref)))

    # symmetric case must also reproduce return_fidel (used for the detuning channel)
    got_sym = batch_infidelity(phases, dt, Omega, blockades, deltas, 1.0, 1.0, R1, R2)
    ref_sym = []
    for b, d in zip(blockades, deltas):
        H = Hamiltonians(Omega_Rabi1=Omega, blockade_inf=False, blockade=b, r_lifetime=R1,
                         Delta1=d, Stark1=0, Stark2=0, resolution=resolution, r_lifetime2=R2,
                         pulse_time=pulse_time)
        fid, _ = H.return_fidel(phases=phases, dt=dt)
        ref_sym.append(1 - fid)
    err_sym = np.max(np.abs(got_sym - np.asarray(ref_sym)))

    if verbose:
        print(f"self-test: max |batched - asym_return_fidel| = {err_asym:.3e}")
        print(f"self-test: max |batched - return_fidel|      = {err_sym:.3e}")
    assert err_asym < 1e-11 and err_sym < 1e-11, "batched propagator disagrees with reference"
    return max(err_asym, err_sym)


# -----------------------------
# Core simulation: one parameter point
# -----------------------------
def evaluate_single_point(
    x,
    config=None,
    num_samples=500,
    seed=1234,
    optimize_phase=True,
    verbose=False,
    return_details=False,
    batched=True,
):
    """Total infidelity at one parameter vector (the objective function).

    Reproduces every channel of budget_1photon_scan.py for a single Omega_Rabi.
    """
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)

    p = vector_to_params(x)
    atom_name = cfg["atom_name"]
    n = p["n"]
    atom_d = p["atom_d"]
    Omega_Rabi = p["Omega_Rabi"]
    w0_rydberg = p["w0_rydberg"]
    trap_depth = p["trap_depth"]

    def _reject(msg):
        if verbose:
            print(f"  rejected (n={n}, d={atom_d:.4g}): {msg}")
        return (_BIG, dict(total=_BIG, invalid=msg)) if return_details else _BIG

    if n < 1 or atom_d <= 0 or Omega_Rabi <= 0 or w0_rydberg <= 0 or trap_depth <= 0:
        return _reject("non-physical parameter")

    r_range = blockade_table_range(atom_name, n)
    if r_range is None:
        return _reject(f"no tabulated blockade for {atom_name} n={n} "
                       f"(have {available_n(atom_name)[0]}-{available_n(atom_name)[-1]})")
    if not (r_range[0] <= atom_d <= r_range[1]):
        # np.interp clamps silently outside the table, which would hand the optimizer a
        # free lunch (blockade frozen at the edge value).  Reject instead.
        return _reject(f"atom_d outside blockade table {r_range[0]}-{r_range[1]} um")

    atom = get_atom(atom_name)
    l = cfg["l"]
    j = cfg["j"]
    resolution = cfg["resolution"]
    pulse_time = cfg["pulse_time"]
    lambda_rydberg = cfg["lambda_rydberg"]
    lambda_trap = cfg["lambda_trap"]
    w0_trap = cfg["w0_trap"]
    T_atom = cfg["T_atom"]
    Bz = cfg["Bz"]
    HF_split = cfg["HF_split"]

    blockade_mrad = find_blockade_Mrad(atom_name, n, atom_d)
    R_lifetime, m_atom, alpha_dc = atom_quantities(atom_name, n, l, j, float(Bz), cfg["alpha_dc"])

    if HF_split is None:
        HF_split = (
            atom.getZeemanEnergyShift(l=1, j=3 / 2, mj=3 / 2, magneticFieldBz=Bz / 10000)
            - atom.getZeemanEnergyShift(l=1, j=3 / 2, mj=1 / 2, magneticFieldBz=Bz / 10000)
        ) / hbar / 1e6

    # ---- atom positions in the trap (same geometry as the scan script) ----
    rng_pos = np.random.default_rng([seed, 0])
    rng_det = np.random.default_rng([seed, 1])
    sigma_r = sigma_r_um(T_atom, trap_depth, w0_trap)
    sigma_z = sigma_z_um(T_atom, trap_depth, w0_trap, lambda_trap)
    ds, c1, c2 = sample_pair_distances(
        n_samples=int(num_samples), sigma_r=sigma_r, sigma_z=sigma_z, x_offset=atom_d, rng=rng_pos
    )
    x1, y1, z1 = c1["x"], c1["y"], c1["z"]
    x2, y2, z2 = c2["x"], c2["y"], c2["z"]

    rabi_center = np.sqrt(relative_gaussian_beam_intensity(0, 0, 0 - atom_d / 2, w0_rydberg, lambda_rydberg))
    rabis1 = np.sqrt(relative_gaussian_beam_intensity(x1, z1, y1 - atom_d / 2, w0_rydberg, lambda_rydberg)) / rabi_center
    rabis2 = np.sqrt(relative_gaussian_beam_intensity(x2 - atom_d, z2, y2 + atom_d / 2, w0_rydberg, lambda_rydberg)) / rabi_center

    # ---- pulse-phase optimisation (with decay, as in the scan script) ----
    H_gen1 = Hamiltonians(Omega_Rabi1=Omega_Rabi, blockade_inf=False, blockade=blockade_mrad,
                          r_lifetime=R_lifetime, Delta1=0, Stark1=0, Stark2=0,
                          resolution=resolution, r_lifetime2=R_lifetime, pulse_time=pulse_time)
    PhaseGuess = [2 * np.pi * 0.1122, 1.0431, -0.7318, 0]
    if optimize_phase:
        # default (BFGS) method, exactly as in budget_1photon_scan.py -- it is both faster
        # and more reliable here than Nelder-Mead
        opt_out = opt.minimize(fun=fid_optimize, x0=PhaseGuess, args=(H_gen1,))
        phase_params = opt_out.x
    else:
        phase_params = np.asarray(PhaseGuess, dtype=float)

    # ---- intrinsic (finite-blockade) error, decay switched off ----
    H_gen1 = Hamiltonians(Omega_Rabi1=Omega_Rabi, blockade_inf=False, blockade=blockade_mrad,
                          r_lifetime=1e10, Delta1=0, Stark1=0, Stark2=0, resolution=resolution,
                          r_lifetime2=1e10, pulse_time=pulse_time)
    time_, phase, dt = phase_cosine_generate(*phase_params, H_gen1.pulse_time, H_gen1.resolution)
    fid1, _ = H_gen1.return_fidel(phases=phase, dt=dt)
    infid_TO1 = 1 - fid1

    # ---- linear-response channels ----
    S_haar = isometry_haar_full()
    # B must be the *normalised* blockade: linear_response.H0 is written with Rabi = 1.
    o_f = build_Oseq(phases=phase, dt=dt, B=blockade_mrad / Omega_Rabi, is_intensity=False)
    o_I = build_Oseq(phases=phase, dt=dt, B=blockade_mrad / Omega_Rabi, is_intensity=True)

    Ii = response_G13(o_I, S_haar, 0, dt=dt)
    RIN_contribution = Ii * cfg["rin_strength"]

    If_1 = response_G13(o_f, S_haar, 0, dt=dt) / Omega_Rabi ** 2
    v_contribution = If_1 * cfg["f_hz_hz2"] * cfg["f_range"] / 1e6 / 1e6

    # Broadband, zero-bias E-field noise -> quadratic Stark detuning noise up to 2*bandwidth.
    if cfg["efield_fluc_on"]:
        dnu_rms = 0.5 * alpha_dc * cfg["edc_fluc_fast"] ** 2 * 1e6   # Hz RMS
        f_hi = 2.0 * cfg["efield_range"]
        fs_E = np.linspace(f_hi / 40, f_hi, 40)
        S_delta = dnu_rms ** 2 / f_hi                                # flat PSD [Hz^2/Hz]
        dfE = fs_E[1] - fs_E[0]
        efield_fluc_contribution = 0.0
        for fE in fs_E:
            omg = 2 * np.pi * fE / 1e6 / Omega_Rabi
            I_w = response_G13(o_f, S_haar, omg, dt=dt) / Omega_Rabi ** 2
            efield_fluc_contribution += I_w * S_delta * dfE / 1e6 / 1e6
        efield_fluc_contribution = float(efield_fluc_contribution)
    else:
        efield_fluc_contribution = 0.0

    # ---- Rydberg decay ----
    infids_decay = (2.95 / Omega_Rabi) / R_lifetime

    # ---- quasi-static detuning: Doppler + DC E + DC B (all rad/s) ----
    doppler_shift = 2 * np.pi / (lambda_rydberg * 1e-6) * np.sqrt(kb * (T_atom * 1e-6) / m_atom)
    delta_edc = abs(-0.5 * alpha_dc * 1e6 * ((cfg["edc_zero"] + cfg["edc_fluc"]) ** 2
                                             - cfg["edc_zero"] ** 2)) * 2 * np.pi
    delta_bdc = atom.getZeemanEnergyShift(l=1, j=3 / 2, mj=3 / 2,
                                          magneticFieldBz=cfg["bdc_fluc"] / 10000) / hbar
    total_shift = np.sqrt(delta_bdc ** 2 + delta_edc ** 2 + doppler_shift ** 2)
    detunings = total_shift / 1e6  # rad/us; Hamiltonians divides by Omega_Rabi internally

    d_samples = rng_det.normal(loc=0.0, scale=detunings, size=int(num_samples))
    if batched:
        infids_s = batch_infidelity(phase, dt, Omega_Rabi, blockade_mrad, d_samples, 1.0, 1.0,
                                    10e9, 10e9)
    else:
        infids_s = np.array([
            1 - Hamiltonians(Omega_Rabi1=Omega_Rabi, blockade_inf=False, blockade=blockade_mrad,
                             r_lifetime=10e9, Delta1=d, Stark1=0, Stark2=0, resolution=resolution,
                             r_lifetime2=10e9, pulse_time=pulse_time).return_fidel(phases=phase, dt=dt)[0]
            for d in d_samples
        ])
    infids_detuning = float(np.mean(infids_s) - infid_TO1)
    infids_detuning_std = float(infids_s.std(ddof=1) / np.sqrt(len(infids_s)))

    if total_shift > 0:
        infids_bdc = delta_bdc ** 2 / total_shift ** 2 * infids_detuning
        infids_edc = delta_edc ** 2 / total_shift ** 2 * infids_detuning
        infids_doppler = doppler_shift ** 2 / total_shift ** 2 * infids_detuning
    else:
        infids_bdc = infids_edc = infids_doppler = 0.0

    # ---- motion: blockade spread (dV) and Rabi inhomogeneity (dOmega) ----
    blockades = find_blockade_Mrad(atom_name, n, ds)  # np.interp is vectorised
    if batched:
        infids_blockade = batch_infidelity(phase, dt, Omega_Rabi, blockades, 0.0, 1.0, 1.0,
                                           10e9, 10e9)
        infids_rabi = batch_infidelity(phase, dt, Omega_Rabi, blockade_mrad, 0.0, rabis1, rabis2,
                                       10e9, 10e9)
    else:
        infids_blockade, infids_rabi = [], []
        for bl, r1, r2 in zip(blockades, rabis1, rabis2):
            Hb = Hamiltonians(Omega_Rabi1=Omega_Rabi, blockade_inf=False, blockade=bl,
                              r_lifetime=10e9, Delta1=0, Stark1=0, Stark2=0, resolution=resolution,
                              r_lifetime2=10e9, pulse_time=pulse_time)
            infids_blockade.append(1 - Hb.asym_return_fidel(phases=phase, dt=dt,
                                                            omega1_scale=1, omega2_scale=1)[0])
            Hr = Hamiltonians(Omega_Rabi1=Omega_Rabi, blockade_inf=False, blockade=blockade_mrad,
                              r_lifetime=10e9, Delta1=0, Stark1=0, Stark2=0, resolution=resolution,
                              r_lifetime2=10e9, pulse_time=pulse_time)
            infids_rabi.append(1 - Hr.asym_return_fidel(phases=phase, dt=dt,
                                                        omega1_scale=r1, omega2_scale=r2)[0])
        infids_blockade = np.asarray(infids_blockade)
        infids_rabi = np.asarray(infids_rabi)

    infids_motion_blockade = float(np.mean(infids_blockade) - infid_TO1)
    infids_motion_rabi = float(np.mean(infids_rabi) - infid_TO1)
    infids_motion_blockade_std = float(infids_blockade.std(ddof=1) / np.sqrt(len(infids_blockade)))
    infids_motion_rabi_std = float(infids_rabi.std(ddof=1) / np.sqrt(len(infids_rabi)))

    # ---- leakage into the nP3/2 mj=1/2 branch ----
    H_leak = LeakageHamiltonians(Omega_Rabi1=Omega_Rabi, blockade_inf=False, blockade=blockade_mrad,
                                 r_lifetime=10e9, Delta1=0, Stark1=0, Stark2=0,
                                 resolution=resolution, r_lifetime2=10e9, pulse_time=pulse_time,
                                 mj12_split=HF_split)
    fid_with_leak, _ = H_leak.return_fidel(phases=phase, dt=dt)
    leakage_mj = float((1 - fid_with_leak) - infid_TO1)

    total = (
        infid_TO1
        + infids_motion_blockade
        + infids_motion_rabi
        + leakage_mj
        + infids_detuning
        + v_contribution
        + RIN_contribution
        + infids_decay
        + efield_fluc_contribution
    )

    details = dict(
        n=int(n),
        atom_d_um=float(atom_d),
        Omega_Rabi_MHz=float(p["Omega_Rabi_MHz"]),
        w0_rydberg_um=float(w0_rydberg),
        trap_depth_uK=float(trap_depth),
        TO=float(infid_TO1),
        blockade=float(infids_motion_blockade),
        rabi=float(infids_motion_rabi),
        leakage=float(leakage_mj),
        detuning=float(infids_detuning),
        efield=float(infids_edc),
        bfield=float(infids_bdc),
        doppler=float(infids_doppler),
        efield_fluc=float(efield_fluc_contribution),
        vnoise=float(v_contribution),
        RIN=float(RIN_contribution),
        decay=float(infids_decay),
        scattering=float("nan"),  # not modelled, excluded from the total
        total=float(total),
        mc_sem=dict(detuning=infids_detuning_std, blockade=infids_motion_blockade_std,
                    rabi=infids_motion_rabi_std),
        phase_params=_to_jsonable(np.asarray(phase_params)),
        blockade_mrad=float(blockade_mrad),
        blockade_MHz=float(blockade_mrad / 2 / np.pi),
        R_lifetime_us=float(R_lifetime),
        alpha_dc_MHz_per_Vcm2=float(alpha_dc),
        HF_split_MHz=float(HF_split) / (2 * np.pi),
        num_samples=int(num_samples),
    )

    if verbose:
        print(f"n={n}, d={atom_d:.4g} um, Omega/2pi={p['Omega_Rabi_MHz']:.4g} MHz, "
              f"w0={w0_rydberg:.4g} um, U={trap_depth:.4g} uK -> total={total:.6e}")

    return (float(total), details) if return_details else float(total)


# -----------------------------
# Omega scan (the budget_1photon_scan.py figure)
# -----------------------------
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
    batched=True,
    tag="optimized",
):
    """Scan Omega at fixed non-Omega parameters; save the same JSON/figure as the scan script."""
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
            xx, config=cfg, num_samples=num_samples, seed=seed, optimize_phase=optimize_phase,
            verbose=verbose, return_details=True, batched=batched,
        )
        details["Omega_Rabi_MHz"] = float(f)
        records.append(details)

    def arr(key):
        return np.asarray([r.get(key, np.nan) for r in records], dtype=float)

    y = {k: arr(k) for k in ["TO", "vnoise", "RIN", "blockade", "efield", "efield_fluc", "bfield",
                             "doppler", "rabi", "decay", "leakage", "detuning", "scattering",
                             "total"]}

    min_idx = int(np.nanargmin(y["total"]))
    min_total = float(y["total"][min_idx])
    if "invalid" in records[min_idx]:
        raise ValueError(f"no valid point in the scan: {records[min_idx]['invalid']}")

    if make_plots:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(f_rabis, y["vnoise"], c="#4e63ff", linewidth=2, label="$v$")
        ax.plot(f_rabis, y["RIN"], c="#ff4da6", linewidth=2, label="RIN")
        ax.plot(f_rabis, y["blockade"], c="#2ecc71", linewidth=2, label=r"$\delta V$")
        ax.plot(f_rabis, y["rabi"], linewidth=2, label=r"$\delta \Omega$")
        ax.plot(f_rabis, y["decay"], c="#7f8c8d", linewidth=2, label=r"$\gamma$")
        ax.plot(f_rabis, y["TO"], c="#b06ae2", linewidth=2, label="TO")
        ax.plot(f_rabis, y["total"], c="k", linewidth=4, label=r"$\Sigma$")
        ax.plot(f_rabis, y["leakage"], linewidth=2, label="leakage")
        ax.plot(f_rabis, y["efield"], linewidth=2, label="E")
        ax.plot(f_rabis, y["efield_fluc"], linewidth=2, linestyle="--", label="E-fluc")
        ax.plot(f_rabis, y["bfield"], linewidth=2, label="B")
        ax.plot(f_rabis, y["doppler"], linewidth=2, label="doppler")
        ax.axhline(1e-3, c="k", linestyle=":")
        ax.set_ylabel("Infidelity", fontsize=14)
        ax.set_xlabel(r"$\Omega/ 2\pi$ [MHz] ", fontsize=14)
        ax.tick_params(labelsize=12)
        ax.set_yscale("log")
        ax.set_title(r"$1 \gamma$ gate", fontsize=16)
        ax.set_ylim([1e-9, 1e-3])
        ax.legend(fontsize=12)
        fig.savefig(os.path.join(result_dir, f"Error_vs_rabi_{tag}.pdf"), bbox_inches="tight")
        plt.close(fig)

        bar_vals = [float(y[k][min_idx]) for k in TOTAL_CHANNELS]
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(TOTAL_CHANNELS, np.abs(bar_vals))
        ax.set_yscale("log")
        ax.set_ylabel("|Infidelity contribution|")
        ax.set_title(f"Contributions at the minimum "
                     f"($\\Omega/2\\pi$ = {f_rabis[min_idx]:.3g} MHz, total = {min_total:.3e})")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        fig.savefig(os.path.join(result_dir, f"contributions_at_min_{tag}.pdf"), bbox_inches="tight")
        plt.close(fig)

    rec0 = records[min_idx]
    config_out = dict(
        atom_name=cfg["atom_name"],
        n=int(p0["n"]), l=cfg["l"], j=float(cfg["j"]), mj=float(cfg["mj"]),
        atom_d_um=float(p0["atom_d"]),
        Bz_G=float(cfg["Bz"]),
        pulse_time=float(cfg["pulse_time"]),
        resolution=int(cfg["resolution"]),
        w0_rydberg_um=float(p0["w0_rydberg"]),
        lambda_rydberg_um=float(cfg["lambda_rydberg"]),
        mj_leak_split_MHz=float(rec0["HF_split_MHz"]),
        alpha_dc_MHz_per_Vcm2=float(rec0["alpha_dc_MHz_per_Vcm2"]),
        T_atom_uK=float(cfg["T_atom"]),
        trap_depth_uK=float(p0["trap_depth"]),
        lambda_trap_um=float(cfg["lambda_trap"]),
        w0_trap_um=float(cfg["w0_trap"]),
        edc_fluc_V_per_cm=float(cfg["edc_fluc"]),
        edc_fluc_fast_V_per_cm=float(cfg["edc_fluc_fast"]),
        efield_range_Hz=float(cfg["efield_range"]),
        efield_fluc_on=bool(cfg["efield_fluc_on"]),
        edc_zero_V_per_m=float(cfg["edc_zero"]),
        bdc_fluc_G=float(cfg["bdc_fluc"]),
        num_samples=int(num_samples),
        phase_noise_Hz2_per_Hz=float(cfg["f_hz_hz2"]),
        f_range_Hz=float(cfg["f_range"]),
        RIN_strength=float(cfg["rin_strength"]),
        seed=int(seed),
        f_Rabi_scan_MHz=dict(start=float(f_rabis[0]), stop=float(f_rabis[-1]), num=int(len(f_rabis))),
        derived=dict(
            blockade_mrad=float(rec0["blockade_mrad"]),
            blockade_MHz=float(rec0["blockade_MHz"]),
            R_lifetime_us=float(rec0["R_lifetime_us"]),
        ),
        minimum=dict(index=min_idx, Omega_Rabi_MHz=float(f_rabis[min_idx]), total=float(min_total)),
    )

    raw_fig2 = dict(
        x=dict(name="Omega_over_2pi_MHz", values=_to_jsonable(f_rabis)),
        y={k: _to_jsonable(v) for k, v in y.items()},
    )

    out = dict(
        meta=dict(script=os.path.basename(__file__) if "__file__" in globals()
                  else "budget_1photon_optimize.py"),
        config={k: _to_jsonable(v) for k, v in config_out.items()},
        raw_fig2=raw_fig2,
        records=[{k: _to_jsonable(v) for k, v in r.items()} for r in records],
    )

    out_json = os.path.join(result_dir, f"scan_1photon_n{int(p0['n'])}_{tag}_config_and_raw.json")
    with open(out_json, "w") as fh:
        json.dump(out, fh, indent=2)

    if verbose:
        print(f"min infid: {min_total:.6e} at Omega/2pi = {f_rabis[min_idx]:.6g} MHz "
              f"(scattering not included)")
        print(f"Saved config + raw scan data to: {out_json}")

    return out, out_json


# -----------------------------
# Optimization driver
# -----------------------------
def _clip_bounds(bounds, cfg):
    """Keep n inside the tabulated range for this atom."""
    bounds = [tuple(float(v) for v in b) for b in bounds]
    ns = available_n(cfg["atom_name"])
    lo, hi = bounds[0]
    bounds[0] = (max(lo, ns[0]), min(hi, ns[-1]))
    return bounds


def run_optimization(
    x0=DEFAULT_X0,
    bounds=DEFAULT_BOUNDS,
    config=None,
    opt_samples=500,
    seed=1234,
    maxiter=50,
    result_dir="result",
    optimize_phase=True,
    batched=True,
    n_grid=None,
    verbose=True,
):
    os.makedirs(result_dir, exist_ok=True)
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)
    bounds = _clip_bounds(bounds, cfg)

    x0 = np.asarray(x0, dtype=float).copy()
    if len(x0) != len(bounds):
        raise ValueError(f"x0 has {len(x0)} entries but {len(bounds)} bounds were given.")
    x0 = np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds])

    history = []

    def make_objective(fixed_n=None):
        def objective(xin):
            x = np.asarray(xin, dtype=float).copy()
            if fixed_n is not None:
                x = np.concatenate(([float(fixed_n)], x))
            # Powell honours bounds, this only guards accidental out-of-range calls.
            lo = np.array([b[0] for b in bounds])
            hi = np.array([b[1] for b in bounds])
            if np.any(x < lo) or np.any(x > hi):
                return _BIG + float(np.sum((np.maximum(lo - x, 0) + np.maximum(x - hi, 0)) ** 2))
            val = evaluate_single_point(
                x, config=cfg, num_samples=opt_samples, seed=seed,
                optimize_phase=optimize_phase, verbose=verbose, return_details=False,
                batched=batched,
            )
            history.append(dict(x=_to_jsonable(np.asarray(x)), total=float(val)))
            return float(val)
        return objective

    if n_grid is None:
        result = opt.minimize(
            make_objective(), x0, method="Powell", bounds=bounds,
            options=dict(maxiter=int(maxiter), disp=verbose, xtol=1e-3, ftol=1e-6),
        )
        best_x = np.asarray(result.x, dtype=float)
        best_fun = float(result.fun)
        success, message = bool(result.success), str(result.message)
    else:
        # Mixed-integer version: n is discrete, so optimise the continuous parameters
        # once per n and keep the best.  Much more reliable than letting Powell walk a
        # step function along the n axis.
        best_x, best_fun, success, message = None, np.inf, False, ""
        for n_try in n_grid:
            if not (bounds[0][0] <= n_try <= bounds[0][1]):
                continue
            sub = opt.minimize(
                make_objective(fixed_n=n_try), x0[1:], method="Powell", bounds=bounds[1:],
                options=dict(maxiter=int(maxiter), disp=False, xtol=1e-3, ftol=1e-6),
            )
            if verbose:
                print(f"  n={int(n_try)}: best total = {sub.fun:.6e}")
            if sub.fun < best_fun:
                best_fun = float(sub.fun)
                best_x = np.concatenate(([float(n_try)], np.asarray(sub.x, dtype=float)))
                success, message = bool(sub.success), str(sub.message)
        if best_x is None:
            raise ValueError("n_grid contains no value inside the n bounds.")
        result = opt.OptimizeResult(x=best_x, fun=best_fun, success=success, message=message)

    best_params = vector_to_params(best_x)
    # Re-evaluate the optimum with the full channel breakdown.
    _, best_details = evaluate_single_point(
        best_x, config=cfg, num_samples=opt_samples, seed=seed, optimize_phase=optimize_phase,
        verbose=False, return_details=True, batched=batched,
    )

    out = dict(
        best_x=_to_jsonable(best_x),
        best_params={k: _to_jsonable(v) for k, v in best_params.items()},
        best_infidelity=float(best_fun),
        best_breakdown={k: _to_jsonable(v) for k, v in best_details.items()},
        success=success,
        message=message,
        parameter_names=PARAMETER_NAMES,
        bounds=[list(b) for b in bounds],
        opt_samples=int(opt_samples),
        maxiter=int(maxiter),
        seed=int(seed),
        config={k: _to_jsonable(v) for k, v in cfg.items()},
        history=history,
    )

    out_json = os.path.join(result_dir, "optimization_result.json")
    with open(out_json, "w") as fh:
        json.dump(out, fh, indent=2)

    print("\n===== OPTIMIZATION RESULT =====")
    print("success:", success)
    print("message:", message)
    print("evaluations:", len(history))
    print("best parameters:")
    for name, val in zip(PARAMETER_NAMES, best_x):
        if name == "n":
            print(f"  {name}: {int(round(val))}")
        else:
            print(f"  {name}: {val:.6g}")
    print("best infidelity:", best_fun)
    print("breakdown at the optimum:")
    for k in TOTAL_CHANNELS:
        print(f"  {k:12s} {best_details[k]:.3e}")
    print(f"Saved optimization result to: {out_json}")

    return result, best_x, out_json


# -----------------------------
# CLI
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Bounded optimization for 1-photon gate infidelity.")
    parser.add_argument("--mode", choices=["optimize", "scan", "optimize_and_scan"], default="optimize")
    parser.add_argument("--result-dir", default="result")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--self-test", action="store_true",
                        help="Check the batched propagator against the reference class and exit.")

    # Parameter point / optimizer start.
    parser.add_argument("--n", type=float, default=DEFAULT_X0[0])
    parser.add_argument("--atom-d", type=float, default=DEFAULT_X0[1])
    parser.add_argument("--omega-rabi-mhz", type=float, default=DEFAULT_X0[2])
    parser.add_argument("--w0-rydberg", type=float, default=DEFAULT_X0[3])
    parser.add_argument("--trap-depth", type=float, default=DEFAULT_X0[4])

    # Bounds.
    parser.add_argument("--n-bounds", type=float, nargs=2, default=DEFAULT_BOUNDS[0])
    parser.add_argument("--atom-d-bounds", type=float, nargs=2, default=DEFAULT_BOUNDS[1])
    parser.add_argument("--omega-rabi-bounds", type=float, nargs=2, default=DEFAULT_BOUNDS[2])
    parser.add_argument("--w0-rydberg-bounds", type=float, nargs=2, default=DEFAULT_BOUNDS[3])
    parser.add_argument("--trap-depth-bounds", type=float, nargs=2, default=DEFAULT_BOUNDS[4])

    parser.add_argument("--opt-samples", type=int, default=500, help="MC samples during optimization.")
    parser.add_argument("--num-samples", type=int, default=10000, help="MC samples for the final scan.")
    parser.add_argument("--maxiter", type=int, default=50)
    parser.add_argument("--n-grid", type=int, nargs="+", default=None,
                        help="Optimize the continuous parameters once per listed n and keep the best.")
    parser.add_argument("--no-phase-opt", action="store_true",
                        help="Use the fixed PhaseGuess instead of re-optimising the pulse phases.")
    parser.add_argument("--no-batch", action="store_true",
                        help="Use the reference (slow) per-sample propagator.")
    parser.add_argument("--scan-points", type=int, default=20)
    parser.add_argument("--scan-min-mhz", type=float, default=2.0)
    parser.add_argument("--scan-max-mhz", type=float, default=40.0)

    # Config overrides.
    parser.add_argument("--atom", choices=["Cs", "Rb"], default=DEFAULT_CONFIG["atom_name"])
    parser.add_argument("--Bz", type=float, default=DEFAULT_CONFIG["Bz"])
    parser.add_argument("--T-atom", type=float, default=DEFAULT_CONFIG["T_atom"])
    parser.add_argument("--pulse-time", type=float, default=DEFAULT_CONFIG["pulse_time"])
    parser.add_argument("--resolution", type=int, default=DEFAULT_CONFIG["resolution"])
    parser.add_argument("--rin-strength", type=float, default=DEFAULT_CONFIG["rin_strength"])
    parser.add_argument("--f-noise", type=float, default=DEFAULT_CONFIG["f_hz_hz2"],
                        help="White frequency-noise PSD [Hz^2/Hz].")
    parser.add_argument("--edc-fluc", type=float, default=DEFAULT_CONFIG["edc_fluc"])
    parser.add_argument("--bdc-fluc", type=float, default=DEFAULT_CONFIG["bdc_fluc"])
    parser.add_argument("--hf-split-mhz", type=float, default=None,
                        help="Effective mj=1/2 leakage splitting [MHz]. Omit to keep the default; "
                             "pass 0 or a negative value to use the bare Zeeman splitting.")
    parser.add_argument("--alpha-dc", type=float, default=None,
                        help="DC polarizability [MHz/(V/cm)^2]. Omit to compute it with ARC.")
    parser.add_argument("--efield-fluc", action="store_true",
                        help="Enable the broadband E-field noise channel.")
    parser.add_argument("--edc-fluc-fast", type=float, default=DEFAULT_CONFIG["edc_fluc_fast"])
    parser.add_argument("--efield-range", type=float, default=DEFAULT_CONFIG["efield_range"])
    return parser.parse_args()


def config_from_args(args):
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(
        atom_name=args.atom,
        Bz=args.Bz,
        T_atom=args.T_atom,
        pulse_time=args.pulse_time,
        resolution=args.resolution,
        rin_strength=args.rin_strength,
        f_hz_hz2=args.f_noise,
        edc_fluc=args.edc_fluc,
        bdc_fluc=args.bdc_fluc,
        efield_fluc_on=bool(args.efield_fluc),
        edc_fluc_fast=args.edc_fluc_fast,
        efield_range=args.efield_range,
    )
    if args.alpha_dc is not None:
        cfg["alpha_dc"] = args.alpha_dc
    if args.hf_split_mhz is not None:
        cfg["HF_split"] = None if args.hf_split_mhz <= 0 else args.hf_split_mhz * 2 * np.pi
    return cfg


def main():
    args = parse_args()
    if args.self_test:
        self_test()
        return

    cfg = config_from_args(args)
    x0 = params_to_vector(args.n, args.atom_d, args.omega_rabi_mhz, args.w0_rydberg, args.trap_depth)
    bounds = [
        tuple(args.n_bounds),
        tuple(args.atom_d_bounds),
        tuple(args.omega_rabi_bounds),
        tuple(args.w0_rydberg_bounds),
        tuple(args.trap_depth_bounds),
    ]
    f_rabis = np.linspace(args.scan_min_mhz, args.scan_max_mhz, args.scan_points)
    optimize_phase = not args.no_phase_opt
    batched = not args.no_batch

    if args.mode in ("optimize", "optimize_and_scan"):
        _, best_x, _ = run_optimization(
            x0=x0, bounds=bounds, config=cfg, opt_samples=args.opt_samples, seed=args.seed,
            maxiter=args.maxiter, result_dir=args.result_dir, optimize_phase=optimize_phase,
            batched=batched, n_grid=args.n_grid,
        )
        if args.mode == "optimize_and_scan":
            # Final scan at the optimized non-Omega parameters; Omega itself is scanned.
            run_scan(best_x, config=cfg, num_samples=args.num_samples, seed=args.seed,
                     result_dir=args.result_dir, f_rabis=f_rabis, optimize_phase=optimize_phase,
                     batched=batched, tag="optimized")
    elif args.mode == "scan":
        run_scan(x0, config=cfg, num_samples=args.num_samples, seed=args.seed,
                 result_dir=args.result_dir, f_rabis=f_rabis, optimize_phase=optimize_phase,
                 batched=batched, tag="fixed")


if __name__ == "__main__":
    main()
