"""
Bounded optimizer for the 2-photon Rydberg gate infidelity budget, with a laser-power
constraint on the two Rydberg beams.

Same channels as budget_2_photon.py:

    total = TO + dV(motion) + dOmega(motion) + scattering(7P) + detuning(E,B,Doppler)
            + laser frequency noise (both arms) + RIN (both arms) + Rydberg decay

Why the power bound matters
---------------------------
The gate runs on Omega_eff = Omega1*Omega2/(2*Delta), so a given (Omega_eff, Delta) fixes the
PRODUCT Omega1*Omega2 = 2*Omega_eff*Delta.  Raising Delta suppresses scattering off the
intermediate 7P state (~1/Delta^2) but costs power (Omega_i ~ sqrt(Delta), P ~ Delta), and a
wider beam costs power as w0^2 while a tighter one increases the motional dOmega error.  Without
a power ceiling the optimizer simply runs Delta to its bound.  With one, the optimum sits ON the
power boundary and the trade-off is real.

    P_i = (pi/4) * w_i^2 * c * eps0 * (Omega_i / d_i)^2 ,  d_i = <a|d|b> in rad/s per (V/m)

For Cs 6S1/2 -> 7P1/2 (459 nm) -> nS1/2 (1038 nm) the dipole moments are 0.159 e.a0 and
~0.008 e.a0, so the 459 beam needs milliwatts and the 1038 beam needs WATTS: the IR arm is
essentially always the binding constraint.

Given the power box, the split between the arms is not free.  For a fixed product, scattering
grows with Omega1^2 + Omega2^2, so the balanced split Omega1 = Omega2 = sqrt(2*Omega_eff*Delta)
is optimal whenever it fits; otherwise the binding arm is clipped to its maximum and the other
arm takes up the slack.  That is done analytically here, so the split is not a free parameter.

Modes
-----
optimize          minimise the total over (n, atom_d, Omega_eff, Delta, w459, w1038, trap_depth)
scan              reproduce the budget_2_photon.py Omega scan at a fixed parameter point
optimize_and_scan both

Examples
--------
    python budget_2photon_optimize.py --mode optimize --p459-mW 200 --p1038-W 5 \\
        --opt-samples 300 --maxiter 12
    python budget_2photon_optimize.py --mode scan --n 70 --atom-d 5 --delta-ghz 5 \\
        --num-samples 10000
    python budget_2photon_optimize.py --power-report        # what the power bound allows

Deviation from budget_2_photon.py (deliberate)
----------------------------------------------
The scan script derives the motional dOmega error from the 459 beam profile alone.  The
two-photon Rabi frequency is proportional to the PRODUCT of the two field amplitudes, so this
script uses sqrt(I459_rel) * sqrt(I1038_rel).  With two different waists that matters.
"""

from budget_monte_carlo import *  # noqa: F401,F403
from budget_1photon_optimize import (batch_infidelity, _to_jsonable, stark_fit_field, get_atom,
                                     _load_alpha_cache, _save_alpha_cache)
from linear_response_2photon import (response_2photon, propagate_U_2photon, O2photon_I1,
                                     O2photon_I2, O2photon_nu1)
from linear_response import isometry_haar_full
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from arc import *  # noqa: F401,F403
import json
import pickle
import scipy.linalg
import scipy.optimize as opt
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
c = 299792458
kb = 1.380649e-23
epi0 = 8.854e-12
bohr_r = 5.291e-11


DEFAULT_CONFIG = dict(
    atom_name="Cs",
    l=0, j=1 / 2, mj=-1 / 2,          # Rydberg target nS1/2
    intermediate_n=7, intermediate_l=1, intermediate_j=1 / 2,
    lambda_1=0.459,                    # um, ground -> intermediate  (arm 1)
    lambda_2=1.038,                    # um, intermediate -> Rydberg (arm 2)
    pulse_time=7.65,
    resolution=200,
    scatter_resolution=10000,          # steps for the 3-level scattering ODE (as in the scan
                                       # script; 2000 steps is 0.08% off and saves only 0.14 s)
    Bz=10.0,
    T_atom=1.0,
    lambda_trap=1.064,
    w0_trap=1.0,
    edc_fluc=1e-3,                     # V/cm
    edc_zero=0.0,
    bdc_fluc=1e-3,                     # G
    pol_dc=None,                       # MHz/(V/cm)^2; None -> ARC StarkMap on nS1/2
    rin_strength=1e-4,
    f_hz_hz2=220.0,                    # arm 1 white FM-noise PSD [Hz^2/Hz]
    f_hz_hz2_2=220.0,                  # arm 2
    f_range=1e5,                       # Hz
    # ---- the power bound ----
    p1_max_W=0.2,                      # 459 nm available power [W]
    p2_max_W=5.0,                      # 1038 nm available power [W]
)

PARAMETER_NAMES = ["n", "atom_d_um", "Omega_eff_MHz", "Delta_GHz", "w459_um", "w1038_um",
                   "trap_depth_uK", "log10_Omega1_over_Omega2"]

#                      n     d[um] Om_eff  Delta[GHz] w459  w1038   U[uK]  log10(Om1/Om2)
DEFAULT_X0 = np.array([70.0, 5.0, 8.0, 5.0, 10.0, 10.0, 1000.0, 0.0], dtype=float)

DEFAULT_BOUNDS = [
    (40.0, 99.0),      # n           (S-state blockade table)
    (2.0, 15.0),       # atom_d_um   (S-state blockade table covers 2-15 um)
    (1.0, 25.0),       # Omega_eff/2pi [MHz]
    (0.5, 40.0),       # Delta/2pi [GHz]
    (2.0, 40.0),       # w459 [um]
    (2.0, 40.0),       # w1038 [um]
    (20.0, 5000.0),    # trap depth [uK]
    (-1.0, 2.0),       # log10(Omega1/Omega2); 0 = balanced, clipped into the power box
]

TOTAL_CHANNELS = ["TO", "blockade", "rabi", "scattering", "efield", "bfield", "doppler",
                  "vnoise", "RIN", "decay"]

_BIG = 1e9

with open("blockades_symmetric_s.pkl", "rb") as _f:
    blockade2_dict = pickle.load(_f)


# -----------------------------
# Blockade (S states)
# -----------------------------
def find_blockade_Mrad_2photon(atom_name, n, d):
    entry = blockade2_dict[(atom_name, atom_name, int(n), int(n))]
    return np.interp(d, entry["r"], entry["B"]) * 1e3 * 2 * np.pi


@lru_cache(maxsize=256)
def blockade_table_range_2photon(atom_name, n):
    try:
        entry = blockade2_dict[(atom_name, atom_name, int(n), int(n))]
    except KeyError:
        return None
    return float(np.min(entry["r"])), float(np.max(entry["r"]))


def available_n_2photon(atom_name):
    return sorted({k[2] for k in blockade2_dict if k[0] == atom_name})


# -----------------------------
# Laser power  <->  Rabi frequency
# -----------------------------
def _dipole_rad_per_s_per_Vm(d_ea0):
    """Dipole matrix element from ARC (units of e*a0) to rad/s per (V/m)."""
    return abs(float(d_ea0)) * bohr_r * e / hbar


def beam_power_W(Omega_rad_per_us, d_ea0, w0_um):
    """Power needed in a Gaussian beam of waist w0 to drive Rabi frequency Omega.

    I_peak = 2P/(pi w0^2),  E = sqrt(2 I/(c eps0)),  Omega = d E.
    """
    d = _dipole_rad_per_s_per_Vm(d_ea0)
    E = float(Omega_rad_per_us) * 1e6 / d                       # V/m
    return np.pi * (float(w0_um) * 1e-6) ** 2 * c * epi0 * E ** 2 / 4.0


def max_rabi_rad_per_us(P_max_W, d_ea0, w0_um):
    """Largest Rabi frequency [rad/us] this beam can reach with P_max in waist w0."""
    d = _dipole_rad_per_s_per_Vm(d_ea0)
    E = np.sqrt(4.0 * float(P_max_W) / (np.pi * (float(w0_um) * 1e-6) ** 2 * c * epi0))
    return d * E / 1e6


def split_from_ratio(omega_product, log10_ratio, om1_max, om2_max):
    """Split Omega1*Omega2 = omega_product at the requested ratio, inside the power box.

    The ratio is a genuine degree of freedom, not something to fix analytically.  Scattering
    grows with Omega1^2 + Omega2^2 and so prefers a balanced split, but the RIN response does
    not: arm 2's intensity-noise coefficient (ktilde_r2 - 1/4Delta) saturates near the ground
    state's 1038 nm polarizability while arm 1's (ktilde_r1 + 1/4Delta) falls as 1/Delta, so the
    2-photon RIN is dominated by arm 2 and is reduced by moving power INTO the blue arm.  At
    n=70, Delta/2pi = 5 GHz the summed RIN response falls from 149 (balanced) to 71 at
    Omega1/Omega2 = 4.  That is exactly what the 459 nm power ceiling then limits.

    r = Omega1/Omega2 is feasible when  prod/om2_max^2 <= r <= om1_max^2/prod;  outside that the
    ratio is CLIPPED (not rejected) so the objective stays continuous for the optimizer.

    Returns (Omega1, Omega2, clipped) in rad/us, or None when the box cannot reach the product.
    """
    if om1_max * om2_max < omega_product:
        return None
    r = 10.0 ** float(log10_ratio)
    r_lo = omega_product / om2_max ** 2
    r_hi = om1_max ** 2 / omega_product
    r_clipped = min(max(r, r_lo), r_hi)
    Omega1 = np.sqrt(omega_product * r_clipped)
    Omega2 = np.sqrt(omega_product / r_clipped)
    return Omega1, Omega2, bool(abs(r_clipped - r) > 1e-12 * max(r, 1.0))


# -----------------------------
# Cached atomic quantities
# -----------------------------
@lru_cache(maxsize=4096)
def atom_quantities_2photon(atom_name, n, cfg_key):
    """(R_lifetime, tau_e, m_atom, d1, d2) -- everything independent of Delta."""
    atom = get_atom(atom_name)
    n_i, l_i, j_i = cfg_key[0], cfg_key[1], cfg_key[2]
    n_g = 6 if atom_name == "Cs" else 5

    R_lifetime = atom.getStateLifetime(n=int(n), l=0, j=0.5, temperature=300,
                                       includeLevelsUpTo=int(n) + 20, s=0.5) * 1e6
    tau_e = atom.getStateLifetime(n=int(n_i), l=int(l_i), j=float(j_i), temperature=300,
                                  includeLevelsUpTo=int(n) + 20, s=0.5) * 1e6
    d1 = atom.getDipoleMatrixElement(n1=n_g, l1=0, j1=0.5, mj1=-0.5,
                                     n2=int(n_i), l2=int(l_i), j2=float(j_i), mj2=0.5, q=1, s=0.5)
    d2 = atom.getDipoleMatrixElement(n1=int(n_i), l1=int(l_i), j1=float(j_i), mj1=0.5,
                                     n2=int(n), l2=0, j2=0.5, mj2=-0.5, q=-1, s=0.5)
    return R_lifetime, tau_e, atom.mass, float(d1), float(d2)


@lru_cache(maxsize=64)
def _dynamic_pol_generators(atom_name, n, cfg_key):
    """ARC DynamicPolarizability generators, built once per n.

    linear_response_2photon.build_Oseq_2photon rebuilds these on every call; the optimizer calls
    it three times per evaluation, and Delta is a continuous variable, so caching the generators
    per n (and evaluating them at the actual frequency) is what makes an optimizer viable.
    """
    atom = get_atom(atom_name)
    n_g = 6 if atom_name == "Cs" else 5
    alpha_g_gen = DynamicPolarizability(atom, n=n_g, l=0, j=0.5, s=0.5)
    alpha_g_gen.defineBasis(n_g, 9)
    alpha_r_gen = DynamicPolarizability(atom, n=int(n), l=0, j=0.5, s=0.5)
    alpha_r_gen.defineBasis(n_g, int(n) + 20)
    return alpha_g_gen, alpha_r_gen


@lru_cache(maxsize=4096)
def ktilde_coefficients(atom_name, n, inter_detuning, cfg_key):
    """kappa-tilde light-shift coefficients (Eq. K7-K9) at this (n, Delta).

    Both polarizabilities are far-off-resonant background terms: they move by only ~1e-4 in
    relative terms between Delta/2pi = 0.5 and 40 GHz, but they are evaluated at the true
    frequency anyway since it costs ~0.05 s.
    """
    atom = get_atom(atom_name)
    n_i, l_i, j_i = cfg_key[0], cfg_key[1], cfg_key[2]
    n_g = 6 if atom_name == "Cs" else 5
    w_qubit = (9192631770 if atom_name == "Cs" else 6834682610) * 2 * np.pi

    v1 = atom.getTransitionFrequency(n1=n_g, l1=0, j1=0.5, n2=int(n_i), l2=int(l_i), j2=float(j_i),
                                     s=0.5) + inter_detuning * 1e6 / 2 / np.pi
    v2 = atom.getTransitionFrequency(n1=int(n_i), l1=int(l_i), j1=float(j_i), n2=int(n), l2=0,
                                     j2=0.5, s=0.5) - inter_detuning * 1e6 / 2 / np.pi

    alpha_g_gen, alpha_r_gen = _dynamic_pol_generators(atom_name, int(n), cfg_key)
    alpha_r_1 = alpha_r_gen.getPolarizability(c / v1, units="SI", accountForStateLifetime=False,
                                              mj=None)[0]
    alpha_1_2 = alpha_g_gen.getPolarizability(c / v2, units="SI", accountForStateLifetime=False,
                                              mj=None)[0]

    _, _, _, d1_ea0, d2_ea0 = atom_quantities_2photon(atom_name, n, cfg_key)
    d1 = d1_ea0 * bohr_r / hbar * e
    d2 = d2_ea0 * bohr_r / hbar * e

    ktilde0_1 = -(1 / 4 / (inter_detuning + w_qubit / 1e6) - 1 / 4 / inter_detuning)
    ktilder_1 = -(alpha_r_1 * 2 * np.pi * 1e6) / 4 / d1 ** 2
    ktilde0_2 = 0.0
    ktilder_2 = (alpha_1_2 * 2 * np.pi * 1e6) / 4 / d2 ** 2
    return ktilde0_1, ktilder_1, ktilde0_2, ktilder_2


@lru_cache(maxsize=4096)
def polarizability_S(atom_name, n, Bz, pol_dc_input):
    """DC polarizability of nS1/2 [MHz/(V/cm)^2], fitted inside the Inglis-Teller limit."""
    if pol_dc_input is not None:
        return float(pol_dc_input)
    key = f"{atom_name}|S|n={int(n)}|Bz={float(Bz):.6g}|IT"
    cache = _load_alpha_cache()
    if key in cache:
        return float(cache[key])
    atom = get_atom(atom_name)
    calc = StarkMap(atom)
    calc.defineBasis(n=int(n), l=0, j=0.5, mj=-0.5, nMin=int(n) - 20, nMax=int(n) + 30, maxL=5,
                     Bz=float(Bz) / 10000)
    calc.diagonalise(np.linspace(0, stark_fit_field(n), 60))
    pol = float(calc.getPolarizability(debugOutput=False))
    cache[key] = pol
    _save_alpha_cache(cache)
    return pol


def build_Oseq_2photon_cached(phases, dt, B, Omega1, Omega2, delta1, delta2, Delta,
                              inter_detuning, n, Oinst_func, ktildes):
    """build_Oseq_2photon with the kappa-tildes passed in instead of recomputed per call."""
    k01, kr1, k02, kr2 = ktildes
    Us = propagate_U_2photon(phases, dt, B, Omega1, Omega2, delta1, delta2, Delta,
                             inter_detuning, n, k01, kr1, k02, kr2)
    Udag = np.conjugate(np.swapaxes(Us, 1, 2))
    Oseq = np.empty((len(phases), Us.shape[1], Us.shape[2]), dtype=complex)
    for k in range(len(phases)):
        Oinst = Oinst_func(phases[k], Omega1, Omega2, Delta, k01, kr1, k02, kr2)
        Oseq[k] = Udag[k] @ Oinst @ Us[k]
    return Oseq


def intermediate_scattering(phases, dt, Omega_Rabi, Omega1, Omega2, inter_detuning, tau_e):
    """Loss off the intermediate state, by propagating the 3-level ladder with and without decay.

    Same construction as budget_2_photon.py (normalised units, |g>,|e>,|r>).
    """
    psi = np.array([1, 0, 0], dtype=complex)
    psi_no = np.array([1, 0, 0], dtype=complex)
    decay_rate = 1.0 / tau_e / Omega_Rabi
    o1 = Omega1 / Omega_Rabi / 2
    o2 = Omega2 / Omega_Rabi / 2
    det = inter_detuning / Omega_Rabi
    for phi in phases:
        ph = np.exp(-1j * phi)
        H0 = np.array([[0, o1 * ph, 0],
                       [o1 * np.conj(ph), det, o2],
                       [0, o2, 0]], dtype=complex)
        H = H0 + np.diag([0, -1j * decay_rate / 2, 0])
        psi = scipy.linalg.expm(-1j * H * dt) @ psi
        psi_no = scipy.linalg.expm(-1j * H0 * dt) @ psi_no
    return float(np.sum(np.abs(psi_no ** 2)) - np.sum(np.abs(psi ** 2)))


# -----------------------------
# Core evaluation
# -----------------------------
def vector_to_params(x):
    x = np.asarray(x, dtype=float)
    return dict(
        n=int(round(x[0])),
        atom_d=float(x[1]),
        Omega_eff_MHz=float(x[2]),
        Omega_Rabi=float(x[2]) * 2 * np.pi,          # rad/us
        Delta_GHz=float(x[3]),
        inter_detuning=float(x[3]) * 1000.0 * 2 * np.pi,  # rad/us
        w459=float(x[4]),
        w1038=float(x[5]),
        trap_depth=float(x[6]),
        log10_ratio=float(x[7]) if len(x) > 7 else 0.0,
    )


def params_to_vector(n, atom_d, omega_eff_mhz, delta_ghz, w459, w1038, trap_depth,
                     log10_ratio=0.0):
    return np.array([n, atom_d, omega_eff_mhz, delta_ghz, w459, w1038, trap_depth, log10_ratio],
                    dtype=float)


def evaluate_single_point(x, config=None, num_samples=500, seed=1234, optimize_phase=True,
                          verbose=False, return_details=False, batched=True):
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)

    p = vector_to_params(x)
    atom_name = cfg["atom_name"]
    n, atom_d = p["n"], p["atom_d"]
    Omega_Rabi, inter_detuning = p["Omega_Rabi"], p["inter_detuning"]
    w459, w1038, trap_depth = p["w459"], p["w1038"], p["trap_depth"]
    cfg_key = (cfg["intermediate_n"], cfg["intermediate_l"], cfg["intermediate_j"])

    def _reject(msg):
        if verbose:
            print(f"  rejected (n={n}, d={atom_d:.3g}, Delta={p['Delta_GHz']:.3g} GHz): {msg}")
        return (_BIG, dict(total=_BIG, invalid=msg)) if return_details else _BIG

    if min(n, atom_d, Omega_Rabi, inter_detuning, w459, w1038, trap_depth) <= 0:
        return _reject("non-physical parameter")
    r_range = blockade_table_range_2photon(atom_name, n)
    if r_range is None:
        return _reject(f"no tabulated blockade for {atom_name} n={n}")
    if not (r_range[0] <= atom_d <= r_range[1]):
        return _reject(f"atom_d outside the blockade table {r_range[0]}-{r_range[1]} um")

    R_lifetime, tau_e, m_atom, d1_ea0, d2_ea0 = atom_quantities_2photon(atom_name, n, cfg_key)

    # ---- the power constraint ----
    omega_product = 2.0 * Omega_Rabi * inter_detuning        # Omega1*Omega2, (rad/us)^2
    om1_max = max_rabi_rad_per_us(cfg["p1_max_W"], d1_ea0, w459)
    om2_max = max_rabi_rad_per_us(cfg["p2_max_W"], d2_ea0, w1038)
    split = split_from_ratio(omega_product, p["log10_ratio"], om1_max, om2_max)
    if split is None:
        need = np.sqrt(omega_product) / 2 / np.pi
        # Scale the penalty with how badly the power box is exceeded.  A flat penalty is a wall
        # with no gradient: Powell's line search stalls against it instead of walking back into
        # the feasible region, which under-converges every run whose optimum sits on the power
        # boundary (i.e. the interesting ones).
        deficit = omega_product / (om1_max * om2_max)
        if verbose:
            print(f"  infeasible (n={n}, Om_eff={p['Omega_eff_MHz']:.3g} MHz, "
                  f"Delta={p['Delta_GHz']:.3g} GHz, w={w459:.3g}/{w1038:.3g} um): "
                  f"need ({need:.0f} MHz)^2, box allows "
                  f"({np.sqrt(om1_max*om2_max)/2/np.pi:.0f} MHz)^2")
        val = _BIG * deficit
        return (val, dict(total=val, invalid="laser power insufficient")) if return_details else val
    Omega1, Omega2, ratio_clipped = split
    P1 = beam_power_W(Omega1, d1_ea0, w459)
    P2 = beam_power_W(Omega2, d2_ea0, w1038)

    blockade_mrad = find_blockade_Mrad_2photon(atom_name, n, atom_d)
    pol_dc = polarizability_S(atom_name, n, float(cfg["Bz"]), cfg["pol_dc"])
    ktildes = ktilde_coefficients(atom_name, n, inter_detuning, cfg_key)
    delta1 = ktildes[1] * Omega1 ** 2 + ktildes[3] * Omega2 ** 2
    delta2 = 0.0

    # ---- pulse phases ----
    H_gen = Hamiltonians(Omega_Rabi1=Omega_Rabi, blockade_inf=False, blockade=blockade_mrad,
                         r_lifetime=R_lifetime, Delta1=0, Stark1=0, Stark2=0,
                         resolution=cfg["resolution"], r_lifetime2=R_lifetime,
                         pulse_time=cfg["pulse_time"])
    PhaseGuess = [2 * np.pi * 0.1122, 1.0431, -0.7318, 0]
    if optimize_phase:
        phase_params = opt.minimize(fun=fid_optimize, x0=PhaseGuess, args=(H_gen,)).x
    else:
        phase_params = np.asarray(PhaseGuess, dtype=float)

    H_gen = Hamiltonians(Omega_Rabi1=Omega_Rabi, blockade_inf=False, blockade=blockade_mrad,
                         r_lifetime=1e10, Delta1=0, Stark1=0, Stark2=0,
                         resolution=cfg["resolution"], r_lifetime2=1e10,
                         pulse_time=cfg["pulse_time"])
    _, phase, dt = phase_cosine_generate(*phase_params, H_gen.pulse_time, H_gen.resolution)
    infid_TO = 1 - H_gen.return_fidel(phases=phase, dt=dt)[0]

    # ---- quasi-static detuning: two-photon Doppler + DC E + DC B ----
    k_eff = 2 * np.pi * (1 / cfg["lambda_1"] - 1 / cfg["lambda_2"]) / 1e-6   # counter-propagating
    doppler_shift = abs(k_eff) * np.sqrt(kb * (cfg["T_atom"] * 1e-6) / m_atom)
    delta_edc = abs(-0.5 * pol_dc * 1e6 * ((cfg["edc_zero"] + cfg["edc_fluc"]) ** 2
                                           - cfg["edc_zero"] ** 2)) * 2 * np.pi
    atom = get_atom(atom_name)
    delta_bdc = atom.getZeemanEnergyShift(l=0, j=0.5, mj=-0.5,
                                          magneticFieldBz=cfg["bdc_fluc"] / 10000) / hbar
    total_shift = np.sqrt(delta_bdc ** 2 + delta_edc ** 2 + doppler_shift ** 2)
    detunings = total_shift / 1e6

    rng_pos = np.random.default_rng([seed, 0])
    rng_det = np.random.default_rng([seed, 1])
    d_samples = rng_det.normal(0.0, detunings, size=int(num_samples))
    infids_s = batch_infidelity(phase, dt, Omega_Rabi, blockade_mrad, d_samples, 1.0, 1.0,
                                10e9, 10e9)
    infids_detuning = float(np.mean(infids_s) - infid_TO)
    sem_detuning = float(infids_s.std(ddof=1) / np.sqrt(len(infids_s)))
    if total_shift > 0:
        infids_bdc = delta_bdc ** 2 / total_shift ** 2 * infids_detuning
        infids_edc = delta_edc ** 2 / total_shift ** 2 * infids_detuning
        infids_doppler = doppler_shift ** 2 / total_shift ** 2 * infids_detuning
    else:
        infids_bdc = infids_edc = infids_doppler = 0.0

    # ---- motion ----
    sigma_r = sigma_r_um(cfg["T_atom"], trap_depth, cfg["w0_trap"])
    sigma_z = sigma_z_um(cfg["T_atom"], trap_depth, cfg["w0_trap"], cfg["lambda_trap"])
    ds, c1, c2 = sample_pair_distances(n_samples=int(num_samples), sigma_r=sigma_r,
                                       sigma_z=sigma_z, x_offset=atom_d, rng=rng_pos)
    x1, y1, z1 = c1["x"], c1["y"], c1["z"]
    x2, y2, z2 = c2["x"], c2["y"], c2["z"]

    def field_ratio(xx, zz, yy, w0, lam):
        ref = np.sqrt(relative_gaussian_beam_intensity(0, 0, 0 - atom_d / 2, w0, lam))
        return np.sqrt(relative_gaussian_beam_intensity(xx, zz, yy, w0, lam)) / ref

    # Omega_eff ~ E459 * E1038, so the two beam profiles multiply (the scan script uses the
    # 459 profile alone).
    rabis1 = (field_ratio(x1, z1, y1 - atom_d / 2, w459, cfg["lambda_1"]) *
              field_ratio(x1, z1, y1 - atom_d / 2, w1038, cfg["lambda_2"]))
    rabis2 = (field_ratio(x2 - atom_d, z2, y2 + atom_d / 2, w459, cfg["lambda_1"]) *
              field_ratio(x2 - atom_d, z2, y2 + atom_d / 2, w1038, cfg["lambda_2"]))

    blockades = find_blockade_Mrad_2photon(atom_name, n, ds)
    infids_blockade = batch_infidelity(phase, dt, Omega_Rabi, blockades, 0.0, 1.0, 1.0, 10e9, 10e9)
    infids_rabi = batch_infidelity(phase, dt, Omega_Rabi, blockade_mrad, 0.0, rabis1, rabis2,
                                   10e9, 10e9)
    infids_motion_blockade = float(np.mean(infids_blockade) - infid_TO)
    infids_motion_rabi = float(np.mean(infids_rabi) - infid_TO)

    # ---- intermediate-state scattering ----
    _, scatter_phase, scatter_dt = phase_cosine_generate(*phase_params, cfg["pulse_time"],
                                                         int(cfg["scatter_resolution"]))
    scattering_e = intermediate_scattering(scatter_phase, scatter_dt, Omega_Rabi, Omega1, Omega2,
                                           inter_detuning, tau_e)

    # ---- Rydberg decay ----
    infids_decay = (2.95 / Omega_Rabi) / R_lifetime

    # ---- linear response (both arms) ----
    S_haar = isometry_haar_full()
    dt_real = dt / Omega_Rabi                    # us; the 2-photon response uses real time
    common = dict(phases=phase, dt=dt_real, B=blockade_mrad, Omega1=Omega1, Omega2=Omega2,
                  delta1=delta1, delta2=delta2, Delta=inter_detuning,
                  inter_detuning=inter_detuning, n=n, ktildes=ktildes)
    o_nu1 = build_Oseq_2photon_cached(Oinst_func=O2photon_nu1, **common)
    If_2p = response_2photon(o_nu1, S_haar, 0, dt_real)
    vnoise = If_2p * (cfg["f_hz_hz2"] + cfg["f_hz_hz2_2"]) * cfg["f_range"] / 1e6 / 1e6

    o_I1 = build_Oseq_2photon_cached(Oinst_func=O2photon_I1, **common)
    o_I2 = build_Oseq_2photon_cached(Oinst_func=O2photon_I2, **common)
    RIN = (response_2photon(o_I1, S_haar, 0, dt_real) +
           response_2photon(o_I2, S_haar, 0, dt_real)) * cfg["rin_strength"]

    total = (infid_TO + infids_motion_blockade + infids_motion_rabi + scattering_e
             + infids_detuning + vnoise + RIN + infids_decay)

    details = dict(
        n=int(n), atom_d_um=float(atom_d), Omega_eff_MHz=float(p["Omega_eff_MHz"]),
        Delta_GHz=float(p["Delta_GHz"]), w459_um=float(w459), w1038_um=float(w1038),
        trap_depth_uK=float(trap_depth),
        TO=float(infid_TO), blockade=float(infids_motion_blockade),
        rabi=float(infids_motion_rabi), scattering=float(scattering_e),
        detuning=float(infids_detuning), efield=float(infids_edc), bfield=float(infids_bdc),
        doppler=float(infids_doppler), vnoise=float(vnoise), RIN=float(RIN),
        decay=float(infids_decay), leakage=float("nan"), total=float(total),
        Omega1_MHz=float(Omega1 / 2 / np.pi), Omega2_MHz=float(Omega2 / 2 / np.pi),
        P459_mW=float(P1 * 1e3), P1038_W=float(P2),
        P459_frac=float(P1 / cfg["p1_max_W"]), P1038_frac=float(P2 / cfg["p2_max_W"]),
        log10_ratio=float(np.log10(Omega1 / Omega2)), ratio_clipped_by_power=bool(ratio_clipped),
        blockade_MHz=float(blockade_mrad / 2 / np.pi), R_lifetime_us=float(R_lifetime),
        tau_intermediate_us=float(tau_e), pol_dc_MHz_per_Vcm2=float(pol_dc),
        d1_ea0=float(d1_ea0), d2_ea0=float(d2_ea0),
        mc_sem=dict(detuning=sem_detuning,
                    blockade=float(infids_blockade.std(ddof=1) / np.sqrt(len(infids_blockade))),
                    rabi=float(infids_rabi.std(ddof=1) / np.sqrt(len(infids_rabi)))),
        phase_params=_to_jsonable(np.asarray(phase_params)), num_samples=int(num_samples),
    )

    if verbose:
        print(f"n={n}, d={atom_d:.3g} um, Om_eff/2pi={p['Omega_eff_MHz']:.3g} MHz, "
              f"Delta/2pi={p['Delta_GHz']:.3g} GHz, w={w459:.3g}/{w1038:.3g} um, "
              f"U={trap_depth:.4g} uK | P={P1*1e3:.1f} mW/{P2:.2f} W "
              f"({100*P1/cfg['p1_max_W']:.0f}%/{100*P2/cfg['p2_max_W']:.0f}%) -> {total:.6e}")

    return (float(total), details) if return_details else float(total)


# -----------------------------
# Omega scan
# -----------------------------
def run_scan(x, config=None, num_samples=10000, seed=1234, result_dir="result", f_rabis=None,
             optimize_phase=True, make_plots=True, verbose=True, batched=True, tag="fixed"):
    os.makedirs(result_dir, exist_ok=True)
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)
    p0 = vector_to_params(x)
    if f_rabis is None:
        f_rabis = np.linspace(1, 25, 25)
    f_rabis = np.asarray(f_rabis, dtype=float)

    records = []
    for f in f_rabis:
        xx = np.asarray(x, dtype=float).copy()
        xx[2] = f
        _, details = evaluate_single_point(xx, config=cfg, num_samples=num_samples, seed=seed,
                                           optimize_phase=optimize_phase, verbose=verbose,
                                           return_details=True, batched=batched)
        records.append(details)

    def arr(key):
        return np.asarray([r.get(key, np.nan) for r in records], dtype=float)

    y = {k: arr(k) for k in ["TO", "vnoise", "RIN", "blockade", "rabi", "efield", "bfield",
                             "doppler", "decay", "scattering", "detuning", "leakage", "total"]}
    valid = np.array([("invalid" not in r) for r in records])
    if not valid.any():
        raise ValueError(f"no feasible point in the scan: {records[0].get('invalid')}")
    min_idx = int(np.nanargmin(np.where(valid, y["total"], np.inf)))
    min_total = float(y["total"][min_idx])

    if make_plots:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(f_rabis, y["vnoise"], c="#4e63ff", linewidth=2, label="$v$")
        ax.plot(f_rabis, y["RIN"], c="#ff4da6", linewidth=2, label="RIN")
        ax.plot(f_rabis, y["blockade"], c="#2ecc71", linewidth=2, label=r"$\delta V$")
        ax.plot(f_rabis, y["rabi"], linewidth=2, label=r"$\delta \Omega$")
        ax.plot(f_rabis, y["decay"], c="#7f8c8d", linewidth=2, label=r"$\gamma$")
        ax.plot(f_rabis, y["TO"], c="#b06ae2", linewidth=2, label="TO")
        ax.plot(f_rabis, y["scattering"], c="#ff9f1a", linewidth=2, label="scattering")
        ax.plot(f_rabis, y["total"], c="k", linewidth=4, label=r"$\Sigma$")
        ax.plot(f_rabis, y["efield"], linewidth=2, label="E")
        ax.plot(f_rabis, y["bfield"], linewidth=2, label="B")
        ax.plot(f_rabis, y["doppler"], linewidth=2, label="doppler")
        ax.axhline(1e-3, c="k", linestyle=":")
        ax.set_ylabel("Infidelity", fontsize=14)
        ax.set_xlabel(r"$\Omega/ 2\pi$ [MHz] ", fontsize=14)
        ax.set_yscale("log")
        ax.set_title(r"$2 \gamma$ gate", fontsize=16)
        ax.set_ylim([1e-9, 1e-3])
        ax.legend(fontsize=12)
        fig.savefig(os.path.join(result_dir, f"Error_vs_rabi_2photon_{tag}.pdf"),
                    bbox_inches="tight")
        plt.close(fig)

        bar_vals = [abs(float(y[k][min_idx])) for k in TOTAL_CHANNELS]
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(TOTAL_CHANNELS, bar_vals)
        ax.set_yscale("log")
        ax.set_ylabel("|Infidelity contribution|")
        ax.set_title(f"2-photon contributions at the minimum "
                     f"($\\Omega/2\\pi$={f_rabis[min_idx]:.3g} MHz, total={min_total:.3e})")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        fig.savefig(os.path.join(result_dir, f"contributions_at_min_2photon_{tag}.pdf"),
                    bbox_inches="tight")
        plt.close(fig)

    rec0 = records[min_idx]
    config_out = {k: _to_jsonable(v) for k, v in cfg.items()}
    config_out.update(
        n=int(p0["n"]), atom_d_um=float(p0["atom_d"]), Delta_GHz=float(p0["Delta_GHz"]),
        w459_um=float(p0["w459"]), w1038_um=float(p0["w1038"]),
        trap_depth_uK=float(p0["trap_depth"]), num_samples=int(num_samples), seed=int(seed),
        f_Rabi_scan_MHz=dict(start=float(f_rabis[0]), stop=float(f_rabis[-1]),
                             num=int(len(f_rabis))),
        derived=dict(blockade_MHz=float(rec0["blockade_MHz"]),
                     R_lifetime_us=float(rec0["R_lifetime_us"]),
                     pol_dc_MHz_per_Vcm2=float(rec0["pol_dc_MHz_per_Vcm2"]),
                     P459_mW=float(rec0["P459_mW"]), P1038_W=float(rec0["P1038_W"])),
        minimum=dict(index=min_idx, Omega_eff_MHz=float(f_rabis[min_idx]), total=float(min_total)),
    )
    out = dict(
        meta=dict(script="budget_2photon_optimize.py"),
        config=config_out,
        raw_fig2=dict(x=dict(name="Omega_over_2pi_MHz", values=_to_jsonable(f_rabis)),
                      y={k: _to_jsonable(v) for k, v in y.items()}),
        records=[{k: _to_jsonable(v) for k, v in r.items()} for r in records],
    )
    out_json = os.path.join(result_dir, f"scan_2photon_n{int(p0['n'])}_{tag}_config_and_raw.json")
    with open(out_json, "w") as fh:
        json.dump(out, fh, indent=2)
    if verbose:
        print(f"min infid: {min_total:.6e} at Omega/2pi = {f_rabis[min_idx]:.6g} MHz")
        print(f"Saved config + raw scan data to: {out_json}")
    return out, out_json


# -----------------------------
# Optimization
# -----------------------------
def run_optimization(x0=DEFAULT_X0, bounds=DEFAULT_BOUNDS, config=None, opt_samples=300,
                     seed=1234, maxiter=15, result_dir="result", optimize_phase=True,
                     batched=True, verbose=True, restarts=3):
    os.makedirs(result_dir, exist_ok=True)
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)
    bounds = [tuple(float(v) for v in b) for b in bounds]
    ns = available_n_2photon(cfg["atom_name"])
    bounds[0] = (max(bounds[0][0], ns[0]), min(bounds[0][1], ns[-1]))
    x0 = np.clip(np.asarray(x0, dtype=float), [b[0] for b in bounds], [b[1] for b in bounds])

    history = []
    incumbent = dict(x=None, f=np.inf)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])

    def objective(xin):
        x = np.asarray(xin, dtype=float)
        if np.any(x < lo) or np.any(x > hi):
            return _BIG + float(np.sum((np.maximum(lo - x, 0) + np.maximum(x - hi, 0)) ** 2))
        val = evaluate_single_point(x, config=cfg, num_samples=opt_samples, seed=seed,
                                    optimize_phase=optimize_phase, verbose=verbose,
                                    return_details=False, batched=batched)
        history.append(dict(x=_to_jsonable(x), total=float(val)))
        if val < incumbent["f"]:
            incumbent["f"], incumbent["x"] = float(val), x.copy()
        return float(val)

    # Powell is start-point sensitive on this objective and regularly reports "terminated
    # successfully" after one cycle of line searches that each found nothing, while a restart
    # from the same point goes on to improve by a factor of 2.  Restart from the incumbent until
    # a pass buys less than 1%.
    result = None
    best_x, best_fun = np.asarray(x0, dtype=float), np.inf
    for k in range(max(1, int(restarts))):
        result = opt.minimize(objective, best_x, method="Powell", bounds=bounds,
                              options=dict(maxiter=int(maxiter), disp=verbose, xtol=1e-3,
                                           ftol=1e-6))
        new_x = np.asarray(incumbent["x"] if incumbent["f"] < result.fun else result.x,
                           dtype=float)
        new_f = float(min(incumbent["f"], result.fun))
        gain = (best_fun - new_f) / abs(best_fun) if np.isfinite(best_fun) else 1.0
        best_x, best_fun = new_x, new_f
        if verbose:
            print(f"  [restart {k + 1}/{restarts}] best = {best_fun:.6e}"
                  f"{'' if k == 0 else f' (gain {100 * gain:.1f}%)'}")
        if k > 0 and gain < 0.01:
            break
    message = str(result.message)
    # Same guard as the 1-photon driver: Powell can finish on a worse point than it visited.
    if incumbent["x"] is not None and incumbent["f"] < best_fun:
        best_x, best_fun = np.asarray(incumbent["x"], dtype=float), float(incumbent["f"])
        message += " (returned the best evaluated point, not the final iterate)"

    _, best_details = evaluate_single_point(best_x, config=cfg, num_samples=opt_samples, seed=seed,
                                            optimize_phase=optimize_phase, verbose=False,
                                            return_details=True, batched=batched)
    out = dict(best_x=_to_jsonable(best_x), best_infidelity=float(best_fun),
               best_breakdown={k: _to_jsonable(v) for k, v in best_details.items()},
               parameter_names=PARAMETER_NAMES, bounds=[list(b) for b in bounds],
               success=bool(result.success), message=message, opt_samples=int(opt_samples),
               maxiter=int(maxiter), seed=int(seed),
               config={k: _to_jsonable(v) for k, v in cfg.items()}, history=history)
    out_json = os.path.join(result_dir, "optimization_result_2photon.json")
    with open(out_json, "w") as fh:
        json.dump(out, fh, indent=2)

    print("\n===== 2-PHOTON OPTIMIZATION RESULT =====")
    print("success:", result.success, "|", message)
    print("evaluations:", len(history))
    for name, val in zip(PARAMETER_NAMES, best_x):
        print(f"  {name}: {int(round(val)) if name == 'n' else round(val, 4)}")
    print(f"  -> Omega1/2pi = {best_details['Omega1_MHz']:.1f} MHz, "
          f"Omega2/2pi = {best_details['Omega2_MHz']:.1f} MHz "
          f"(ratio {10**best_details['log10_ratio']:.2f}"
          f"{', clipped by the power box' if best_details['ratio_clipped_by_power'] else ''})")
    print(f"  -> P(459) = {best_details['P459_mW']:.1f} mW "
          f"({100*best_details['P459_frac']:.0f}% of budget), "
          f"P(1038) = {best_details['P1038_W']:.2f} W "
          f"({100*best_details['P1038_frac']:.0f}% of budget)")
    print("best infidelity:", best_fun)
    print("breakdown at the optimum:")
    for k in TOTAL_CHANNELS:
        print(f"  {k:12s} {best_details[k]:.3e}")
    print(f"Saved optimization result to: {out_json}")
    return result, best_x, out_json


def power_report(cfg, n=70, w459=10.0, w1038=10.0):
    """What the power bound allows, before any gate simulation."""
    cfg_key = (cfg["intermediate_n"], cfg["intermediate_l"], cfg["intermediate_j"])
    _, _, _, d1, d2 = atom_quantities_2photon(cfg["atom_name"], n, cfg_key)
    om1 = max_rabi_rad_per_us(cfg["p1_max_W"], d1, w459)
    om2 = max_rabi_rad_per_us(cfg["p2_max_W"], d2, w1038)
    print(f"{cfg['atom_name']} n={n}, waists {w459}/{w1038} um")
    print(f"  d1 = {d1:.4f} e.a0   d2 = {d2:.5f} e.a0")
    print(f"  P(459) <= {cfg['p1_max_W']*1e3:.0f} mW  ->  Omega1/2pi <= {om1/2/np.pi:.0f} MHz")
    print(f"  P(1038) <= {cfg['p2_max_W']:.2f} W  ->  Omega2/2pi <= {om2/2/np.pi:.0f} MHz")
    print(f"\n  Largest Omega_eff/2pi [MHz] reachable vs Delta:")
    print(f"  {'Delta/2pi[GHz]':>15} {'max Om_eff/2pi[MHz]':>21}")
    for D in [1, 2, 5, 10, 20, 40]:
        om_eff = om1 * om2 / (2 * D * 1000 * 2 * np.pi)
        print(f"  {D:15.0f} {om_eff/2/np.pi:21.2f}")


def parse_args():
    p = argparse.ArgumentParser(description="Bounded 2-photon gate optimization with a laser-power limit.")
    p.add_argument("--mode", choices=["optimize", "scan", "optimize_and_scan"], default="optimize")
    p.add_argument("--result-dir", default="result")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--power-report", action="store_true",
                   help="Print what the power bound allows and exit.")

    p.add_argument("--n", type=float, default=DEFAULT_X0[0])
    p.add_argument("--atom-d", type=float, default=DEFAULT_X0[1])
    p.add_argument("--omega-eff-mhz", type=float, default=DEFAULT_X0[2])
    p.add_argument("--delta-ghz", type=float, default=DEFAULT_X0[3])
    p.add_argument("--w459", type=float, default=DEFAULT_X0[4])
    p.add_argument("--w1038", type=float, default=DEFAULT_X0[5])
    p.add_argument("--trap-depth", type=float, default=DEFAULT_X0[6])

    p.add_argument("--n-bounds", type=float, nargs=2, default=DEFAULT_BOUNDS[0])
    p.add_argument("--atom-d-bounds", type=float, nargs=2, default=DEFAULT_BOUNDS[1])
    p.add_argument("--omega-eff-bounds", type=float, nargs=2, default=DEFAULT_BOUNDS[2])
    p.add_argument("--delta-bounds", type=float, nargs=2, default=DEFAULT_BOUNDS[3])
    p.add_argument("--w459-bounds", type=float, nargs=2, default=DEFAULT_BOUNDS[4])
    p.add_argument("--w1038-bounds", type=float, nargs=2, default=DEFAULT_BOUNDS[5])
    p.add_argument("--trap-depth-bounds", type=float, nargs=2, default=DEFAULT_BOUNDS[6])
    p.add_argument("--log10-ratio", type=float, default=DEFAULT_X0[7],
                   help="log10(Omega1/Omega2); 0 = balanced arms.")
    p.add_argument("--ratio-bounds", type=float, nargs=2, default=DEFAULT_BOUNDS[7])

    # The power bound.
    p.add_argument("--p459-mW", type=float, default=DEFAULT_CONFIG["p1_max_W"] * 1e3,
                   help="Available 459 nm power at the atoms [mW].")
    p.add_argument("--p1038-W", type=float, default=DEFAULT_CONFIG["p2_max_W"],
                   help="Available 1038 nm power at the atoms [W].")

    p.add_argument("--opt-samples", type=int, default=300)
    p.add_argument("--num-samples", type=int, default=10000)
    p.add_argument("--maxiter", type=int, default=15)
    p.add_argument("--restarts", type=int, default=3,
                   help="Restart Powell from its own best point up to this many times; stops "
                        "early when a pass gains less than 1%%.")
    p.add_argument("--no-phase-opt", action="store_true")
    p.add_argument("--scan-points", type=int, default=25)
    p.add_argument("--scan-min-mhz", type=float, default=1.0)
    p.add_argument("--scan-max-mhz", type=float, default=25.0)

    p.add_argument("--atom", choices=["Cs", "Rb"], default=DEFAULT_CONFIG["atom_name"])
    p.add_argument("--Bz", type=float, default=DEFAULT_CONFIG["Bz"])
    p.add_argument("--T-atom", type=float, default=DEFAULT_CONFIG["T_atom"])
    p.add_argument("--rin-strength", type=float, default=DEFAULT_CONFIG["rin_strength"])
    p.add_argument("--f-noise", type=float, default=DEFAULT_CONFIG["f_hz_hz2"])
    p.add_argument("--f-noise-2", type=float, default=DEFAULT_CONFIG["f_hz_hz2_2"])
    p.add_argument("--edc-fluc", type=float, default=DEFAULT_CONFIG["edc_fluc"])
    p.add_argument("--bdc-fluc", type=float, default=DEFAULT_CONFIG["bdc_fluc"])
    p.add_argument("--pol-dc", type=float, default=None)
    p.add_argument("--scatter-resolution", type=int,
                   default=DEFAULT_CONFIG["scatter_resolution"])
    return p.parse_args()


def config_from_args(args):
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(atom_name=args.atom, Bz=args.Bz, T_atom=args.T_atom,
               rin_strength=args.rin_strength, f_hz_hz2=args.f_noise, f_hz_hz2_2=args.f_noise_2,
               edc_fluc=args.edc_fluc, bdc_fluc=args.bdc_fluc,
               scatter_resolution=args.scatter_resolution,
               p1_max_W=args.p459_mW * 1e-3, p2_max_W=args.p1038_W)
    if args.pol_dc is not None:
        cfg["pol_dc"] = args.pol_dc
    return cfg


def main():
    args = parse_args()
    cfg = config_from_args(args)
    if args.power_report:
        power_report(cfg, n=int(round(args.n)), w459=args.w459, w1038=args.w1038)
        return

    x0 = params_to_vector(args.n, args.atom_d, args.omega_eff_mhz, args.delta_ghz, args.w459,
                          args.w1038, args.trap_depth, args.log10_ratio)
    bounds = [tuple(args.n_bounds), tuple(args.atom_d_bounds), tuple(args.omega_eff_bounds),
              tuple(args.delta_bounds), tuple(args.w459_bounds), tuple(args.w1038_bounds),
              tuple(args.trap_depth_bounds), tuple(args.ratio_bounds)]
    f_rabis = np.linspace(args.scan_min_mhz, args.scan_max_mhz, args.scan_points)
    optimize_phase = not args.no_phase_opt

    if args.mode in ("optimize", "optimize_and_scan"):
        _, best_x, _ = run_optimization(x0=x0, bounds=bounds, config=cfg,
                                        opt_samples=args.opt_samples, seed=args.seed,
                                        maxiter=args.maxiter, result_dir=args.result_dir,
                                        optimize_phase=optimize_phase, restarts=args.restarts)
        if args.mode == "optimize_and_scan":
            run_scan(best_x, config=cfg, num_samples=args.num_samples, seed=args.seed,
                     result_dir=args.result_dir, f_rabis=f_rabis, optimize_phase=optimize_phase,
                     tag="optimized")
    else:
        run_scan(x0, config=cfg, num_samples=args.num_samples, seed=args.seed,
                 result_dir=args.result_dir, f_rabis=f_rabis, optimize_phase=optimize_phase,
                 tag="fixed")


if __name__ == "__main__":
    main()
