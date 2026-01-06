#!/usr/bin/env python3
"""
Cs 60P: estimate *total detected* BBR-induced ionization rate following the structure of
Beterov et al. (NJP 2009):  W_BBR^tot = W_BBR + W_SFI + W_BBR^mix + W_SFI^mix

What this script does (practical replication tool):
  1) W_BBR (direct BBR photoionization) via their analytic Eq. (27).
  2) W_SFI (BBR-driven population transfer to states above field-ionization threshold)
     via their analytic Eq. (30).
  3) "Mixing" terms via a small rate-equation model over nearby n' states (n±N)
     using hydrogenic-scaled dipole matrix elements as an approximation.

IMPORTANT:
  - Steps (1) and (2) match the paper’s *analytic* formulae closely.
  - Step (3) is an *approximate* mixing model unless you replace the transition rates
    with Dyachkov–Pankratov / GDK-based matrix elements like the paper’s numerics.

If you want closer agreement to Fig. 13 for Cs, I can help you swap in DP/GDK matrix elements
once you tell me whether your 60P is P1/2 or P3/2 (quantum defects differ).
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, Tuple, List
import matplotlib.pyplot as plt
# ----------------------------
# Atomic / physical constants
# ----------------------------
# Atomic units (a.u.) helpers
AU_EFIELD_V_PER_CM = 5.142206747e9  # 1 a.u. electric field in V/cm
AU_TIME_S = 2.4188843265857e-17     # 1 a.u. time in seconds
AU_ENERGY_J = 4.3597447222071e-18   # 1 a.u. energy in joules
K_B_J_PER_K = 1.380649e-23
C_AU = 137.035999084  # speed of light in atomic units

# Planck occupancy nbar(omega) with omega in a.u. (Hartree), T in K
def nbar(omega_au: float, T: float) -> float:
    if omega_au <= 0:
        return 0.0
    x = (omega_au * AU_ENERGY_J) / (K_B_J_PER_K * T)
    # for large x, nbar ~ exp(-x)
    if x > 700:
        return 0.0
    return 1.0 / (math.exp(x) - 1.0)


# ----------------------------
# Cesium quantum defects
# ----------------------------
# Representative Cs defects consistent with the paper's differences:
#   mu_S - mu_P = 0.458701
#   mu_P - mu_D = 1.12661
#
# NOTE: For highest fidelity you should set mu_P to the correct series (P1/2 vs P3/2).
MU: Dict[str, float] = {
    "S": 4.049325,
    "P": 3.590624,
    "D": 2.464014,
    "F": 0.031064,
}
LVAL = {"S": 0, "P": 1, "D": 2, "F": 3}
LNAME = {v: k for k, v in LVAL.items()}

# Cs A_L coefficients from Beterov Table 2
A_L = {"S": 0.85, "P": 1.10, "D": 0.35}


# ----------------------------
# Beterov analytic formulas
# ----------------------------
def w_bbr_eq27(n: int, L: str, T: float) -> float:
    """
    Direct BBR photoionization rate W_BBR (s^-1) using the paper's analytic Eq. (27).
    Uses n_eff = n - mu_L and phase terms from quantum-defect differences.
    """
    if L not in A_L:
        raise ValueError(f"Unsupported L='{L}'. Use one of {list(A_L.keys())}.")

    muL = MU[L]
    n_eff = n - muL
    if n_eff <= 0:
        raise ValueError("n_eff <= 0. Check quantum defect.")

    # prefactor
    pref = A_L[L] * (11500.0 * T) / (n_eff ** (7.0 / 3.0))

    # log term: ln( 1 / (1 - exp(-157890/(T n_eff^2))) )
    x = -157890.0 / (T * (n_eff ** 2))
    ex = math.exp(x) if x > -745 else 0.0
    log_term = math.log(1.0 / (1.0 - ex))

    # phase deltas
    Lnum = LVAL[L]

    def delta(mu_a: float, mu_b: float) -> float:
        return math.pi * (mu_a - mu_b)

    if L == "S":
        # paper keeps only L -> L+1 channel for S in Eq. (27)
        muLp1 = MU[LNAME[Lnum + 1]]
        dplus = delta(muL, muLp1)
        phase = math.cos(dplus + math.pi / 6.0) ** 2
        return pref * phase * log_term

    muLp1 = MU[LNAME[Lnum + 1]]
    muLm1 = MU[LNAME[Lnum - 1]]
    dplus = delta(muL, muLp1)
    dminus = delta(muLm1, muL)
    phase = (math.cos(dplus + math.pi / 6.0) ** 2) + (math.cos(dminus - math.pi / 6.0) ** 2)
    return pref * phase * log_term


def n_crit_from_field(E_v_per_cm: float) -> float:
    """
    Classical adiabatic field-ionization threshold:
      F (a.u.) ≈ 1 / (16 n^4)
      => n_c = (1/(16F))^(1/4)
    """
    F_au = E_v_per_cm / AU_EFIELD_V_PER_CM
    if F_au <= 0:
        return float("inf")
    return (1.0 / (16.0 * F_au)) ** 0.25


def w_sfi_eq30(n: int, L: str, T: float, E_v_per_cm: float) -> float:
    """
    BBR-induced SFI contribution W_SFI (s^-1) using the paper's analytic Eq. (30).
    Implemented in the same style as Eq. (27), with a log-difference involving n_c.
    """
    if E_v_per_cm <= 0.0:
        return 0.0
    if L not in A_L:
        raise ValueError(f"Unsupported L='{L}'. Use one of {list(A_L.keys())}.")

    muL = MU[L]
    n_eff = n - muL
    if n_eff <= 0:
        raise ValueError("n_eff <= 0. Check quantum defect.")

    n_c = n_crit_from_field(E_v_per_cm)

    # If the threshold is above the state (very weak field), SFI contribution vanishes
    if n_c >= 5e6:
        return 0.0

    # prefactor (paper shows n^(7/3); neff is a better alkali extension)
    pref = A_L[L] * (11500.0 * T) / (n_eff ** (7.0 / 3.0))

    # phase terms
    Lnum = LVAL[L]

    def delta(mu_a: float, mu_b: float) -> float:
        return math.pi * (mu_a - mu_b)

    if L == "S":
        muLp1 = MU[LNAME[Lnum + 1]]
        dplus = delta(muL, muLp1)
        phase = math.cos(dplus + math.pi / 6.0) ** 2
    else:
        muLp1 = MU[LNAME[Lnum + 1]]
        muLm1 = MU[LNAME[Lnum - 1]]
        dplus = delta(muL, muLp1)
        dminus = delta(muLm1, muL)
        phase = (math.cos(dplus + math.pi / 6.0) ** 2) + (math.cos(dminus - math.pi / 6.0) ** 2)

    # bracket term in Eq. (30):
    #   ln[1/(1-exp(157890/(T n_c^2) - 157890/(T n_eff^2)))] - ln[1/(1-exp(-157890/(T n_eff^2)))]
    a = 157890.0 / T
    term1_exp = a * (1.0 / (n_c ** 2) - 1.0 / (n_eff ** 2))
    # exp(term1_exp) might be >1; but it appears inside 1-exp(...), so guard.
    # For typical parameters, term1_exp is negative.
    if term1_exp > 700:
        exp1 = float("inf")
    elif term1_exp < -745:
        exp1 = 0.0
    else:
        exp1 = math.exp(term1_exp)

    # first log
    denom1 = 1.0 - exp1
    if denom1 <= 0:
        # If this happens, you’re outside the intended regime; return 0 rather than NaN.
        log1 = 0.0
    else:
        log1 = math.log(1.0 / denom1)

    # second log
    x2 = -a / (n_eff ** 2)
    exp2 = math.exp(x2) if x2 > -745 else 0.0
    log2 = math.log(1.0 / (1.0 - exp2))

    bracket = log1 - log2
    if bracket <= 0:
        return 0.0

    return pref * phase * bracket


# ----------------------------
# Approximate mixing model (rate equations)
# ----------------------------
@dataclass(frozen=True)
class RydbergState:
    n: int
    L: str  # "S","P","D"
    def neff(self) -> float:
        return self.n - MU[self.L]


def omega_transition_au(a: RydbergState, b: RydbergState) -> float:
    """
    Transition angular frequency in atomic units (Hartree) using hydrogenic energies:
      E = -1/(2 n_eff^2)
    Returns positive for upward transitions (absorption).
    """
    Ea = -0.5 / (a.neff() ** 2)
    Eb = -0.5 / (b.neff() ** 2)
    return (Eb - Ea)  # Hartree (a.u.)


def dipole_r_au_hydrogenic_scale(a: RydbergState, b: RydbergState) -> float:
    """
    Crude hydrogenic scaling for radial dipole matrix element |<a|r|b>| in a.u.
    For nearby Rydberg levels, order of magnitude ~ n_eff^2.
    """
    n_eff = 0.5 * (a.neff() + b.neff())
    return 1.5 * (n_eff ** 2)  # adjustable scale factor


def einstein_A_au(omega_au: float, dipole_au: float) -> float:
    """
    Einstein A in atomic units: A = (4/3) * (omega^3 / c^3) * |d|^2
    """
    if omega_au <= 0:
        return 0.0
    return (4.0 / 3.0) * (omega_au ** 3) * (dipole_au ** 2) / (C_AU ** 3)


def rate_bbr_absorption_s(a: RydbergState, b: RydbergState, T: float) -> float:
    """
    BBR-induced absorption rate a->b (s^-1), approximate:
      W_abs = A_ba * nbar(omega)
    where A_ba is spontaneous emission rate b->a (same matrix element).
    """
    omega = omega_transition_au(a, b)
    if omega <= 0:
        return 0.0
    d = dipole_r_au_hydrogenic_scale(a, b)
    A_ba_au = einstein_A_au(omega, d)  # b->a has same omega and d
    W_au = A_ba_au * nbar(omega, T)
    return W_au / AU_TIME_S


def spontaneous_A_s(a: RydbergState, b: RydbergState) -> float:
    """
    Approximate spontaneous emission rate a->b (s^-1) using the same A formula.
    """
    omega = omega_transition_au(b, a)  # emission a->b has omega = Ea-Eb = -(Eb-Ea)
    omega = -omega
    if omega <= 0:
        return 0.0
    d = dipole_r_au_hydrogenic_scale(a, b)
    A_au = einstein_A_au(omega, d)
    return A_au / AU_TIME_S


def tau_spontaneous_guess_s(state: RydbergState, tau_ref_s: float = 30e-6, neff_ref: float = 30.0) -> float:
    """
    Very rough spontaneous lifetime scaling ~ n_eff^3, with a reference point.
    Tune tau_ref_s to match your known lifetime for Cs P states.
    """
    ne = state.neff()
    return tau_ref_s * (ne / neff_ref) ** 3


def total_detected_rate(
    n0: int,
    L0: str,
    T: float,
    E_extract_v_per_cm: float,
    n_mix_span: int = 10,
    t1_s: float = 0.0,
    t2_s: float = 2.0e-6,
    dt_s: float = 2.0e-9,   # 2 ns default; you can loosen to 5–10 ns for speed
    include_backtransfer: bool = False,  # optional; forward-only is often stable/ok for a first pass
) -> Tuple[float, Dict[str, float]]:
    """
    Paper-style mixing:
      - Solve rate equations for populations N_i(t), N_j(t) (Euler time stepping)
      - Ion signal is integral over the detection window of sum_s N_s(t)*W_ion(s)
      - Mixing is NOT added as a decay channel for the initial state's lifetime.

    Returns:
      W_detected = ions/(t2-t1) in s^-1 and a breakdown dict.

    Notes:
      - Uses your existing W_BBR from Eq.(27) and W_SFI from Eq.(30)
      - Transfer rates W_{a->b} are still based on the approximate functions in this script
        (rate_bbr_absorption_s / spontaneous_A_s). That part is the main physics-limiting step.
    """
    if t2_s <= t1_s:
        raise ValueError("t2_s must be > t1_s")
    if dt_s <= 0:
        raise ValueError("dt_s must be > 0")

    init = RydbergState(n0, L0)

    # Ionization-like contributions for each state used in detection integrand
    def Wion(state: RydbergState) -> float:
        return w_bbr_eq27(state.n, state.L, T) + w_sfi_eq30(state.n, state.L, T, E_extract_v_per_cm)

    # "True loss" (used only in population decay): spontaneous + direct photoionization + (optional SFI term)
    # In the paper, redistribution is not a loss. Loss = processes that remove Rydberg population.
    def Gamma_loss(state: RydbergState) -> float:
        tau_sp = tau_spontaneous_guess_s(state)
        W_sp = 1.0 / tau_sp
        # You may choose whether W_SFI should be treated as loss before extraction;
        # keeping it in loss is a reasonable approximation if the field acts during the window.
        return W_sp + w_bbr_eq27(state.n, state.L, T) + w_sfi_eq30(state.n, state.L, T, E_extract_v_per_cm)

    # Build neighbor set for mixing: n±span with L'=L±1 (restricted to S,P,D)
    coupled: List[RydbergState] = []
    Lnum0 = LVAL[L0]
    for dn in range(-n_mix_span, n_mix_span + 1):
        if dn == 0:
            continue
        n = n0 + dn
        if n <= 1:
            continue
        for dL in (-1, +1):
            Ln = Lnum0 + dL
            if Ln < 0 or Ln > 2:  # keep S,P,D only
                continue
            coupled.append(RydbergState(n, LNAME[Ln]))

    # Transfer rates between states (approx): BBR absorption + spontaneous emission (if downhill)
    # We'll compute i->j, and optionally j->i if include_backtransfer=True
    Wij: Dict[RydbergState, float] = {}
    Wji: Dict[RydbergState, float] = {}

    for st in coupled:
        # i -> j: BBR absorption (up) + spontaneous emission (down, if applicable)
        W = rate_bbr_absorption_s(init, st, T)
        A = 0.0
        if omega_transition_au(st, init) > 0:  # st lower than init => emission possible init->st
            A = spontaneous_A_s(init, st)
        Wij[st] = W + A

        if include_backtransfer:
            # j -> i similarly
            Wb = rate_bbr_absorption_s(st, init, T)
            Ab = 0.0
            if omega_transition_au(init, st) > 0:  # init lower than st => emission possible st->init
                Ab = spontaneous_A_s(st, init)
            Wji[st] = Wb + Ab
        else:
            Wji[st] = 0.0

    # Initial population
    Ni = 1.0
    Nj = {st: 0.0 for st in coupled}

    # Precompute state-specific rates for speed
    Gamma_i = Gamma_loss(init)
    Wion_i = Wion(init)

    Gamma_j = {st: Gamma_loss(st) for st in coupled}
    Wion_j = {st: Wion(st) for st in coupled}

    # Time stepping
    t = 0.0
    ions = 0.0
    ions_direct = 0.0
    ions_mix = 0.0

    sum_Wij = sum(Wij.values())

    # Run long enough to cover detection window; you can start at 0 to match paper-style treatment
    t_end = t2_s
    n_steps = int(math.ceil(t_end / dt_s))

    for _ in range(n_steps):
        # accumulate ions in detection window using current populations
        if t1_s <= t <= t2_s:
            direct_rate = Ni * Wion_i
            mix_rate = sum(Nj[st] * Wion_j[st] for st in coupled)
            ions_direct += direct_rate * dt_s
            ions_mix += mix_rate * dt_s
            ions += (direct_rate + mix_rate) * dt_s

        # Euler update: populations
        # dNi/dt = -Gamma_i*Ni - Ni*sum_j Wij + sum_j (Nj*Wji)
        back_into_i = sum(Nj[st] * Wji[st] for st in coupled) if include_backtransfer else 0.0
        dNi = (-Gamma_i * Ni) - (Ni * sum_Wij) + back_into_i

        # dNj/dt = +Ni*Wij - Gamma_j*Nj - Nj*Wji (if backtransfer enabled)
        new_Nj = {}
        for st in coupled:
            dNj = (Ni * Wij[st]) - (Gamma_j[st] * Nj[st]) - (Nj[st] * Wji[st])
            new_Nj[st] = Nj[st] + dNj * dt_s

        Ni = Ni + dNi * dt_s
        Nj = new_Nj

        # clamp tiny negatives from Euler
        if Ni < 0.0:
            Ni = 0.0
        for st in coupled:
            if Nj[st] < 0.0:
                Nj[st] = 0.0

        t += dt_s

    W_detected = ions / (t2_s - t1_s)
    W_detected_direct = ions_direct / (t2_s - t1_s)
    W_detected_mix = ions_mix / (t2_s - t1_s)

    breakdown = {
        "W_detected_total": W_detected,
        "W_detected_direct": W_detected_direct,
        "W_detected_mix": W_detected_mix,
        "Gamma_i_loss": Gamma_i,         # this is what actually shortens Ni in this model
        "W_BBR_direct": w_bbr_eq27(n0, L0, T),
        "W_SFI_direct": w_sfi_eq30(n0, L0, T, E_extract_v_per_cm),
        "sum_Wij_mix": sum_Wij,          # redistribution strength (NOT a loss term in paper sense)
        "dt_s": dt_s,
        "n_neighbors": float(len(coupled)),
    }
    return W_detected, breakdown


# ----------------------------
# Main: Cs 60P defaults (Fig. 13-style)
# ----------------------------
def main(n0, T, E,):
    # n0, L0 = 50, "P"
    L0 = "P"
    # T = 10

    # Fig. 13 in the paper shows Cs total rates for extraction pulses E=5 and 10 V/cm.
    # Detection window: Cs discussion uses a short window; set defaults here (edit as needed).
    t1, t2 = 0.0e-6, 10e-6
    # E = 0.02
    # for E in (0, 5.0):
    # for t2 in [0.1e-6]:
    Wtot, bd = total_detected_rate(
        n0=n0,
        L0=L0,
        T=T,
        E_extract_v_per_cm=E,
        n_mix_span=3,
        t1_s=t1,
        t2_s=t2,
    )
    print(f"\nCs {n0}{L0} @ T={T:.1f} K, extraction E={E:.1f} V/cm, window [{t1*1e6:.2f},{t2*1e6:.2f}] us")
    print(f"  W_tot = {Wtot:.6g} s^-1   (1/W_tot = {1.0/Wtot if Wtot>0 else float('inf'):.6g} s)")
    print("  breakdown:")
    for k, v in bd.items():
        if k.endswith("_s"):
            print(f"    {k:>14s} = {v:.6g} us")
        else:
            print(f"    {k:>14s} = {v:.6g} s^-1")
    return Wtot, bd

if __name__ == "__main__":

    fig, ax = plt.subplots()

    ns = range(15,75,5)
    Ts = [10, 100, 300,1000]
    for T in Ts:
        pd_rate = []
        l_rate = []
        for n in ns:
            wtot, bd = main(n0=n,T=T,E=10)
            # pd_rate.append(bd['W_BBR_direct']+bd['W_SFI_direct']+bd['W_detected_mix'])
            pd_rate.append(bd['sum_Wij_mix'])
        ax.plot(ns, pd_rate, label="T={} K".format(T))
        # ax.plot(ns, l_rate, label="$\tau$ "+ "T={} K".format(T))
    ax.set_xlabel('n', fontsize=16)
    ax.set_yscale('log')
    # ax.set_ylim([1e-4,5e4])
    ax.set_ylabel('photoionization rate ($s^{-1}$ )', fontsize=16)
    ax.grid()
    ax.set_title('mixing rate')
    ax.legend(fontsize=14)
    ax.tick_params(labelsize=13)
    fig.tight_layout()
    fig.savefig("mixing_rate.pdf")
    # plt.plot(ns, l_rate, label="L")
    plt.show()