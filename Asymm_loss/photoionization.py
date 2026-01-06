import numpy as np
from scipy import constants
from scipy.integrate import simpson
from scipy.special import coulombf

from arc import Caesium  # ARC alkali atom class

# -----------------------------
# Physical constants (SI)
# -----------------------------
h = constants.h
hbar = constants.hbar
c = constants.c
e = constants.e
me = constants.m_e
eps0 = constants.epsilon_0

a0 = constants.physical_constants["Bohr radius"][0]      # m
Eh = constants.physical_constants["Hartree energy"][0]   # J
barn = 1e-28                                             # m^2

# -----------------------------
# Helpers
# -----------------------------
def omega_si(wavelength_m: float) -> float:
    return 2.0 * np.pi * c / wavelength_m

def photon_energy_au(wavelength_m: float) -> float:
    """Photon energy in atomic units (Hartree)."""
    return (h * c / wavelength_m) / Eh

def energy_ev_to_au(E_ev: float) -> float:
    """Convert eV to atomic units (Hartree)."""
    return (E_ev * constants.e) / Eh

def continuum_energy_from_bound_and_photon(E_bound_au: float, E_ph_au: float) -> float:
    """
    ARC getEnergy gives E_bound < 0 (relative to ionization threshold).
    Absorb photon: eps = E_ph + E_bound.
    """
    return E_ph_au + E_bound_au

def u_from_R(r_au, R):
    """Convert ARC radial R(r) to u(r)=r*R(r). r in a0."""
    return r_au * R

def normalize_bound_u(r_au, u):
    """Bound-state normalization: ∫|u|^2 dr = 1 in a.u."""
    norm = np.sqrt(simpson(u*u, r_au))
    return u / norm

def coulomb_energy_normalization_scale(r_au, u_raw, eps_au, l, r_match_min=2000.0):
    """
    ARC radialWavefunction for positive energy returns an unnormalized scattering solution.
    We rescale it to the standard *energy-normalized* Coulomb function:
        u_eps,l(r) -> sqrt(2/(pi*k)) * F_l(eta, k r)
    by matching amplitudes in the asymptotic region.

    This is a practical normalization fix for photoionization cross sections.
    """
    k = np.sqrt(2.0 * eps_au)      # a.u.
    eta = -1.0 / k                 # Z=1 attractive Coulomb for alkali valence e-

    rho = k * r_au
    F, _Fp = coulombf(l, eta, rho)
    u_target = np.sqrt(2.0 / (np.pi * k)) * F

    # choose matching region at large r
    mask = r_au >= r_match_min
    if np.count_nonzero(mask) < 50:
        raise ValueError("Not enough points in asymptotic region. Increase outerLimit or lower r_match_min.")

    # least-squares amplitude match: u_raw * A ≈ u_target
    A = np.dot(u_raw[mask], u_target[mask]) / np.dot(u_raw[mask], u_raw[mask])
    return u_raw * A

def radial_M_velocity_form(r_au, u_bound, u_cont, l, lp):
    """
    Velocity form radial integral like Cardman et al. (their Eq. 4 structure):
    M = ∫ u_cont(r) [ d/dr  ∓ (l_>/r) ] u_bound(r) dr
    where upper sign (minus) if l_> = l' (i.e. lp=l+1), lower sign (plus) if l_>=l (lp=l-1).
    """
    dr = r_au[1] - r_au[0]
    du_dr = np.gradient(u_bound, dr)
    lgt = max(l, lp)

    if lgt == lp:  # lp = l+1
        integrand = u_cont * (du_dr - u_bound * (lgt / r_au))
    else:          # lp = l-1
        integrand = u_cont * (du_dr + u_bound * (lgt / r_au))

    return simpson(integrand, r_au)

def sigma_velocity_form_SI(M_au, omega):
    """
    Same prefactor form as in the earlier code:
      σ ∝ (π e^2 ħ^2)/(3 ε0 m_e^2 ω c) * |M|^2
    The (l_>/(2l+1)) channel weight is applied outside.
    """
    pref = (np.pi * e**2 * hbar**2) / (3.0 * eps0 * me**2 * omega * c)
    # M_au is dimensionless in this formulation after energy-normalization; no extra unit fix here.
    return pref * (np.abs(M_au)**2)

def ionization_rate(I_W_m2, sigma_m2, wavelength_nm):
    """Γ = I σ /(ħ ω)"""
    lam = wavelength_nm * 1e-9
    return I_W_m2 * sigma_m2 / (hbar * omega_si(lam))

# -----------------------------
# Main computation
# -----------------------------
def cs_photoionization_sigma_arc(
    n=60, l=1, j=1.5, wavelength_nm=319.0,
    innerLimit_au=1e-4, outerLimit_au=3e4, step_au=0.02,
    r_match_min_au=4000.0,
):
    """
    Computes PI cross section for Cs |n l j> at wavelength_nm using ARC wavefunctions.

    Notes:
    - bound wavefunction: Numerov from ARC via radialWavefunction at E<0
    - continuum wavefunction: Numerov from ARC via radialWavefunction at E>0,
      then rescaled to energy-normalized Coulomb amplitude by asymptotic matching.
    """
    atom = Caesium()

    # Quantum defect (for your own inspection / reporting)
    delta = atom.getQuantumDefect(n, l, j)  # ARC call  [oai_citation:4‡ARC - Alkali Rydberg Calculator](https://arc-alkali-rydberg-calculator.readthedocs.io/en/latest/generated/arc.alkali_atom_functions.AlkaliAtom.getQuantumDefect.html?utm_source=chatgpt.com)

    # State energy relative to ionization threshold in eV (negative)
    E_bound_ev = atom.getEnergy(n, l, j)    # ARC call  [oai_citation:5‡ARC - Alkali Rydberg Calculator](https://arc-alkali-rydberg-calculator.readthedocs.io/en/latest/generated/arc.alkali_atom_functions.AlkaliAtom.getEnergy.html?utm_source=chatgpt.com)
    E_bound_au = energy_ev_to_au(E_bound_ev)

    # Photon energy and continuum kinetic energy (a.u.)
    lam_m = wavelength_nm * 1e-9
    E_ph_au = photon_energy_au(lam_m)
    eps_au = continuum_energy_from_bound_and_photon(E_bound_au, E_ph_au)
    if eps_au <= 0:
        raise ValueError("Photon is below threshold for ionization from this Rydberg state.")

    # ARC radialWavefunction signature:
    # radialWavefunction(l, s, j, stateEnergy, innerLimit, outerLimit, step)  [oai_citation:6‡ARC - Alkali Rydberg Calculator](https://arc-alkali-rydberg-calculator.readthedocs.io/en/latest/generated/arc.alkali_atom_functions.AlkaliAtom.radialWavefunction.html?utm_source=chatgpt.com)
    s = 0.5

    # ---- Bound state wavefunction
    r_b, R_b, idx_b = atom.radialWavefunction(
        l=l, s=s, j=j,
        stateEnergy=E_bound_au,
        innerLimit=innerLimit_au, outerLimit=outerLimit_au, step=step_au
    )
    r_b = np.array(r_b, dtype=float)
    R_b = np.array(R_b, dtype=float)
    u_b = normalize_bound_u(r_b, u_from_R(r_b, R_b))

    # ---- Continuum: sum over dipole-allowed partial waves l' = l±1
    omega = omega_si(lam_m)

    channels = []
    sigma_total = 0.0

    for lp in [l - 1, l + 1]:
        if lp < 0:
            continue

        # Continuum wavefunction from ARC at +eps
        r_c, R_c, idx_c = atom.radialWavefunction(
            l=lp, s=s, j=float(lp) + 0.5 if j > l else float(lp) - 0.5,  # simple j choice; see note below
            stateEnergy=eps_au,
            innerLimit=innerLimit_au, outerLimit=outerLimit_au, step=step_au
        )
        r_c = np.array(r_c, dtype=float)
        R_c = np.array(R_c, dtype=float)
        u_c_raw = u_from_R(r_c, R_c)

        # Interpolate bound u to the continuum grid if needed
        if (len(r_c) != len(r_b)) or (np.max(np.abs(r_c - r_b)) > 1e-12):
            u_b_on_c = np.interp(r_c, r_b, u_b, left=0.0, right=0.0)
        else:
            u_b_on_c = u_b

        # Energy-normalize continuum by matching to Coulomb asymptotic amplitude
        u_c = coulomb_energy_normalization_scale(
            r_c, u_c_raw, eps_au, lp, r_match_min=r_match_min_au
        )

        # Velocity-form matrix element and channel cross section
        M = radial_M_velocity_form(r_c, u_b_on_c, u_c, l, lp)

        weight = max(l, lp) / (2 * l + 1)  # l_>/(2l+1)
        sigma_lp = weight * sigma_velocity_form_SI(M, omega)

        channels.append((lp, sigma_lp))
        sigma_total += sigma_lp

    return {
        "n": n, "l": l, "j": j,
        "wavelength_nm": wavelength_nm,
        "quantum_defect": delta,
        "E_bound_ev": E_bound_ev,
        "eps_au": eps_au,
        "channels": [{"lprime": lp, "sigma_m2": sig, "sigma_barn": sig / barn} for lp, sig in channels],
        "sigma_total_m2": sigma_total,
        "sigma_total_barn": sigma_total / barn,
    }


if __name__ == "__main__":
    res = cs_photoionization_sigma_arc(n=60, l=1, j=1.5, wavelength_nm=319.0)
    print(f"Cs {res['n']}P3/2 @ {res['wavelength_nm']} nm")
    print(f"Quantum defect δ = {res['quantum_defect']:.6f}")
    print(f"Bound energy = {res['E_bound_ev']:.6e} eV")
    print(f"Continuum eps (a.u.) = {res['eps_au']:.6e}")
    for ch in res["channels"]:
        print(f"  l'={ch['lprime']}: σ = {ch['sigma_barn']:.4g} barn")
    print(f"Total σ = {res['sigma_total_barn']:.4g} barn")

    # Example ionization rate at intensity I (W/m^2)
    I = 1e6
    Gamma = ionization_rate(I, res["sigma_total_m2"], res["wavelength_nm"])
    print(f"Γ(I={I:.1e} W/m^2) = {Gamma:.3e} s^-1")