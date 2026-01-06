import numpy as np
from dataclasses import dataclass
from scipy import integrate, constants, special
import mpmath as mp
from boudstate_gpt_new import solve_bound_state_nodegated
from arc import *
import sys

# =========================
# Units / constants
# =========================
a0 = constants.physical_constants["Bohr radius"][0]          # m
Eh = constants.physical_constants["Hartree energy"][0]       # J
alpha_fs = constants.alpha
J_to_au = 1.0 / Eh
a0_2_to_m2 = a0**2

# =========================
# Cs Marinescu model potential (ARC-style)
# (same functional form as used widely with Raithel group work)
# V(r) = -Zeff(r)/r - alphaC/(2 r^4)*(1-exp(-(r/rc)^6))
# Zeff(r)=1+(Z-1)e^{-a1 r}-r(a3+a4 r)e^{-a2 r}
# r in a0, V in Hartree
# =========================
atom = Rubidium87()
a_Z = atom.Z
a_alphaC = atom.alphaC #9.0760

a_a1 = atom.a1 #np.array([3.69628474, 4.44088978, 3.78717363, 2.39848933])
a_a2 = atom.a2 #np.array([1.64915255, 1.92828831, 1.57027864, 1.76810544])
a_a3 = atom.a3 #np.array([-9.86069196, -16.79597770, -11.65588970, -12.07106780])
a_a4 = atom.a4 #np.array([0.19579987, -0.8163314, 0.52942835, 0.77256589])
a_rc = atom.rc# np.array([1.66242117, 1.50195124, 4.86851938, 4.79831327])


def V_marinescu_cs(r, l):
    l_use = min(max(l, 0), 3)
    a1, a2, a3, a4, rc = a_a1[l_use], a_a2[l_use], a_a3[l_use], a_a4[l_use], a_rc[l_use]
    Zeff = 1.0 + (a_Z - 1.0)*np.exp(-a1*r) - r*(a3 + a4*r)*np.exp(-a2*r)
    Vc = -Zeff/r
    Vpol = -(a_alphaC/(2.0*r**4))*(1.0 - np.exp(-(r/rc)**6))
    return Vc + Vpol

def V_eff(r, l):
    return V_marinescu_cs(r, l) + l*(l+1)/(2.0*r**2)

# =========================
# Numerov (outward/inward) with rescaling
# Solve: u'' = 2*(V_eff - E)*u  (a.u.)
# Convert to u'' + k2(r) u = 0 with k2 = -2*(V_eff - E) = 2*(E - V_eff)
# =========================
def numerov_outward(r, k2, l, rescale_every=400):
    h = r[1] - r[0]
    h2 = h*h
    u = np.zeros_like(r, dtype=np.float64)

    # near origin: u ~ r^(l+1)
    u[0] = r[0]**(l+1)
    u[1] = r[1]**(l+1)

    for i in range(1, len(r)-1):
        u[i+1] = ((2.0*(1.0 - 5.0*h2*k2[i]/12.0)*u[i]) - (1.0 + h2*k2[i-1]/12.0)*u[i-1]) / (1.0 + h2*k2[i+1]/12.0)
        if (i % rescale_every) == 0:
            m = max(abs(u[i]), abs(u[i+1]), 1e-300)
            if m > 1e150:
                u[:i+2] /= m
            elif m < 1e-150:
                u[:i+2] /= m
    return u

def numerov_inward_bound(r, k2, E, rescale_every=400):
    # bound asymptotic: u ~ exp(-kappa r), kappa = sqrt(-2E)
    h = r[1] - r[0]
    h2 = h*h
    u = np.zeros_like(r, dtype=np.float64)
    kappa = np.sqrt(-2.0*E)

    u[-1] = 1e-200
    u[-2] = u[-1]*np.exp(kappa*h)  # moving inward increases

    for i in range(len(r)-2, 0, -1):
        u[i-1] = ((2.0*(1.0 - 5.0*h2*k2[i]/12.0)*u[i]) - (1.0 + h2*k2[i+1]/12.0)*u[i+1]) / (1.0 + h2*k2[i-1]/12.0)
        if (i % rescale_every) == 0:
            m = max(abs(u[i]), abs(u[i-1]), 1e-300)
            if m > 1e150:
                u[i-1:] /= m
            elif m < 1e-150:
                u[i-1:] /= m
    return u

# =========================
# Continuum radial function: integrate outward then normalize by matching to Coulomb F_l
# At large r, V -> -1/r, so Coulomb functions are appropriate.
# =========================
def continuum_state(
    E, l, r,
    fit_window=(15000, 19000)
):
    """
    Returns energy-normalized continuum radial function u_E,l(r)
    """
    Vef = V_eff(r, l)

    k2 = 2.0*(E - Vef)
    u = numerov_outward(r, k2, l)

    # --- asymptotic normalization ---
    k = np.sqrt(2.0*E)

    mask = (r >= fit_window[0]) & (r <= fit_window[1])
    rr = r[mask]
    uu = u[mask]

    S = np.sin(k * rr)
    C = np.cos(k * rr)
    A = np.vstack([S, C]).T
    Asin, Acos = np.linalg.lstsq(A, uu, rcond=None)[0]
    amplitude = np.sqrt(Asin ** 2 + Acos ** 2)

    norm_target = np.sqrt(2.0 / (np.pi * k))  # energy-normalized
    u *= (norm_target / amplitude)
    return u


# =========================
# Paper formula: shell-averaged sigma (Eq. 3) using M (Eq. 4), EDA velocity form
# σ̄ = (π e^2 ħ^2 /(3 ε0 m_e^2 ω c)) * [ l_>/(2l+1) ] * |M|^2 * (1/(E_H a0^2))
# with M = ∫ u_E,l'(r) [u'_n,l(r) ∓ u_n,l(r)/r * l_>] dr   (Eq. 4)
# upper sign: l_> = l' ; lower sign: l_> = l
# =========================
def radial_M_velocity(uE, ub, r, l, lp):
    # derivative of bound u
    dub = np.gradient(ub, r)
    lgt = max(l, lp)
    if lp == l + 1:
        # l_> = lp, upper sign in paper Eq.4
        integrand = uE * (dub - (lgt)*ub/r)
    elif lp == l - 1:
        # l_> = l, lower sign
        integrand = uE * (dub + (lgt)*ub/r)
    else:
        raise ValueError("E1 requires lp = l±1")
    return integrate.simpson(integrand, r)

def sigma_shell_avg_from_M(M_au, l, omega_SI, eps0=constants.epsilon_0):
    # Eq. (3) prefactor (SI), then convert matrix element units:
    # In the paper they include the conversion factor (1/(E_H a0^2)) for |M|^2.  [oai_citation:3‡2102.09622v2.pdf](sediment://file_00000000c82c71fd86dced0001b6a039)
    e = constants.e
    me = constants.m_e
    c = constants.c
    pref = (np.pi * e**2 * constants.hbar**2) / (3.0 * eps0 * me**2 * omega_SI * c)
    # l_> factor in Eq.3 is already included for a given channel; we apply the paper’s total-channel factor:
    # σ̄_{n,l}^{ε,l'} = pref * [l_>/(2l+1)] * |M|^2 * (1/(E_H a0^2))
    # We'll pass l_> separately when summing channels.
    conv = 1.0/(Eh * a0**2)
    return pref * (M_au**2) * conv

@dataclass
class PIResult:
    sigma_m2: float
    gamma_s: float
    E_bound_au: float
    E_free_au: float
    sigma_channels_m2: dict

def photoionization_pi_cs_rate(
    n, l, j,
    wavelength_m,
    intensity_W_m2,
    r_min=1e-3, r_max=120000.0, h=0.02,
):
    r = np.arange(r_min, r_max + h, h)

    # bound state
    defect = atom.getQuantumDefect(n=n, l=l, j=j)
    st = solve_bound_state_nodegated(
        n=n, l=l, j=j, delta0=defect,
        r_min=r_min, r_max=r_max, h=h
    )
    E_bound = st.E_au
    u_bound = st.u

    # photon / continuum energy
    nu = constants.c / wavelength_m
    omega_SI = 2.0*np.pi*nu
    Eph_J = constants.h*nu
    Eph_au = Eph_J * J_to_au

    E_free = Eph_au - abs(E_bound)
    if E_free <= 0:
        raise ValueError("Photon energy below threshold.")

    # continuum channels l' = l±1
    sigma_total = 0.0
    sigma_ch = {}

    for lp in [l-1, l+1]:
        if lp < 0:
            continue
        uE = continuum_state(E_free, lp, r, fit_window=(0.7*r_max, 0.9*r_max))

        M = radial_M_velocity(uE, u_bound, r, l, lp)  # atomic units (as in paper)
        lgt = max(l, lp)
        # Eq.3 channel: multiply sigma(M) by l_>/(2l+1)
        sigma_base = sigma_shell_avg_from_M(M, l, omega_SI)
        sigma_lp = sigma_base * (lgt/(2*l + 1))
        sigma_ch[lp] = sigma_lp
        sigma_total += sigma_lp

    # rate from photon flux: Γ = I σ /(ħ ω) (Eq. 9)  [oai_citation:4‡2102.09622v2.pdf](sediment://file_00000000c82c71fd86dced0001b6a039)
    gamma = intensity_W_m2 * sigma_total / (constants.hbar * omega_SI)

    return PIResult(
        sigma_m2=sigma_total,
        gamma_s=gamma,
        E_bound_au=E_bound,
        E_free_au=E_free,
        sigma_channels_m2=sigma_ch
    )

# =========================
# Example: your beam (peak intensity), n=50 and 60, Cs nP3/2, 319 nm
# =========================
if __name__ == "__main__":
    lam = 1064e-9
    P = 0.020
    w = 3e-6  # 1/e^2 radius
    I0 = 2*P/(np.pi*w*w)
    n =  int(sys.argv[1])
    # ns = np.arange(20,95, 5)
    # for n in ns:
    res = photoionization_pi_cs_rate(
        n=n, l=1, j=1.5,
        wavelength_m=lam,
        intensity_W_m2=I0,
        r_min=1e-3, r_max=3*n**2, h=1/2/n
    )
    print(f"Rb {n}P3/2 @ {lam} m")
    print(f"  E_bound = {res.E_bound_au:.6e} Ha")
    print(f"  E_free  = {res.E_free_au:.6e} Ha")
    print(f"  sigma   = {res.sigma_m2*1e28:.3e} barns   ({res.sigma_m2*1e4:.3e} cm^2)")
    print(f"  gamma   = {res.gamma_s:.3e} s^-1   tau={1/res.gamma_s*1e6:.2f} us")
    print(f"  channels: { {lp: s for lp,s in res.sigma_channels_m2.items()} }")
    print()