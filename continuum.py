import numpy as np
from scipy import integrate
from arc import *
import matplotlib.pyplot as plt
# =========================
# Rb effective potential (same as bound-state code)
# =========================
atom = Rubidium87()
RB_Z = atom.Z
RB_alphaC = atom.alphaC #9.0760

RB_a1 = atom.a1 #np.array([3.69628474, 4.44088978, 3.78717363, 2.39848933])
RB_a2 = atom.a2 #np.array([1.64915255, 1.92828831, 1.57027864, 1.76810544])
RB_a3 = atom.a3 #np.array([-9.86069196, -16.79597770, -11.65588970, -12.07106780])
RB_a4 = atom.a4 #np.array([0.19579987, -0.8163314, 0.52942835, 0.77256589])
RB_rc = atom.rc# np.array([1.66242117, 1.50195124, 4.86851938, 4.79831327])

def V_marinescu_rb(r, l):
    l_use = min(max(l, 0), 3)
    a1, a2, a3, a4, rc = (
        RB_a1[l_use], RB_a2[l_use], RB_a3[l_use],
        RB_a4[l_use], RB_rc[l_use]
    )
    Zeff = 1.0 + (RB_Z - 1.0)*np.exp(-a1*r) - r*(a3 + a4*r)*np.exp(-a2*r)
    Vc = -Zeff/r
    Vpol = -(RB_alphaC/(2.0*r**4))*(1.0 - np.exp(-(r/rc)**6))
    return Vc + Vpol

def V_eff(r, l):
    return V_marinescu_rb(r, l) + l*(l+1)/(2.0*r**2)

# =========================
# Numerov outward integrator (same core as bound state)
# =========================
def numerov_outward(r, k2, l, rescale_every=500):
    h = r[1] - r[0]
    h2 = h*h
    u = np.zeros_like(r)

    # regular solution near origin
    u[0] = r[0]**(l+1)
    u[1] = r[1]**(l+1)

    for i in range(1, len(r)-1):
        u[i+1] = (
            (2*(1 - 5*h2*k2[i]/12)*u[i]
             - (1 + h2*k2[i-1]/12)*u[i-1])
            / (1 + h2*k2[i+1]/12)
        )

        if i % rescale_every == 0:
            m = max(abs(u[i]), abs(u[i+1]), 1e-300)
            u[:i+2] /= m

    return u

# =========================
# Continuum wavefunction solver
# =========================
def continuum_state_rb(
    E, l,
    r_min=1e-2,
    r_max=20000.0,
    h=0.02,
    fit_window=(15000, 19000)
):
    """
    Returns energy-normalized continuum radial function u_E,l(r)
    """

    r = np.arange(r_min, r_max+h, h)
    Vef = V_eff(r, l)

    k2 = 2.0*(E - Vef)
    u = numerov_outward(r, k2, l)

    # --- asymptotic normalization ---
    k = np.sqrt(2.0*E)

    i0 = int(fit_window[0] / h)
    i1 = int(fit_window[1] / h)

    rr = r[i0:i1]
    uu = u[i0:i1]

    # Fit to A*sin(kr + phi)
    S = np.sin(k*rr)
    C = np.cos(k*rr)

    A = np.vstack([S, C]).T
    coeff, _, _, _ = np.linalg.lstsq(A, uu, rcond=None)
    Asin, Acos = coeff

    amplitude = np.sqrt(Asin**2 + Acos**2)

    # Energy-normalized amplitude
    norm_target = np.sqrt(2.0/(np.pi*k))

    u *= (norm_target / amplitude)

    return r, u

def check_continuum_energy_normalization(r, u, E, fit_window):
    """
    Check energy normalization of a continuum wavefunction u(r).

    Parameters
    ----------
    r : ndarray
        Radial grid (a0)
    u : ndarray
        Radial wavefunction u(r)
    E : float
        Continuum energy (Hartree)
    fit_window : tuple
        (r_min, r_max) region where asymptotic form holds

    Returns
    -------
    dict with fitted amplitude, expected amplitude, relative error
    """

    k = np.sqrt(2.0 * E)

    # select fitting region
    mask = (r >= fit_window[0]) & (r <= fit_window[1])
    rr = r[mask]
    uu = u[mask]

    # fit u = A sin(kr) + B cos(kr)
    S = np.sin(k * rr)
    C = np.cos(k * rr)
    M = np.vstack([S, C]).T

    coeff, _, _, _ = np.linalg.lstsq(M, uu, rcond=None)
    Asin, Acos = coeff

    A_fit = np.sqrt(Asin**2 + Acos**2)
    A_expected = np.sqrt(2.0 / (np.pi * k))

    return {
        "A_fit": A_fit,
        "A_expected": A_expected,
        "relative_error": (A_fit - A_expected) / A_expected
    }

if __name__ == "__main__":
    # Example parameters
    E = 1.10  # Hartree
    l = 1  # D-wave
    r, u = continuum_state_rb(E, l, r_min=1e-3, r_max=1200, fit_window=(800,1000))
    plt.plot(r,u)
    plt.xlim([0,200])
    plt.show()
    # Check normalization
    result = check_continuum_energy_normalization(
        r, u, E,
        fit_window=(800, 1000)
    )

    print("Fitted amplitude      =", result["A_fit"])
    print("Expected amplitude    =", result["A_expected"])
    print("Relative error        =", result["relative_error"])