import numpy as np
from dataclasses import dataclass
from scipy import integrate, constants, optimize
from arc import *
import matplotlib.pyplot as plt
import sys
import os
# =========================
# Cs Marinescu/ARC model potential parameters (same form as before)
# r in a0, V in Hartree
# =========================


atom = Cesium()
atom_Z = atom.Z
atom_alphaC = atom.alphaC #9.0760

atom_a1 = atom.a1 #np.array([3.69628474, 4.44088978, 3.78717363, 2.39848933])
atom_a2 = atom.a2 #np.array([1.64915255, 1.92828831, 1.57027864, 1.76810544])
atom_a3 = atom.a3 #np.array([-9.86069196, -16.79597770, -11.65588970, -12.07106780])
atom_a4 = atom.a4 #np.array([0.19579987, -0.8163314, 0.52942835, 0.77256589])
atom_rc = atom.rc# np.array([1.66242117, 1.50195124, 4.86851938, 4.79831327])
alpha_fs = constants.alpha  # fine-structure constant

def arc_energy_to_Ha(atom, n, l, j):
    Eh_J = constants.physical_constants["Hartree energy"][0]
    E_eV = atom.getEnergy(n, l, j)
    return (E_eV * constants.e) / Eh_J


def V_marinescu(r, l):
    l_use = min(max(int(l), 0), 3)
    a1, a2, a3, a4, rc = atom_a1[l_use], atom_a2[l_use], atom_a3[l_use], atom_a4[l_use], atom_rc[l_use]
    Zeff = 1.0 + (atom_Z - 1.0) * np.exp(-a1 * r) - r * (a3 + a4 * r) * np.exp(-a2 * r)
    Vc = -Zeff/r
    Vpol = -(atom_alphaC / (2.0 * r ** 4)) * (1.0 - np.exp(-(r / rc) ** 6))
    return Vc + Vpol

def ls_expectation(l: int, j: float) -> float:
    """<L·S> for s=1/2 (atomic units, i.e. ħ=1)."""
    return 0.5 * (j*(j+1.0) - l*(l+1.0) - 0.75)

def V_so(r: np.ndarray, l: int, j: float) -> np.ndarray:
    """
    Breit–Pauli (Pauli) spin-orbit:
      V_so(r) = (alpha^2/2) * (1/r) * dV/dr * <L·S>
    where V is the CENTRAL potential V_marinescu_cs(r,l) (not including centrifugal).
    r in a0, V in Hartree.
    """

    V = V_marinescu(r, l)  # <-- your existing central potential
    dVdr = np.gradient(V, r, edge_order=2)
    rr = np.maximum(r, 1e-12)
    return 0.5 * (alpha_fs**2) * (dVdr / rr) * ls_expectation(l, j)

def V_eff(r, l,j):
    return V_marinescu(r, l) + l*(l + 1)/(2.0 * r ** 2) + V_so(r, l, j)

# =========================
# Numerov outward/inward (rescaled)
# Solve u'' + k^2(r) u = 0, with k^2 = 2(E - V_eff)
# =========================
def numerov_outward(r, k2, l, rescale_every=400):
    h = r[1] - r[0]
    h2 = h*h
    u = np.zeros_like(r, dtype=np.float64)
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
    h = r[1] - r[0]
    h2 = h*h
    u = np.zeros_like(r, dtype=np.float64)
    # local decay constant at r_max (more robust than constant kappa)
    kappa_end = np.sqrt(max(-k2[-1], 1e-300))
    u[-1] = 1e-200
    u[-2] = u[-1]*np.exp(kappa_end*h)
    for i in range(len(r)-2, 0, -1):
        u[i-1] = ((2.0*(1.0 - 5.0*h2*k2[i]/12.0)*u[i]) - (1.0 + h2*k2[i+1]/12.0)*u[i+1]) / (1.0 + h2*k2[i-1]/12.0)
        if (i % rescale_every) == 0:
            m = max(abs(u[i]), abs(u[i-1]), 1e-300)
            if m > 1e150:
                u[i-1:] /= m
            elif m < 1e-150:
                u[i-1:] /= m
    return u

def log_derivative_5pt(u, r, i):
    h = r[1] - r[0]
    if i < 2 or i > len(r)-3 or abs(u[i]) < 1e-220:
        return np.nan
    du = (-u[i+2] + 8*u[i+1] - 8*u[i-1] + u[i-2])/(12*h)
    return du/u[i]

def count_nodes(u):
    s = np.sign(u); s[s == 0] = 1
    return int(np.sum(s[1:]*s[:-1] < 0))

@dataclass
class BoundState:
    n: int
    l: int
    j: int
    E_au: float
    r: np.ndarray
    u: np.ndarray  # normalized: ∫ u^2 dr = 1

def mismatch_windowed(uo, ui, r, i_match, w=40):
    # RMS of (L_out - L_in) over i_match-w ... i_match+w
    i0 = max(2, i_match - w)
    i1 = min(len(r)-3, i_match + w)
    diffs = []
    for i in range(i0, i1+1):
        Lo = log_derivative_5pt(uo, r, i)
        Li = log_derivative_5pt(ui, r, i)
        if np.isfinite(Lo) and np.isfinite(Li):
            diffs.append(Lo - Li)
    if len(diffs) < 10:
        return np.nan
    diffs = np.array(diffs)
    return np.sqrt(np.mean(diffs*diffs))

def solve_bound_numerov_ground(
    n, l, j,
    r_min=1e-5, r_max=800.0, h=0.002,
    w_rms=40
):
    r = np.arange(r_min, r_max + h, h)
    # ARC energy seed (Hartree)
    E0 = arc_energy_to_Ha(atom, n, l, j)

    # choose whether you use SO or not:
    Vef = V_eff(r, l, j)  # or V_eff(r,l) if no SO
    i_turn = int(np.argmin(np.abs(Vef - E0)))
    r_turn = r[i_turn]
    # r_match = 5
    r_match = max(0.8 * r_turn, 5)
    print('r match:', r_match)
    i_match = int(np.argmin(np.abs(r - r_match)))
    i_match = max(2+w_rms, min(len(r)-3-w_rms, i_match))

    def rms_mismatch(E):
        k2 = 2.0*(E - Vef)
        uo = numerov_outward(r, k2, l)
        ui = numerov_inward_bound(r, k2, E)

        if abs(uo[i_match]) < 1e-220 or abs(ui[i_match]) < 1e-220:
            return 1e99

        ui *= (uo[i_match]/ui[i_match])
        return mismatch_windowed(uo, ui, r, i_match, w=w_rms)



    # minimize RMS in a tight bracket around E0
    bracket = (E0*1.10, E0*0.90)  # widen if needed
    res = optimize.minimize_scalar(rms_mismatch, bracket=bracket, method="brent")
    E_best = float(res.x)

    # build final wavefunction at E_best
    k2 = 2.0*(E_best - Vef)
    uo = numerov_outward(r, k2, l)
    ui = numerov_inward_bound(r, k2, E_best)
    ui *= (uo[i_match]/ui[i_match])

    # blend stitch to remove kink
    u = np.zeros_like(r)
    Nblend = 200
    i0 = max(0, i_match - Nblend)
    i1 = min(len(r)-1, i_match + Nblend)
    x = np.linspace(0.0, 1.0, i1-i0+1)
    w = 0.5 - 0.5*np.cos(np.pi*x)
    u[:i0] = uo[:i0]
    u[i1+1:] = ui[i1+1:]
    u[i0:i1+1] = (1-w)*uo[i0:i1+1] + w*ui[i0:i1+1]

    # normalize
    u /= np.sqrt(integrate.simpson(u*u, r))
    return BoundState(
        n=n, l=l, j=j,
        r=r,
        E_au=E_best,
        u=u,
    )

# =========================
# Dipole matrix element <6s|r|7p> (length gauge)
# For u=rR:  <a|r|b> = ∫ u_a(r) * r * u_b(r) dr
# =========================
def radial_dipole_length(ua, ub, r):
    return float(integrate.simpson(ua * r * ub, r))

if __name__ == "__main__":
    # Solve Cs 6S and 7P on the SAME grid for clean integration
    # Low-n requires finer h than Rydberg states; start with h=1e-3, r_max~600 a0

    r_min = 1e-6
    n =  int(sys.argv[1])
    r_max = max(100,8*n**2)
    h = min(0.001,1/8/n)
    print(f'h={h}, r_min={r_min}, r_max={r_max}')
    # folder = 'results'
    # os.makedirs(folder, exist_ok=True)


    s6 = solve_bound_numerov_ground(n=6, l=0, j=0.5, r_min=r_min, r_max=r_max, h=h)
    np12 = solve_bound_numerov_ground(n=n, l=1, j=0.5, r_min=r_min, r_max=r_max, h=h)

    # Radial integral in atomic units (a0), i.e. in units of ea0 for the dipole operator
    R12 = radial_dipole_length(s6.u, np12.u, s6.r)

    # Fine-structure reduced matrix elements (alkali S1/2 -> PJ)
    d_np12 = -np.sqrt(2.0/3.0) * R12  # in ea0

    fig, ax = plt.subplots(ncols=3)
    ax[0].plot(s6.r, s6.u)
    ax[1].plot(np12.r, np12.u)
    ax[0].set_xlim([0,50])
    ax[0].set_title('$6S_{1/2}$')
    ax[1].set_title(f'${n}S_{1/2}$')

    np32 = solve_bound_numerov_ground(n=n, l=1, j=1.5, r_min=r_min, r_max=r_max, h=h)

    # Radial integral in atomic units (a0), i.e. in units of ea0 for the dipole operator
    R32 = radial_dipole_length(s6.u, np32.u, s6.r)

    # Fine-structure reduced matrix elements (alkali S1/2 -> PJ)
    d_np32 = np.sqrt(4.0 / 3.0) * R32  # in ea0
    print("Cs bound energies (model) [Hartree]:")
    print(f"  E(6S) = {s6.E_au:.8e} Ha")
    print(f"  E({n}P1/2) = {np12.E_au:.8e} Ha")
    print(f"  E({n}P3/2) = {np32.E_au:.8e} Ha")
    print()
    print(f"Radial integral R = <6S|r|{n}P> (length gauge):")
    print(f"  R12 = {R12:.6f}  (in a0, i.e. ea0 for dipole)")
    print(f"  R32 = {R32:.6f}  (in a0, i.e. ea0 for dipole)")
    print()
    print("Reduced dipole matrix elements (fine structure):")
    print(f"  <6S1/2 || d || {n}P1/2> = {d_np12:.6f} ea0")
    print(f"  <6S1/2 || d || {n}P3/2> = {d_np32:.6f} ea0")

    # fig, ax = plt.subplots()
    # r = np.linspace(1.0, 100.0, 1000)
    # V = V_eff(r,l=1,j=1.5)
    # ax.plot(r,V)
    # E = arc_energy_to_Ha(atom, n=7, l=1, j=1.5)
    # ax.axhline(E, color='k')
    # ax.set_yscale('log')
    # fig.show()
    # # fig.savefig(os.path.join(folder,'Wavefunction.pdf'))