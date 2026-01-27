import numpy as np
from dataclasses import dataclass
from scipy import integrate, constants
from scipy.linalg import eigh_tridiagonal

from arc import *
import matplotlib.pyplot as plt
# =========================
# Atomic units conversions
# =========================
Eh = constants.physical_constants["Hartree energy"][0]   # J
a0 = constants.physical_constants["Bohr radius"][0]      # m
alpha_fs = constants.alpha  # fine-structure constant

atom = Cesium()
atom_Z = atom.Z
atom_alphaC = atom.alphaC #9.0760

atom_a1 = atom.a1 #np.array([3.69628474, 4.44088978, 3.78717363, 2.39848933])
atom_a2 = atom.a2 #np.array([1.64915255, 1.92828831, 1.57027864, 1.76810544])
atom_a3 = atom.a3 #np.array([-9.86069196, -16.79597770, -11.65588970, -12.07106780])
atom_a4 = atom.a4 #np.array([0.19579987, -0.8163314, 0.52942835, 0.77256589])
atom_rc = atom.rc# np.array([1.66242117, 1.50195124, 4.86851938, 4.79831327])

def V_marinescu(r, l):
    """Rb model potential V(r) in a.u. (Hartree). r in a0."""
    l_use = min(max(l, 0), 3)
    a1, a2, a3, a4, rc = atom_a1[l_use], atom_a2[l_use], atom_a3[l_use], atom_a4[l_use], atom_rc[l_use]
    Zeff = 1.0 + (atom_Z - 1.0)*np.exp(-a1*r) - r*(a3 + a4*r)*np.exp(-a2*r)
    Vc = -Zeff/r
    Vpol = -(atom_alphaC/(2.0*r**4))*(1.0 - np.exp(-(r/rc)**6))
    return Vc + Vpol

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

def ls_expectation(l: int, j: float) -> float:
    """<L·S> for s=1/2 (atomic units, i.e. ħ=1)."""
    return 0.5 * (j*(j+1.0) - l*(l+1.0) - 0.75)



def V_eff_so(r: np.ndarray, l: int, j: float) -> np.ndarray:
    """V_eff = V + l(l+1)/(2r^2) + V_so."""
    rr = np.maximum(r, 1e-12)
    Vcent = l*(l+1.0) / (2.0 * rr**2)
    return V_marinescu(r, l) + Vcent + V_so(r, l, j)


def V_eff(r, l):
    """Effective potential for u(r)=rR(r): V(r) + l(l+1)/(2r^2)."""
    return V_marinescu(r, l) + l*(l+1)/(2.0*r**2)


@dataclass
class BoundStateFD:
    n: int
    l: int
    j: float
    E_au: float
    r_a0: np.ndarray
    u: np.ndarray   # normalized so ∫ u^2 dr = 1

def solve_bound_state_fd(
    n: int, l: int, j: float,
    E_target_au: float,
    r_min: float = 1e-5,
    r_max: float = 800.0,
    h: float = 2e-3,
):
    """
    Finite-difference (matrix) solver for bound states:
      - stable for low-n / ground states
      - returns eigenstate closest to E_target_au

    Uses u(r)=rR(r), and solves:
      [-1/2 d^2/dr^2 + V_eff(r,l)] u = E u
    with u(r_min)=u(r_max)=0 boundary conditions.
    """

    # grid
    r = np.arange(r_min, r_max + h, h)

    # interior points (Dirichlet boundaries)
    ri = r[1:-1]
    N = ri.size

    V = V_eff_so(ri, l, j)  # your existing V_eff

    # Tridiagonal kinetic operator for -1/2 d^2/dr^2 with spacing h:
    # main: 1/h^2, off: -1/(2 h^2)
    off = -0.5 / (h*h) * np.ones(N-1)
    diag = (1.0 / (h*h)) * np.ones(N) + V

    # Solve eigenvalues/vectors of tridiagonal matrix
    evals, evecs = eigh_tridiagonal(diag, off)

    # pick eigenvalue closest to target
    idx = int(np.argmin(np.abs(evals - E_target_au)))
    E = float(evals[idx])
    u_inner = evecs[:, idx]

    # enforce consistent sign (optional)
    if u_inner[np.argmax(np.abs(u_inner))] < 0:
        u_inner = -u_inner

    # reinsert boundaries
    u = np.zeros_like(r)
    u[1:-1] = u_inner

    # normalize: ∫ u^2 dr = 1
    u /= np.sqrt(integrate.simpson(u*u, r))

    return BoundStateFD(n=n, l=l, j=j, E_au=E, r_a0=r, u=u)

def radial_dipole_length(ua, ub, r):
    return float(integrate.simpson(ua * r * ub, r))

if __name__ == "__main__":
    ng, lg, jg = 6, 0, 0.5
    Eg_target_eV = atom.getEnergy(ng, lg, jg)          # ARC: energy in cm^-1 relative to ionization limit (usually negative)
    Eg_target_au = (Eg_target_eV * constants.e) /  constants.physical_constants["Hartree energy"][0]
    st_g = solve_bound_state_fd(
        n=ng, l=lg, j=jg,
        E_target_au=Eg_target_au,
        r_min=1e-5, r_max=200.0, h=0.004
    )
    print("E_target (au) =", Eg_target_au)
    print("E_found  (au) =", st_g.E_au)
    print("norm ∫u^2dr =", integrate.simpson(st_g.u*st_g.u, st_g.r_a0))

    ne, le, je = 7, 1, 0.5
    Ee_target_eV = atom.getEnergy(ne, le, je)          # ARC: energy in cm^-1 relative to ionization limit (usually negative)
    Ee_target_au = (Ee_target_eV * constants.e) /  constants.physical_constants["Hartree energy"][0]
    st_e = solve_bound_state_fd(
        n=ne, l=le, j=je,
        E_target_au=Ee_target_au,
        r_min=1e-5, r_max=200.0, h=0.004
    )
    print("E_target (au) =", Ee_target_au)
    print("E_found  (au) =", st_e.E_au)
    print("norm ∫u^2dr =", integrate.simpson(st_e.u*st_e.u, st_e.r_a0))
    fig, ax = plt.subplots(ncols=2)
    ax[0].plot(st_g.r_a0, st_g.u, label='ground state')
    ax[1].plot(st_e.r_a0, st_e.u, label='excited state')
    ax[0].set_xlim([0, 50])
    ax[1].set_xlim([0, 50])
    fig.show()
    R = radial_dipole_length(st_g.u, st_e.u, st_g.r_a0)
    d_7p12 = np.sqrt(2.0 / 3.0) * R
    print("Cs bound energies (model) [Hartree]:")
    print()
    print("Radial integral R = <6S|r|7P> (length gauge):")
    print(f"  R = {R:.6f}  (in a0, i.e. ea0 for dipole)")
    print()
    print("Reduced dipole matrix elements (fine structure):")
    print(f"  <6S1/2 || d || 7P1/2> = {d_7p12:.6f} ea0")