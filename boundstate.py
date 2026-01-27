import numpy as np
from dataclasses import dataclass
from scipy import integrate, constants

from arc import *
import matplotlib.pyplot as plt
# =========================
# Atomic units conversions
# =========================
Eh = constants.physical_constants["Hartree energy"][0]   # J
a0 = constants.physical_constants["Bohr radius"][0]      # m
alpha_fs = constants.alpha  # fine-structure constant
# =========================
# Rb Marinescu model potential (ARC Rubidium85 parameters)
# r in a0, V in Hartree (atomic units)
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
    """Rb model potential V(r) in a.u. (Hartree). r in a0."""
    l_use = min(max(l, 0), 3)
    a1, a2, a3, a4, rc = RB_a1[l_use], RB_a2[l_use], RB_a3[l_use], RB_a4[l_use], RB_rc[l_use]
    Zeff = 1.0 + (RB_Z - 1.0)*np.exp(-a1*r) - r*(a3 + a4*r)*np.exp(-a2*r)
    Vc = -Zeff/r
    Vpol = -(RB_alphaC/(2.0*r**4))*(1.0 - np.exp(-(r/rc)**6))
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

    V = V_marinescu_rb(r, l)  # <-- your existing central potential
    dVdr = np.gradient(V, r, edge_order=2)
    rr = np.maximum(r, 1e-12)
    return 0.5 * (alpha_fs**2) * (dVdr / rr) * ls_expectation(l, j)


def V_eff(r, l, j):
    """Effective potential for u(r)=rR(r): V(r) + l(l+1)/(2r^2)."""
    return V_marinescu_rb(r, l) + l*(l+1)/(2.0*r**2)+ V_so(r, l, j)

# =========================
# Numerov integrators with rescaling
# Equation: u'' + k^2(r) u = 0, k^2 = 2(E - V_eff)
# =========================
def numerov_outward(r, k2, l, rescale_every=400):
    h = r[1] - r[0]
    h2 = h*h
    u = np.zeros_like(r, dtype=np.float64)

    # regular at origin: u ~ r^(l+1)
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

def numerov_inward_bound(r, k2, E, rescale_every=600):
    h = r[1] - r[0]
    h2 = h*h
    u = np.zeros_like(r, dtype=np.float64)

    # Local decay constant at r_max:
    # In forbidden region, k2 < 0 and kappa = sqrt(-k2)
    kappa_end = np.sqrt(max(-k2[-1], 1e-300))

    u[-1] = 1e-200
    u[-2] = u[-1] * np.exp(kappa_end*h)

    for i in range(len(r)-2, 0, -1):
        u[i-1] = ((2.0*(1.0 - 5.0*h2*k2[i]/12.0)*u[i]) - (1.0 + h2*k2[i+1]/12.0)*u[i+1]) / (1.0 + h2*k2[i-1]/12.0)

        if (i % rescale_every) == 0:
            m = max(abs(u[i]), abs(u[i-1]), 1e-300)
            if m > 1e150:
                u[i-1:] /= m
            elif m < 1e-150:
                u[i-1:] /= m
    return u

def log_derivative(u, r, i):
    du = (u[i+1]-u[i-1])/(r[i+1]-r[i-1])
    return du/u[i]

def log_derivative_5pt(u, r, i):
    # 5-point stencil for u' (O(h^4)) — much smoother than 3-point
    h = r[1] - r[0]
    if i < 2 or i > len(r)-3 or abs(u[i]) < 1e-220:
        return np.nan
    du = (-u[i+2] + 8*u[i+1] - 8*u[i-1] + u[i-2]) / (12*h)
    return du / u[i]


def count_nodes(u):
    s = np.sign(u)
    s[s == 0] = 1
    return int(np.sum(s[1:]*s[:-1] < 0))

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

def refine_energy_by_rms(E_lo, E_hi, l, r, Vef, i_match, w=40, n_grid=61):
    # scan energies and choose the one with minimal RMS mismatch
    Es = np.linspace(E_lo, E_hi, n_grid)
    best = (np.inf, None, None, None)
    for E in Es:
        k2 = 2.0*(E - Vef)
        uo = numerov_outward(r, k2, l)
        ui = numerov_inward_bound(r, k2, E)
        if abs(ui[i_match]) < 1e-220 or abs(uo[i_match]) < 1e-220:
            continue
        ui *= (uo[i_match]/ui[i_match])
        rms = mismatch_windowed(uo, ui, r, i_match, w=w)
        if np.isfinite(rms) and rms < best[0]:
            best = (rms, E, uo, ui)
    if best[1] is None:
        raise RuntimeError("RMS refinement failed; try moving i_match or increasing r_max.")
    return best[1], best[2], best[3]  # E_best, uo_best, ui_best

def pick_safe_match_index(u_out, i_center, window=3000):
    """
    Avoid matching at/near a node: pick the index near i_center where |u_out| is largest.
    """
    lo = max(10, i_center - window)
    hi = min(len(u_out) - 11, i_center + window)
    return lo + int(np.argmax(np.abs(u_out[lo:hi])))


# =========================
# Bound-state energy by log-derivative matching
# =========================
@dataclass
class BoundState:
    n: int
    l: int
    j: float
    E_au: float         # Hartree
    r_a0: np.ndarray
    u: np.ndarray       # normalized so ∫u^2 dr = 1
    R: np.ndarray       # R = u/r
    nodes: int
    norm_u: float
    norm_R: float


def energy_scan_node_window(n, l, delta0, Vef, r, i_turn,
                            frac_span=0.3, nE=500):
    """
    Scan energies around E0 to find the best contiguous window where node count == target_nodes.
    Returns (E_low, E_high) where outward solution has target_nodes.
    """
    target_nodes = n - l - 1
    n_eff = n - delta0
    E0 = -1.0/(2.0*n_eff**2)

    # scan from (1+frac_span)*E0 (more negative) to (1-frac_span)*E0 (less negative)
    # note E0 < 0, so "more negative" means multiply by >1
    E_grid = np.linspace((1.0 + frac_span)*E0, (1.0 - frac_span)*E0, nE)

    nodes = np.zeros_like(E_grid, dtype=int)
    # compute node count for each energy
    for i, E in enumerate(E_grid):
        k2 = 2.0*(E - Vef)
        uo = numerov_outward(r, k2, l)
        nodes[i] = count_nodes(uo)

    # find indices where node count matches target
    good = np.where(nodes == target_nodes)[0]
    if len(good) == 0:
        # Tell caller what we saw (useful for debugging)
        raise RuntimeError(
            f"No energies in scan produced target node count {target_nodes}. "
            f"Observed node range: {nodes.min()}..{nodes.max()}. "
            f"Try increasing frac_span, r_max, or reducing h."
        )

    # find the longest contiguous block in 'good'
    blocks = []
    start = good[0]
    prev = good[0]
    for idx in good[1:]:
        if idx == prev + 1:
            prev = idx
        else:
            blocks.append((start, prev))
            start = prev = idx
    blocks.append((start, prev))

    # choose the longest block
    bstart, bend = max(blocks, key=lambda b: b[1]-b[0])

    # convert to energy window (take neighbors to ensure we're inside the plateau)
    E_low = E_grid[bstart]
    E_high = E_grid[bend]

    # Ensure E_low < E_high in the usual sense? (remember negative numbers)
    # We'll just return as-is; the caller will handle ordering.
    return E_low, E_high


def choose_match_index_by_rms(uo, ui, r, i_turn, w_match=250, w_rms=40, k2=None):
    """
    Choose i_match near i_turn that minimizes windowed RMS log-derivative mismatch.
    Optionally restrict to classically-allowed side with k2>0 (recommended).
    """
    lo = max(2 + w_rms, i_turn - w_match)
    hi = min(len(r) - 3 - w_rms, i_turn + w_match)

    best = (np.inf, None)
    for i in range(lo, hi + 1):
        if abs(uo[i]) < 1e-12 or abs(ui[i]) < 1e-220:
            continue
        if k2 is not None:
            # strongly recommended: match in/near allowed region to avoid outward contamination
            if k2[i] < 0:
                continue

        scale = uo[i] / ui[i]
        ui_scaled = ui * scale
        rms = mismatch_windowed(uo, ui_scaled, r, i, w=w_rms)
        if np.isfinite(rms) and rms < best[0]:
            best = (rms, i)

    if best[1] is None:
        # fallback: old method if RMS selection fails
        return pick_safe_match_index(uo, i_turn, window=w_match)
    return best[1]

def solve_bound_state_nodegated(
    n: int,
    l: int,
    j: float,
    delta0: float,
    r_min: float = 1e-2,
    r_max: float = 300000.0,
    h: float = 0.01,
    max_iter: int = 120,
):
    """
    Robust bound-state solver:
      1) scan energy to find window where node count == target_nodes
      2) within that window, bracket log-derivative mismatch sign change
      3) bisection with safe match-point choice
    """
    if n <= l:
        raise ValueError("Need n > l.")

    r = np.arange(r_min, r_max + h, h)
    Vef = V_eff(r, l, j)
    target_nodes = n - l - 1

    # QDT energy guess for turning point location
    n_eff = n - delta0
    E0 = -1.0/(2.0*n_eff**2)
    i_turn = int(np.argmin(np.abs(Vef - E0)))
    i_turn = max(20, min(len(r)-21, i_turn))

    def mismatch_nodes(E):
        k2 = 2.0*(E - Vef)
        uo = numerov_outward(r, k2, l)
        nodes = count_nodes(uo)

        # match index near turning point but avoid nodes
        # i_match = pick_safe_match_index(uo, i_turn, window=300)

        ui = numerov_inward_bound(r, k2, E)

        i_match = choose_match_index_by_rms(
            uo, ui, r, i_turn,
            w_match=150,  # search ±250 points around turning point
            w_rms=40,  # RMS window for mismatch
            k2=k2  # restrict to k2>0 (allowed region)
        )


        if abs(ui[i_match]) < 1e-220 or abs(uo[i_match]) < 1e-220:
            return np.nan, nodes, uo, ui, i_match

        ui *= (uo[i_match]/ui[i_match])

        L_out = log_derivative_5pt(uo, r, i_match)
        L_in  = log_derivative_5pt(ui, r, i_match)
        return (L_out - L_in), nodes, uo, ui, i_match

    # --- Step 1: find an energy window where nodes are correct
    # start with moderate scan span; if fails, widen automatically
    for frac_span in [0.35, 0.60, 0.90]:
        try:
            E_a, E_b = energy_scan_node_window(n, l, delta0, Vef, r, i_turn,
                                               frac_span=frac_span, nE=1000)
            break
        except RuntimeError as e:
            last_err = e
            continue
    else:
        raise last_err

    # Ensure ordering
    E_lo, E_hi = (E_a, E_b) if E_a < E_b else (E_b, E_a)

    # --- Step 2: within [E_lo, E_hi], find a sign change of mismatch while keeping correct nodes
    # We'll sample mismatch at many points and find adjacent opposite signs.
    Es = np.linspace(E_lo, E_hi, 101)
    fs = []
    for E in Es:
        f, nodes, *_ = mismatch_nodes(E)
        if nodes != target_nodes or (not np.isfinite(f)):
            fs.append(np.nan)
        else:
            fs.append(f)
    fs = np.array(fs)

    # find adjacent indices with finite opposite sign
    bracket = None
    for i in range(len(Es)-1):
        if np.isfinite(fs[i]) and np.isfinite(fs[i+1]) and np.sign(fs[i]) != np.sign(fs[i+1]):
            bracket = (Es[i], Es[i+1], fs[i], fs[i+1])
            break

    if bracket is None:
        raise RuntimeError(
            "Found node-correct energy window, but couldn't bracket a mismatch sign change. "
            "Try increasing r_max (e.g. 400000) or reducing h (0.005), "
            "or increase the scan resolution (nE) inside energy_scan_node_window."
        )

    E_lo, E_hi, f_lo, f_hi = bracket

    # --- Step 3: bisection inside the bracket
    ui_best = None
    i_match_best = None
    Em_best = None

    for _ in range(max_iter):
        Em = 0.5*(E_lo + E_hi)
        f_m, nodes_m, uo, ui, i_match = mismatch_nodes(Em)

        # enforce node count
        if nodes_m != target_nodes or (not np.isfinite(f_m)):
            # stay inside bracket by nudging slightly
            # (in practice this rarely triggers once we're in a node-correct window)
            Em = 0.5*(E_lo + E_hi)
            f_m, nodes_m, uo, ui, i_match = mismatch_nodes(Em)

        if nodes_m != target_nodes or (not np.isfinite(f_m)):
            # If still bad, shrink interval a bit and continue
            E_lo = 0.5*(E_lo + Em)
            E_hi = 0.5*(E_hi + Em)
            continue

        Em_best, uo_best, ui_best, i_match_best = Em, uo, ui, i_match

        if np.sign(f_m) == np.sign(f_lo):
            E_lo, f_lo = Em, f_m
        else:
            E_hi, f_hi = Em, f_m

        if abs(E_hi - E_lo) < 1e-12:
            break

    if uo_best is None:
        raise RuntimeError("Bisection failed to produce a valid wavefunction. Try smaller h and larger r_max.")

    Em_best, uo_best, ui_best = refine_energy_by_rms(
        E_lo, E_hi, l, r, Vef, i_match_best,
        w=40, n_grid=41
    )
    print('Refined energy:', Em_best)
    # stitch & normalize  (BLENDED to avoid visible kink)
    u = np.zeros_like(r)

    # ensure inward piece is scaled to match outward at the match index
    ui_best = ui_best * (uo_best[i_match_best] / ui_best[i_match_best])

    # choose blend half-width (in points)
    Nblend = 120  # try 50–200

    i0 = max(0, i_match_best - Nblend)
    i1 = min(len(r) - 1, i_match_best + Nblend)

    # left side from outward, right side from inward
    u[:i0] = uo_best[:i0]
    u[i1+1:] = ui_best[i1+1:]

    # smooth cosine crossfade in [i0, i1]
    x = np.linspace(0.0, 1.0, i1 - i0 + 1)
    w = 0.5 - 0.5*np.cos(np.pi * x)   # 0 -> 1 smoothly
    u[i0:i1+1] = (1.0 - w) * uo_best[i0:i1+1] + w * ui_best[i0:i1+1]

    # normalize
    u /= np.sqrt(integrate.simpson(u*u, r))
    R = u / r

    norm_u = float(integrate.simpson(u*u, r))
    norm_R = float(integrate.simpson((R*R)*(r*r), r))

    return BoundState(
        n=n, l=l, j=j,
        E_au=float(Em_best),
        r_a0=r,
        u=u,
        R=R,
        nodes=count_nodes(u),
        norm_u=norm_u,
        norm_R=norm_R,
    )
# =========================
# Example: Rb nP3/2 bound state generation + normalization check
# =========================

if __name__ == "__main__":
    # Example: Rb nP3/2 quantum defect delta0 (Rb85) ~ 2.6416737
    # Use the correct delta0 for your isotope/series if needed.
    lr = 0
    jr = 1/2

    for n in [6]:
        defect = atom.getQuantumDefect(n=n, l=lr, j=jr)
        st = solve_bound_state_nodegated(
            n=n, l=lr, j=jr, delta0=defect,
            r_min=1e-3, r_max=4*n**2, h=0.001
        )
        print(f"Rb {n}P3/2 bound state")
        print(f"  Energy E = {st.E_au:.8e} Ha = {st.E_au*Eh/constants.h:.3e} Hz (in frequency units)")
        print(f"  Nodes (numerical) = {st.nodes}  (expected ~ n-l-1 = {n-lr-1})")
        print(f"  Norm check: ∫u^2 dr = {st.norm_u:.10f}  (should be 1)")
        print(f"  Norm check: ∫|R|^2 r^2 dr = {st.norm_R:.10f} (should be 1)")
        print()
        fig, ax = plt.subplots(nrows=2)
        ax[0].plot(st.r_a0, st.u)
        ax[1].plot(st.r_a0, st.u)
        ax[1].set_ylim([0.0, 20])
        fig.show()
    # fig, ax = plt.subplots(nrows=2)
    # rs = np.arange(1e-5,800,0.1)
    # vs = V_eff(rs,1)
    # ax[0].plot(rs,V_eff(rs,1))
    # ax[0].set_ylim([-1e-2,-1e-5])
    # # ax[0].set_yscale('log')
    # fig.show()