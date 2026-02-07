import numpy as np
from arc import *
dim = 9
idx = {"00":0, "01":1, "10":2, "11":3, "0r":4, "r0":5, "1r":6, "r1":7, "rr":8}

h = 6.626e-34
e = 1.602e-19
a0 = 5.291e-11
hbar = h/2/np.pi
EH = 4.359744e-18
c = 299792458
kb = 1.380649e-23
me = 9.1093837e-31
epi0 = 8.854e-12
bohr_r = 5.291e-11
w_qubit = 9192631770*2*np.pi

def O2photon_I1(phase, Omega1,Omega2, Delta, ktilde0_1, ktilder_1, ktilde0_2, ktilder_2) -> np.ndarray:
    """
    Intensity-noise operator for arm 1 (Eq. K8). [oai_citation:9‡PRXQuantum.6.010331.pdf](sediment://file_00000000ff6871f898efa8a5b8425e17)
    """
    Om1 = Omega1
    Om2 = Omega2

    O = np.zeros((dim, dim), dtype=complex)

    # First term: (Ω1Ω2)/(8Δ) Σ_i (|1_i><r_i| + h.c.)
    Oeff_I = (Om1 * Om2) / (8.0 * Delta)
    phase = np.exp(-1j * phase)

    # same couplings as H_eff, scaled
    for (a,b) in [("01","0r"), ("11","1r"), ("r1","rr"), ("10","r0"), ("11","r1"), ("1r","rr")]:
        O[idx[a], idx[b]] += Oeff_I * phase
        O[idx[b], idx[a]] += Oeff_I * np.conj(phase)

    # + κ̃0,1 * Ω1^2 * Σ_i |0_i><0_i|   (up to the same Ω^2/Δ scaling in Eq. K8)
    # Eq. (K8) shows this as proportional to κ̃0,1 Ω1(t)^2 (formatting is compressed in the PDF parse).
    ls1 = (Om1*Om1)
    for state in ["00", "01", "0r"]:
        O[idx[state], idx[state]] += ktilde0_1 * ls1
    for state in ["00", "10", "r0"]:
        O[idx[state], idx[state]] += ktilde0_1 * ls1

    # + (κ̃r,1 - 1/(4Δ)) * Ω1^2 * Σ_i |r_i><r_i|   (Eq. K8) [oai_citation:10‡PRXQuantum.6.010331.pdf](sediment://file_00000000ff6871f898efa8a5b8425e17)
    coeff_r = (ktilder_1 - 1.0/(4.0*Delta)) * ls1
    for state in ["0r","r0","1r","r1"]:
        O[idx[state], idx[state]] += coeff_r
    O[idx["rr"], idx["rr"]] += 2.0 * coeff_r
    return O


def O2photon_I2(phase, Omega1,Omega2, Delta, ktilde0_1, ktilder_1, ktilde0_2, ktilder_2) -> np.ndarray:
    """
    Intensity-noise operator for arm 2 (Eq. K9). [oai_citation:11‡PRXQuantum.6.010331.pdf](sediment://file_00000000ff6871f898efa8a5b8425e17)
    """
    Om1 = Omega1
    Om2 = Omega2

    O = np.zeros((dim, dim), dtype=complex)

    # First term: (Ω1Ω2)/(8Δ) Σ_i (|1_i><r_i| + h.c.)
    Oeff_I = (Om1 * Om2) / (8.0 * Delta)
    phase = np.exp(-1j * phase)

    for (a,b) in [("01","0r"), ("11","1r"), ("r1","rr"), ("10","r0"), ("11","r1"), ("1r","rr")]:
        O[idx[a], idx[b]] += Oeff_I * phase
        O[idx[b], idx[a]] += Oeff_I * np.conj(phase)

    # + κ̃0,2 * Ω2^2 * Σ_i |0_i><0_i|
    ls2 = (Om2*Om2)
    for state in ["00", "01", "0r"]:
        O[idx[state], idx[state]] += ktilde0_2 * ls2
    for state in ["00", "10", "r0"]:
        O[idx[state], idx[state]] += ktilde0_2 * ls2

    # + (κ̃r,2 + 1/(4Δ)) * Ω2^2 * Σ_i |r_i><r_i|  (Eq. K9) [oai_citation:12‡PRXQuantum.6.010331.pdf](sediment://file_00000000ff6871f898efa8a5b8425e17)
    coeff_r = (ktilder_2 + 1.0/(4.0*Delta)) * ls2
    for state in ["0r","r0","1r","r1"]:
        O[idx[state], idx[state]] += coeff_r
    O[idx["rr"], idx["rr"]] += 2.0 * coeff_r
    return O


def O2photon_nu1(phase, Omega1,Omega2, Delta, ktilde0_1, ktilder_1, ktilde0_2, ktilder_2) -> np.ndarray:
    """
    Frequency-noise operator for laser 1 (Eq. K10). [oai_citation:13‡PRXQuantum.6.010331.pdf](sediment://file_00000000ff6871f898efa8a5b8425e17)
    In the effective 3-level reduction it becomes proportional to Σ_i |r_i><r_i|.
    """
    O = np.zeros((dim, dim), dtype=complex)
    for state in ["0r","r0","1r","r1"]:
        O[idx[state], idx[state]] += -2*np.pi
    O[idx["rr"], idx["rr"]] += -4*np.pi
    return O


def O2photon_nu2(phase, Omega1,Omega2, Delta, ktilde0_1, ktilder_1, ktilde0_2, ktilder_2) -> np.ndarray:
    """
    Frequency-noise operator for laser 2 (Eq. K11). [oai_citation:14‡PRXQuantum.6.010331.pdf](sediment://file_00000000ff6871f898efa8a5b8425e17)
    Same effective form as O_nu1 in this reduction.
    """
    return O2photon_nu1(phase, Omega1,Omega2, Delta, ktilde0_1, ktilder_1, ktilde0_2, ktilder_2)


def response_2photon(Oseq, S, omega_noise, dt):
    """
    Eq. (G13) Haar-average implemented via lag correlations (fast O(Nt log Nt)-ish).
    """
    Nt_local = Oseq.shape[0]
    D = S.shape[1]
    Sdag = S.conj().T

    X = np.einsum("ad,tdk->tak", Sdag, Oseq)        # (Nt, D, dim)
    Y = np.einsum("tdk,kb->tdb", Oseq, S)           # (Nt, dim, D)
    M = np.einsum("ad,tdk,kb->tab", Sdag, Oseq, S)  # (Nt, D, D)
    c = np.trace(M, axis1=1, axis2=2)

    A_corr = np.zeros(2*Nt_local-1, dtype=complex)
    for a in range(D):
        for j in range(dim):
            A_corr += np.correlate(X[:, a, j], np.conj(Y[:, j, a]), mode="full")

    B_corr = np.zeros(2*Nt_local-1, dtype=complex)
    for a in range(D):
        for b in range(D):
            B_corr += np.correlate(M[:, a, b], np.conj(M[:, b, a]), mode="full")

    C_corr = np.correlate(c, np.conj(c), mode="full")

    Ksum = (1.0/D)*A_corr - (1.0/(D*(D+1.0)))*(B_corr + C_corr)
    K = np.real(Ksum) * (dt*dt)

    lags = np.arange(-(Nt_local-1), Nt_local)
    tau = lags * dt
    return np.sum(K * np.cos(omega_noise*tau))

def step_unitary_2photon(H, dt_):
    evals, evecs = np.linalg.eigh(H)
    return (evecs * np.exp(-1j*evals*dt_)) @ evecs.conj().T

def H_eff_2photon(phase: float, B: float, Omega1: float, Omega2: float,
          delta1: float, delta2: float, Delta: float, n: int,
          inter_detuning: float, ktilde0_1, ktilder_1, ktilde0_2, ktilder_2 ) -> np.ndarray:
    """
    Implements Eq. (K7) in the 9D basis.

    Eq. (K7) structure:
      H_eff =
        (Ω1 Ω2)/(4Δ) Σ_i ( |1_i><r_i| + h.c. )
        + (Ω1^2)/(Δ) * light shifts on |0> and |r| (relative to |1|) for arm 1
        + (Ω2^2)/(Δ) * light shifts on |0> and |r| (relative to |1|) for arm 2
        - { δ1+δ2 + (Ω1^2-Ω2^2)/(4Δ) } Σ_i |r_i><r_i|
        + B |rr><rr|
    with coefficients as written in Eq. (K7). [oai_citation:8‡PRXQuantum.6.010331.pdf](sediment://file_00000000ff6871f898efa8a5b8425e17)

    NOTE: The parsed PDF formatting can drop some parentheses; this implementation
    follows the displayed Eq. (K7) and the consistency with the effective noise
    operators (K8,K9).
    """
    Om1 = Omega1
    Om2 = Omega2
    de1 = delta1
    de2 = delta2

    # Effective two-photon coupling strength in Eq. (K7): Ω1Ω2/(4Δ)
    Oeff = (Om1 * Om2) / (4.0 * Delta)

    # Optional phase on effective coupling
    phase = np.exp(-1j * phase)

    # Self-light-shift term in Eq. (K7): (Ω1^2 - Ω2^2)/(4Δ)
    self_ls = (Om1*Om1 - Om2*Om2) / (4.0 * Delta)

    # Build Hamiltonian
    H = np.zeros((dim, dim), dtype=complex)

    # --- Effective |1><->|r| couplings (apply to each atom) ---
    # Atom 2: |01><->|0r| and |11><->|1r| and |r1><->|rr|
    H[idx["01"], idx["0r"]] += Oeff * phase
    H[idx["0r"], idx["01"]] += Oeff * np.conj(phase)

    H[idx["11"], idx["1r"]] += Oeff * phase
    H[idx["1r"], idx["11"]] += Oeff * np.conj(phase)

    H[idx["r1"], idx["rr"]] += Oeff * phase
    H[idx["rr"], idx["r1"]] += Oeff * np.conj(phase)

    # Atom 1: |10><->|r0| and |11><->|r1| and |1r><->|rr|
    H[idx["10"], idx["r0"]] += Oeff * phase
    H[idx["r0"], idx["10"]] += Oeff * np.conj(phase)

    H[idx["11"], idx["r1"]] += Oeff * phase
    H[idx["r1"], idx["11"]] += Oeff * np.conj(phase)

    H[idx["1r"], idx["rr"]] += Oeff * phase
    H[idx["rr"], idx["1r"]] += Oeff * np.conj(phase)

    # --- Light shifts from arm 1 and arm 2 on |0> and |r| relative to |1| ---
    # From Eq. (K7), these scale ~ Ωj^2/Δ with coefficients κ̃.
    ls1 = (Om1*Om1)
    ls2 = (Om2*Om2)



    # Apply |0> shifts: both atoms, so |00>,|01>,|10> get one/two contributions.
    # In this truncated basis we add shifts depending on how many qubits are in |0>.
    # (This is the natural embedding of Σ_i |0_i><0_i|.)
    # States containing a |0> on atom 1: |00>,|01>,|0r>
    # States containing a |0> on atom 2: |00>,|10>,|r0>
    for state in ["00", "01", "0r"]:
        H[idx[state], idx[state]] += (ktilde0_1*ls1 + ktilde0_2*ls2)
    for state in ["00", "10", "r0"]:
        H[idx[state], idx[state]] += (ktilde0_1*ls1 + ktilde0_2*ls2)

    # Apply |r> shifts: Σ_i |r_i><r_i| (singly excited r) and |rr| gets 2x
    r_shift = (ktilder_1*ls1 + ktilder_2*ls2)
    for state in ["0r","r0","1r","r1"]:
        H[idx[state], idx[state]] += r_shift
    H[idx["rr"], idx["rr"]] += 2.0 * r_shift

    # --- Detuning-like term on |r> manifold from Eq. (K7) ---
    det_r = (de1 + de2 + self_ls)
    for state in ["0r","r0","1r","r1"]:
        H[idx[state], idx[state]] += -det_r
    H[idx["rr"], idx["rr"]] += -2.0 * det_r

    # Blockade
    H[idx["rr"], idx["rr"]] += B
    return H


def propagate_U_2photon(phases, dt, B, Omega1, Omega2, delta1, delta2, Delta, inter_detuning, n,
                        ktilde0_1, ktilder_1, ktilde0_2, ktilder_2):
    Nt = len(phases)
    U = np.eye(dim, dtype=complex)
    Us = np.empty((Nt, dim, dim), dtype=complex)
    Us[0] = U
    for k in range(Nt-1):
        phase_i = 0.5*(phases[k]+phases[k+1])
        U = step_unitary_2photon(H_eff_2photon(phase_i, B=B, Omega1=Omega1,
                               Omega2=Omega2, delta1=delta1, delta2=delta2,
                               Delta=Delta, n=n , ktilde0_1=ktilde0_1, ktilder_1=ktilder_1,
                                               ktilde0_2=ktilde0_2, ktilder_2=ktilder_2,
          inter_detuning=inter_detuning), dt) @ U
        Us[k+1] = U
    return Us

def build_Oseq_2photon(phases, dt, B, Omega1, Omega2, delta1, delta2, Delta, inter_detuning, n,  Oinst_func):
    Nt = len(phases)
    atom = Cesium()
    v_photon1 = atom.getTransitionFrequency(n1=6, l1=0, j1=1 / 2, n2=7, l2=1, j2=1 / 2, s=0.5)
    v_photon1 += inter_detuning * 1e6 / 2 / np.pi
    v_photon2 = atom.getTransitionFrequency(n1=6, l1=1, j1=1 / 2, n2=n, l2=0, j2=1 / 2, s=0.5)
    v_photon2 -= inter_detuning * 1e6 / 2 / np.pi

    alpha_g_gen = DynamicPolarizability(atom, n=6, l=1, j=1 / 2, s=0.5)
    alpha_g_gen.defineBasis(6, 9)
    alpha_r_gen = DynamicPolarizability(atom, n=n, l=0, j=1 / 2, s=0.5)
    alpha_r_gen.defineBasis(6, n + 20)

    alpha_r_1 = alpha_r_gen.getPolarizability(c / (v_photon1), units='SI', accountForStateLifetime=False, mj=None)[0]
    d1 = atom.getDipoleMatrixElement(n1=6, l1=0, j1=1 / 2, mj1=-1 / 2, n2=7, l2=1, j2=1 / 2, mj2=1 / 2, q=1,
                                     s=0.5) * bohr_r / hbar * e

    ktilde0_1 = -(1 / 4 / (Delta + w_qubit / 1e6) - 1 / 4 / Delta)
    ktilder_1 = -(alpha_r_1 * 2 * np.pi * 1e6) / 4 / d1 ** 2 + 1 / 4 / Delta

    d2 = atom.getDipoleMatrixElement(n1=7, l1=1, j1=1 / 2, mj1=1 / 2, n2=n, l2=0, j2=1 / 2, mj2=-1 / 2, q=-1,
                                     s=0.5) * bohr_r / hbar * e
    alpha_1_2 = alpha_g_gen.getPolarizability(c / (v_photon2), units='SI', accountForStateLifetime=False, mj=None)[0]

    ktilde0_2 = 0.0
    ktilder_2 = -1 / 4 / Delta + (alpha_1_2 * 2 * np.pi * 1e6) / 4 / d2 ** 2

    Us = propagate_U_2photon(phases, dt, B, Omega1, Omega2, delta1, delta2, Delta, inter_detuning, n,ktilde0_1,
                             ktilder_1, ktilde0_2, ktilder_2)
    Udag = np.conjugate(np.swapaxes(Us, 1, 2))
    Oseq = np.empty((Nt, dim, dim), dtype=complex)

    for k in range(Nt):
        Oinst = Oinst_func(phases[k], Omega1,Omega2, Delta,
                           ktilde0_1, ktilder_1, ktilde0_2, ktilder_2)
        Oseq[k] = Udag[k] @ Oinst @ Us[k]
    return Oseq