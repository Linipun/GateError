import numpy as np
import matplotlib.pyplot as plt

dim = 9
idx = {"00": 0, "01": 1, "10": 2, "11": 3, "0r": 4, "r0": 5, "1r": 6, "r1": 7, "rr": 8}


def H0(phase, B) -> np.ndarray:
    """
    Ideal CZ Hamiltonian in the simplified model:
      H0(t) = (Ω/2) Σ_i ( e^{-iφ(t)} |1_i><r_i| + h.c. ) + B |rr><rr|
    expressed in the 9D truncated basis above.
    """
    # ph = phi(tt)
    Rabi = 1
    e = np.exp(-1j * phase)
    H = np.zeros((dim, dim), dtype=complex)

    # Qubit 2 couplings:
    # |01> <-> |0r|
    H[idx["01"], idx["0r"]] += (Rabi / 2) * e
    H[idx["0r"], idx["01"]] += (Rabi / 2) * np.conj(e)
    # |11> <-> |1r|
    H[idx["11"], idx["1r"]] += (Rabi / 2) * e
    H[idx["1r"], idx["11"]] += (Rabi / 2) * np.conj(e)
    # |r1> <-> |rr|
    H[idx["r1"], idx["rr"]] += (Rabi / 2) * e
    H[idx["rr"], idx["r1"]] += (Rabi / 2) * np.conj(e)

    # Qubit 1 couplings:
    # |10> <-> |r0|
    H[idx["10"], idx["r0"]] += (Rabi / 2) * e
    H[idx["r0"], idx["10"]] += (Rabi / 2) * np.conj(e)
    # |11> <-> |r1|
    H[idx["11"], idx["r1"]] += (Rabi / 2) * e
    H[idx["r1"], idx["11"]] += (Rabi / 2) * np.conj(e)
    # |1r> <-> |rr|
    H[idx["1r"], idx["rr"]] += (Rabi / 2) * e
    H[idx["rr"], idx["1r"]] += (Rabi / 2) * np.conj(e)

    # Blockade shift
    H[idx["rr"], idx["rr"]] += B
    return H


def O_nu() -> np.ndarray:
    """
    Frequency noise operator (detuning noise):
      O_ν = -2π Σ_i |r_i><r_i|
    In this truncated basis: singly excited r-states contribute -2π,
    and |rr> contributes -4π.
    """
    O = np.zeros((dim, dim), dtype=complex)
    for k in ["0r", "r0", "1r", "r1"]:
        O[idx[k], idx[k]] += -2 * np.pi
    O[idx["rr"], idx["rr"]] += -4 * np.pi
    return O


def O_I_inst(phase) -> np.ndarray:
    """
    Intensity noise operator (amplitude noise):
      O_I(t) = (Ω/4) Σ_i ( e^{-iφ(t)} |1_i><r_i| + h.c. )
    Same graph as H0 but Ω/4 instead of Ω/2.
    """
    # ph = phi(tt)
    Rabi = 1
    e = np.exp(-1j * phase)
    pref = Rabi / 4
    O = np.zeros((dim, dim), dtype=complex)

    # Qubit 2 edges
    O[idx["01"], idx["0r"]] += pref * e
    O[idx["0r"], idx["01"]] += pref * np.conj(e)
    O[idx["11"], idx["1r"]] += pref * e
    O[idx["1r"], idx["11"]] += pref * np.conj(e)
    O[idx["r1"], idx["rr"]] += pref * e
    O[idx["rr"], idx["r1"]] += pref * np.conj(e)

    # Qubit 1 edges
    O[idx["10"], idx["r0"]] += pref * e
    O[idx["r0"], idx["10"]] += pref * np.conj(e)
    O[idx["11"], idx["r1"]] += pref * e
    O[idx["r1"], idx["11"]] += pref * np.conj(e)
    O[idx["1r"], idx["rr"]] += pref * e
    O[idx["rr"], idx["1r"]] += pref * np.conj(e)

    return O


def step_unitary(H: np.ndarray, dt_: float) -> np.ndarray:
    evals, evecs = np.linalg.eigh(H)  # H is Hermitian
    return (evecs * np.exp(-1j * evals * dt_)) @ evecs.conj().T


def propagate_U(phases, dt, B) -> np.ndarray:
    Nt = len(phases)
    U = np.eye(dim, dtype=complex)
    Us = np.empty((Nt, dim, dim), dtype=complex)
    Us[0] = U
    for k in range(Nt - 1):
        # tm = 0.5 * (t[k] + t[k + 1])
        U = step_unitary(H0(phases[k + 1], B=B), dt) @ U
        Us[k + 1] = U
    return Us


def build_Oseq(phases, dt, B, is_intensity: bool) -> np.ndarray:
    Nt = len(phases)
    Us = propagate_U(phases=phases, dt=dt, B=B)
    Udag = np.conjugate(np.swapaxes(Us, 1, 2))

    Onu = O_nu()
    Oseq = np.empty((Nt, dim, dim), dtype=complex)
    for k in range(Nt):
        Oinst = O_I_inst(phases[k]) if is_intensity else Onu
        Oseq[k] = Udag[k] @ Oinst @ Us[k]
    return Oseq


def isometry_haar_full() -> np.ndarray:
    """Isometry S (dim x 4) embedding logical {|00>,|01>,|10>,|11>} into full 9D."""
    S = np.zeros((dim, 4), dtype=complex)
    for i in range(4):
        S[i, i] = 1.0
    return S

def isometry_symmetric() -> np.ndarray:
    """Isometry S (dim x 3) for the symmetric logical subspace."""
    S = np.zeros((dim, 3), dtype=complex)
    S[idx["00"], 0] = 1.0
    S[idx["01"], 1] = 1 / np.sqrt(2)
    S[idx["10"], 1] = 1 / np.sqrt(2)
    S[idx["11"], 2] = 1.0
    return S

def response_G13(Oseq: np.ndarray, S: np.ndarray, omegas_noise: float, dt: float) -> np.ndarray:
    """
    Computes the universal response I(ω) for ω in omegas using Eq. (G13)
    with an isometry S (dim x D) defining the Haar ensemble subspace.
      P = S S†  (projector), D = Tr(P)

    Discrete form:
      I(ω) ≈ Σ_{j,k} cos(ω (t_j - t_k)) * K(t_j, t_k) * dt^2
    where K is the Eq.(G13) averaged connected correlator.
    """
    Nt_local = Oseq.shape[0]
    D = S.shape[1]
    Sdag = S.conj().T

    # Sequences needed for correlations
    # X(t)=S† O(t)  (D x dim),  Y(t)=O(t) S (dim x D)
    X = np.einsum("ad,tdk->tak", Sdag, Oseq)  # (Nt, D, dim)
    Y = np.einsum("tdk,kb->tdb", Oseq, S)  # (Nt, dim, D)

    # M(t)=S† O(t) S (D x D)
    M = np.einsum("ad,tdk,kb->tab", Sdag, Oseq, S)  # (Nt, D, D)

    # c(t)=Tr[O(t)P]=Tr[M(t)]
    c = np.trace(M, axis1=1, axis2=2)  # (Nt,)

    # Build lag correlations over k = -(Nt-1)...(Nt-1):
    # A_k = Σ_l Tr[ S†O(l+k) O(l) S ]  = Σ_l Tr[ X(l+k) Y(l) ]
    A_corr = np.zeros(2 * Nt_local - 1, dtype=complex)
    for a in range(D):
        for j in range(dim):
            A_corr += np.correlate(X[:, a, j], np.conj(Y[:, j, a]), mode="full")

    # B_k = Σ_l Tr[ M(l+k) M(l) ]
    B_corr = np.zeros(2 * Nt_local - 1, dtype=complex)
    for a in range(D):
        for b in range(D):
            B_corr += np.correlate(M[:, a, b], np.conj(M[:, b, a]), mode="full")

    # C_k = Σ_l c(l+k) c(l)
    C_corr = np.correlate(c, np.conj(c), mode="full")

    # Eq. (G13) kernel summed over the second time index:
    # Ksum(k) = (1/D) A_k - 1/(D(D+1)) (B_k + C_k)
    Ksum = (1.0 / D) * A_corr - (1.0 / (D * (D + 1.0))) * (B_corr + C_corr)

    # Discrete double integral factor
    K = np.real(Ksum) * (dt * dt)

    lags = np.arange(-(Nt_local - 1), Nt_local)
    tau = lags * dt

    # I(ω) = Σ_k K(k) cos(ω tau_k)
    return np.sum(K * np.cos(omegas_noise * tau))


# -----------------------------
# 8) Main: compute universal curves and plot
# -----------------------------
if __name__ == "__main__":
    # Frequency axis: x = ω/Ω = 2π f/Ω (as in Fig. 15)
    x = np.linspace(0.0, 3.0, 501)
    Omega = 7.7*2*np.pi
    omegas = x * Omega

    # Build Heisenberg sequences
    Oseq_nu = build_Oseq(is_intensity=False)
    Oseq_I = build_Oseq(is_intensity=True)

    # Isometries
    S_haar = isometry_haar_full()   # D=4
    S_sym = isometry_symmetric()    # D=3

    # Responses (Appendix G Eq. G13)
    I_nu_haar = response_G13(Oseq_nu, S_haar, omegas)
    I_nu_sym = response_G13(Oseq_nu, S_sym, omegas)
    I_I_haar = response_G13(Oseq_I, S_haar, omegas)
    I_I_sym = response_G13(Oseq_I, S_sym, omegas)

    # Fig. 15(a) convention often shown as Ω^2 Iν/(2π)^2 with "×4π^2"
    nu_plot_haar = (Omega**2) * I_nu_haar / ((2 * np.pi) ** 2)
    nu_plot_sym = (Omega**2) * I_nu_sym / ((2 * np.pi) ** 2)

    # Plot (a): frequency response
    plt.figure(figsize=(6.6, 4.0))
    plt.plot(x, nu_plot_haar, label="Haar")
    plt.plot(x, nu_plot_sym, label="Sym")
    plt.xlim(0, 3.0)
    plt.xlabel(r"Normalized frequency $2\pi f/\Omega$")
    plt.ylabel(r"Frequency response $\Omega^2 I_\nu/(2\pi)^2$  (axis shows $\times 4\pi^2$)")
    plt.title("Universal frequency-noise response (Appendix G)")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()

    # Plot (b): intensity response
    plt.figure(figsize=(6.6, 4.0))
    plt.plot(x, I_I_haar, label="Haar")
    plt.plot(x, I_I_sym, label="Sym")
    plt.xlim(0, 3.0)
    plt.xlabel(r"Normalized frequency $2\pi f/\Omega$")
    plt.ylabel(r"Intensity response $I_I$")
    plt.title("Universal intensity-noise response (Appendix G)")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()