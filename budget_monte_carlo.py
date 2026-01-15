import numpy as np
import matplotlib.pyplot as plt
from arc import *
import pickle
import scipy.optimize as opt
import scipy
import os
import json
from datetime import datetime
from pathlib import Path


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


with open('blockades_symmetric.pkl', 'rb') as file:
    # Load the object from the file
    blockade_dict = pickle.load(file)


def find_blockade_Mrad(atom_name, n, d):
    blockades = blockade_dict['Cs'][str(n)]
    b_r = [l[0] for l in blockades]
    b_values = [l[1] for l in blockades]
    b = np.interp(d, b_r, b_values)
    return b * 1e3 * 2 * np.pi


class Hamiltonians:
    """
    Provides Hamiltonian constructors given pulse parameters.
    """

    def __init__(self,
                 Omega_Rabi1,
                 blockade_inf: bool,
                 blockade,
                 r_lifetime,
                 r_lifetime2,
                 Delta1,
                 Stark1,
                 Stark2,
                 pulse_time,
                 resolution
                 ):
        # raw parameters
        self.Omega_Rabi1 = Omega_Rabi1
        self.blockade_inf = blockade_inf
        self.blockade = blockade
        self.r_lifetime = r_lifetime
        self.Delta1 = Delta1
        self.resolution = resolution
        self.Stark1 = Stark1
        self.Stark2 = Stark2
        self.r_lifetime2 = r_lifetime2

        self.pulse_time = pulse_time
        self.resolution = resolution

        # fixed initial state
        self.initial_psi01 = np.array([1, 0], dtype=complex)
        self.init_parameters()

        self.phase = np.zeros(resolution)

    # Dynamic initial parameters
    def init_parameters(self):
        """Re-derive normalized / derived parameters"""
        assert self.blockade_inf in (0, 1), "blockade_inf must be 0 (finite blockade) or 1 (infinite blockade)"
        self.normalized_blockade = self.blockade / self.Omega_Rabi1

        if self.blockade_inf == 1:
            self.initial_psi11 = self.initial_psi01.copy()
        else:
            self.initial_psi11 = np.array([1, 0, 0, 0], dtype=complex)

        self.decay_rate = (1 / self.r_lifetime) / self.Omega_Rabi1
        self.decay_rate2 = (1 / self.r_lifetime2) / self.Omega_Rabi1

    def H11(self, phase_i, omega_scale: float = 1.0):
        """Two-atom Hamiltonian at a given phase (Hermitian)."""
        Omega1 = omega_scale * np.exp(1j * phase_i) / 2
        Delta1 = self.Delta1 / self.Omega_Rabi1
        Stark1 = self.Stark1 / self.Omega_Rabi1
        Stark2 = self.Stark2 / self.Omega_Rabi1
        if self.blockade_inf == 1:
            H = np.array([
                [0, Omega1 * np.sqrt(2)],
                [np.conj(Omega1) * np.sqrt(2), Delta1],
            ], complex)
            decay_matrix = np.diag([0, -1j * self.decay_rate / 2])
        else:
            B = self.normalized_blockade
            H = np.array([
                [0, Omega1, Omega1, 0],
                [np.conj(Omega1), Delta1 + Stark1, 0, Omega1],
                [np.conj(Omega1), 0, Delta1 + Stark2, Omega1],
                [0, np.conj(Omega1), np.conj(Omega1), 2 * Delta1 + Stark1 + Stark2 + B],

            ], complex)
            decay_matrix = np.diag([0, self.decay_rate, self.decay_rate2, self.decay_rate + self.decay_rate2])
            H += (-1j * decay_matrix / 2)
        return H

    def H01(self, phase_i, omega_scale: float = 1.0):
        """Single-atom Hamiltonian at a given phase (Hermitian)."""
        Omega1 = omega_scale * np.exp(1j * phase_i) / 2
        Delta1 = self.Delta1 / self.Omega_Rabi1

        H = np.array([
            [0, Omega1],
            [np.conj(Omega1), Delta1],
        ], complex)
        decay_matrix = np.diag([0, -1j * self.decay_rate / 2])
        H += decay_matrix
        return H

    @staticmethod
    def bell_state_fidelity(psi01, psi11):
        """Compute Bell state fidelity from single- and two-atom amplitudes."""
        return (1 / 16) * np.abs(1 + 2 * psi01[0] - psi11[0]) ** 2  # bell state fidelity

    def return_fidel(self, phases=None, dt=None, omega_scale=1):
        psi01 = self.initial_psi01.copy()
        psi11 = self.initial_psi11.copy()
        for phi in phases:
            U01 = scipy.linalg.expm(-1j * self.H01(phi, omega_scale=omega_scale) * dt)
            U11 = scipy.linalg.expm(-1j * self.H11(phi, omega_scale=omega_scale) * dt)
            psi01 = U01 @ psi01
            psi11 = U11 @ psi11
        # remove global phase
        gp = psi01[0] / np.abs(psi01[0])
        psi01 /= gp
        psi11 /= gp ** 2
        return self.bell_state_fidelity(psi01, psi11), gp  # returns [fidelity, rotation_phase]


def fid_optimize(param, fid_gen):
    time, phase, dt = phase_cosine_generate(*param, fid_gen.pulse_time, fid_gen.resolution)
    # print(phase)
    # print(dt)
    fid, global_phi = fid_gen.return_fidel(phases=phase, dt=dt)
    return 1 - fid


def phase_cosine_generate(A, w, phi, gamma, pulse_time, resolution):
    times = np.linspace(0, pulse_time, resolution)
    dt = times[1] - times[0]
    phases = A * np.cos(w * times - phi) + gamma * times
    return times, phases, dt


def sample_pair_distances(
        n_samples=1000,
        sigma_r=0.0849,
        sigma_z=0.425,
        x_offset=2.5,
        rng=None
):
    """
    Monte Carlo distances between two point particles drawn from Gaussian clouds.

    Geometry:
      - Particle A: (x, y, z) ~ N( (0, 0, 0), diag([sigma_r^2, sigma_r^2, sigma_z^2]) )
      - Particle B: (x, y, z) ~ N( (x_offset, 0, 0), diag([sigma_r^2, sigma_r^2, sigma_z^2]) )

    Parameters
    ----------
    n_samples : int
        Number of particle pairs to simulate.
    sigma_r : float
        Standard deviation of the transverse (x,y) Gaussian for each cloud.
    sigma_z : float
        Standard deviation of the longitudinal z Gaussian for each cloud.
    x_offset : float
        Separation between the two cloud centers along +x (both have z0=0).
    rng : np.random.Generator or None
        Optional NumPy random generator for reproducibility.

    Returns
    -------
    distances : np.ndarray, shape (n_samples,)
        Euclidean separation distances for each pair.
    coords_a : dict
        Sampled coordinates for particle A: {'x','y','z'} each (n_samples,)
    coords_b : dict
        Sampled coordinates for particle B: {'x','y','z'} each (n_samples,)
    """
    if rng is None:
        rng = np.random.default_rng()

    # Particle A
    ax = rng.normal(loc=0.0, scale=sigma_r, size=n_samples)
    ay = rng.normal(loc=0.0, scale=sigma_r, size=n_samples)
    az = rng.normal(loc=0.0, scale=sigma_z, size=n_samples)

    # Particle B (shifted by x_offset along x; same z0=0)
    bx = rng.normal(loc=x_offset, scale=sigma_r, size=n_samples)
    by = rng.normal(loc=0.0, scale=sigma_r, size=n_samples)
    bz = rng.normal(loc=0.0, scale=sigma_z, size=n_samples)

    # Distances
    dx = ax - bx
    dy = ay - by
    dz = az - bz
    distances = np.sqrt(dx * dx + dy * dy + dz * dz)

    coords_a = {"x": ax, "y": ay, "z": az}
    coords_b = {"x": bx, "y": by, "z": bz}
    return distances, coords_a, coords_b


def sigma_r_um(atom_T_uK, U_trap_max_uK, w0_um):
    return (0.5) * np.sqrt(atom_T_uK / U_trap_max_uK)


def sigma_z_um(atom_T_uK, U_trap_max_uK, w0_um, wavelength_um):
    zR = np.pi * w0_um ** 2 / wavelength_um
    return zR * np.sqrt(atom_T_uK / (2 * U_trap_max_uK))


def sample_gaussian(std, rng=None):
    """
    Draw a single value from a Gaussian distribution with mean 0
    and standard deviation `std`.

    Parameters
    ----------
    std : float
        Standard deviation of the Gaussian distribution.
    rng : np.random.Generator or None
        Optional NumPy random generator for reproducibility.

    Returns
    -------
    float
        A single random value from N(0, std^2).
    """
    if rng is None:
        rng = np.random.default_rng()
    return rng.normal(loc=0.0, scale=std)



if __name__ == "__main__":
    outdirs ='results'
    os.makedirs(outdirs, exist_ok=True)

    ## configuration ##
    atom_name = 'Cs'
    n = 50
    l = 1
    j = 3 / 2
    mj = 3 / 2

    atom_d = 2.5  # um
    Omega_Rabi = 1.5 * 2 * np.pi
    pulse_time = 7.65
    resolution = 100  # number of phase steps in the pulse

    T_atom = 15  # uK
    trap_depth = 300  # uK
    lambda_trap = 1.064  # um
    w0_trap = 1.2  # um

    num_samples = 100

    ## configuration ##


    if atom_name == "Rb":
        atom = Rubidium()
    elif atom_name == "Cs":
        atom = Caesium()
    blockade_mrad = find_blockade_Mrad(atom_name, n, atom_d)
    R_lifetime = atom.getStateLifetime(n=n,l=l,j=j,temperature=300, includeLevelsUpTo=n+20,s=0.5)*1e6
    m_atom = atom.mass

    #Find optimized phase
    H_gen = Hamiltonians(Omega_Rabi1=Omega_Rabi, blockade_inf=False, blockade=blockade_mrad, r_lifetime=R_lifetime,
                         Delta1=0,
                         Stark1=0, Stark2=0, resolution=resolution, r_lifetime2=R_lifetime, pulse_time=pulse_time)
    PhaseGuess = [2 * np.pi * 0.1122, 1.0431, -0.7318, 0]
    time, phase_guess, dt = phase_cosine_generate(*PhaseGuess, H_gen.pulse_time, H_gen.resolution)
    # fid_optimize(PhaseGuess, H_gen)
    # H_gen.return_fidel
    fid, global_phi = H_gen.return_fidel(phases=phase_guess, dt=dt)
    print('Infidelity before optimizer:', 1 - fid)
    opt_out = opt.minimize(fun=fid_optimize, x0=PhaseGuess, args=(H_gen))
    phase_params = opt_out.x
    # print(phase_params)
    infid = opt_out.fun
    print('Infidelity after optimizer:', infid)
    print('phase parameter', phase_params)
    ##%%
    time, phase, dt = phase_cosine_generate(*phase_params, H_gen.pulse_time,H_gen.resolution)
    fid, global_phi = H_gen.return_fidel(phases=phase, dt=dt)
    fig, ax = plt.subplots()
    ax.plot(time, phase)
    ax.set_xlabel('time $(\Omega T)$')
    ax.set_ylabel('phase (rad)')
    fig.savefig(os.path.join(outdirs,'phases.pdf'))

    #shot-to-shot detuning
    deltas = np.linspace(0, 0.1, 100)
    deltas *= Omega_Rabi
    infids_mean_detuning = []
    infids_std_detuing = []
    for delta in deltas:
        infids_s = []
        for i in range(num_samples):
            d = sample_gaussian(delta)
            H_gen = Hamiltonians(Omega_Rabi1=Omega_Rabi, blockade_inf=False, blockade=blockade_mrad,
                                 r_lifetime=R_lifetime, Delta1=d,
                                 Stark1=0, Stark2=0, resolution=resolution, r_lifetime2=R_lifetime,
                                 pulse_time=pulse_time)
            fid, global_phi = H_gen.return_fidel(phases=phase, dt=dt)
            infids_s.append(1 - fid)
        infids_s = np.asarray(infids_s)
        infids_mean_detuning.append(np.mean(infids_s))
        infids_std_detuing.append(infids_s.std(ddof=1))

    infids_mean_detuning = np.asarray(infids_mean_detuning)
    infids_std_detuing  = np.asarray(infids_std_detuing)
    sem_detuning = infids_std_detuing / np.sqrt(num_samples)
    fractional_delta = deltas/Omega_Rabi
    fig2, ax2 = plt.subplots(ncols=3, figsize=(12, 4))
    ax2[0].plot(fractional_delta, infids_mean_detuning, label="mean(1 - fid)")
    ax2[0].fill_between(
        fractional_delta,
        infids_mean_detuning - sem_detuning,
        infids_mean_detuning + sem_detuning,
        alpha=0.25,
        label="±1 std"
    )
    ax2[0].set_xlabel(r"$\delta \Delta_{DC}/ \Omega $", fontsize=20)
    ax2[0].set_ylabel("$1 - \mathcal{F}$", fontsize=20)
    ax2[0].set_title('Shot-to-shot detuning')


    # shot-to-shot Blockade
    infids_mean_blockade = []
    infids_std_blockade = []
    T_atoms = np.linspace(5, 20, 100)
    for T_atom in T_atoms:
        infids_s = []
        sigma_r = sigma_r_um(T_atom, trap_depth, w0_trap)
        sigma_z = sigma_z_um(T_atom, trap_depth, w0_trap, lambda_trap)
        distances = sample_pair_distances(
            n_samples=num_samples,
            sigma_r=sigma_r,
            sigma_z=sigma_z,
            x_offset=atom_d,
            rng=None
        )[0]
        for d in distances:
            blockade = find_blockade_Mrad(atom_name, n, d)
            H_gen = Hamiltonians(Omega_Rabi1=Omega_Rabi, blockade_inf=False, blockade=blockade, r_lifetime=R_lifetime,
                                 Delta1=0,
                                 Stark1=0, Stark2=0, resolution=resolution, r_lifetime2=R_lifetime,
                                 pulse_time=pulse_time)
            fid, global_phi = H_gen.return_fidel(phases=phase, dt=dt)
            infids_s.append(1 - fid)
        infids_s = np.asarray(infids_s)
        infids_mean_blockade.append(np.mean(infids_s))
        infids_std_blockade.append(infids_s.std(ddof=1))
    infids_mean_blockade = np.asarray(infids_mean_blockade)
    infids_std_blockade = np.asarray(infids_std_blockade)
    sem_blockade = infids_std_blockade / np.sqrt(num_samples)

    ax2[1].plot(T_atoms, infids_mean_blockade, label="mean(1 - fid)")
    ax2[1].fill_between(
        T_atoms,
        infids_mean_blockade - sem_blockade,
        infids_mean_blockade + sem_blockade,
        alpha=0.25,
        label="±1 std"
    )
    ax2[1].set_xlabel(r"$T(\mu K)$", fontsize=20)
    ax2[1].set_ylabel("$1 - \mathcal{F}$", fontsize=20)
    ax2[1].set_title('Shot-to-shot Bloackade')

    #shot-to-shot Rabi
    delta_omegas = np.linspace(0, 0.1, 100)
    infids_mean_omega = []
    infids_std_omega = []
    for delta_omega in delta_omegas:
        infids_s = []
        for i in range(num_samples):
            omega_scaled = sample_gaussian(delta_omega)
            H_gen = Hamiltonians(Omega_Rabi1=Omega_Rabi, blockade_inf=False, blockade=blockade_mrad,
                                 r_lifetime=R_lifetime, Delta1=0,
                                 Stark1=0, Stark2=0, resolution=resolution, r_lifetime2=R_lifetime,
                                 pulse_time=pulse_time)
            fid, global_phi = H_gen.return_fidel(phases=phase, dt=dt, omega_scale=(1 - omega_scaled))
            infids_s.append(1 - fid)
        infids_s = np.asarray(infids_s)
        infids_mean_omega.append(np.mean(infids_s))
        infids_std_omega.append(infids_s.std(ddof=1))
    infids_mean_omega = np.asarray(infids_mean_omega)
    infids_std_omega = np.asarray(infids_std_omega)
    sem_omega = infids_std_omega / np.sqrt(num_samples)
    ax2[2].plot(delta_omegas, infids_mean_omega, label="mean(1 - fid)")
    ax2[2].fill_between(
        delta_omegas,
        infids_mean_omega - sem_omega,
        infids_mean_omega + sem_omega,
        alpha=0.25,
        label="±1 std"
    )
    ax2[2].set_xlabel(r"$\delta \Omega_{DC}/ \Omega $", fontsize=20)
    ax2[2].set_ylabel("$1 - \mathcal{F}$", fontsize=20)
    ax2[2].set_title('Shot-to-shot Rabi')

    fig2.savefig(os.path.join(outdirs,'shot-to-shot_calculation.pdf'))

    config = {
        "atom_name": atom_name,
        "n": n, "l": l, "j": float(j), "mj": float(mj),
        "atom_d_um": atom_d,
        "Omega_Rabi_rad_per_us": Omega_Rabi,
        "pulse_time_OmegaT": pulse_time,
        "resolution": resolution,
        "T_atom_uK": T_atom,
        "trap_depth_uK": trap_depth,
        "lambda_trap_um": lambda_trap,
        "w0_trap_um": w0_trap,
        "num_samples": num_samples,
        "blockade_mrad": float(blockade_mrad),
        "R_lifetime_us": float(R_lifetime),
        "PhaseGuess": [float(x) for x in PhaseGuess],
        "phase_params_opt": [float(x) for x in phase_params],
        "infid_before_opt": float(1 - fid),  # fid computed just before optimizer in your code
        "infid_after_opt": float(infid),
    }

    # Save config JSON
    config_path = os.path.join(outdirs, f"run_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    # Save raw x/y used in fig2 (lossless)
    raw_npz_path = os.path.join(outdirs,f"infid_raw.npz")
    np.savez(
        raw_npz_path,
        x_detuning=fractional_delta, y_detuning=infids_mean_detuning, sem_detuning=sem_detuning,
        T_blockade=T_atoms, infid_blockade=infids_mean_blockade, sem_blockade=sem_blockade,
        x_rabi=delta_omegas, y_rabi=infids_mean_omega, sem_rabi=sem_omega,
    )


    print(f"Saved: {config_path}")
    print(f"Saved: {raw_npz_path}")