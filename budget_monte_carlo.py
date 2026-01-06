import pickle
import numpy as np
import sys
from arc import *
import matplotlib.pyplot as plt
import scipy.linalg
import scipy.optimize as opt
import csv
from scipy.linalg import expm
from scipy.optimize import minimize

rabi_rate = 2 * np.pi * 1 #MHz

atom_name = str(sys.argv[1])
atom_n = int(sys.argv[2])
r_idx = int(sys.argv[3])
atom_separation_um = np.linspace(1.5,7,100)[r_idx]

# Open the dictionary that stores the blockades vs r
with open('blockades_symmetric.pkl', 'rb') as file:
    # Load the object from the file
    blockade_dict = pickle.load(file)
    
# Open the dictionary with the saved best pulse parameters
with open('gate_params_no_MC.pkl', 'rb') as file:
    # Load the object from the file
    start_params = pickle.load(file)

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
    by = rng.normal(loc=0.0,        scale=sigma_r, size=n_samples)
    bz = rng.normal(loc=0.0,        scale=sigma_z, size=n_samples)

    # Distances
    dx = ax - bx
    dy = ay - by
    dz = az - bz
    distances = np.sqrt(dx*dx + dy*dy + dz*dz)

    coords_a = {"x": ax, "y": ay, "z": az}
    coords_b = {"x": bx, "y": by, "z": bz}
    return distances, coords_a, coords_b

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
	
def light_shift_MHz(
    atom, #str, "Rb" or "Cs"
    n, #princ. quantum number
    E_field, #float, E field in mV/cm    
):
    """
    Returns the light shift in MHz from DC electric fileds
    
    Parameters
    ________________________________________________
    atom    : str
            Name of the atom. Either "Rb" or "Cs"
    n       : int
            Principal quantum number
    E_field : float
            Electric field magnitude, in mV / cm
    
    """
                #[n, pol_Rb, pol_Cs]
    pol_list = [[40,58.04709796,207.82965883],
                 [41,69.71219943,251.13878599],
                 [42,83.33062005,301.95970593],
                 [43,99.16752349,361.34678555],
                 [44,117.51610268,430.46904331],
                 [45,138.69974179,510.61901751],
                 [46,163.07428571,603.22198567],
                 [47,191.03041839,709.84550606],
                 [48,222.99615299,832.20924188],
                 [49,259.43943519,972.19502014],
                 [50,300.87086175,1131.8570448],
                 [51,347.84651525,1313.43218834],
                 [52,400.97091648,1519.35024601],
                 [53,460.90009543,1752.24399028],
                 [54,528.34478067,2014.95888112],
                 [55,604.07370744,2310.56219291],
                 [56,688.91704336,2642.35131888],
                 [57,783.76992986,3013.86087147],
                 [58,889.59613756,3428.86834507],
                 [59,1007.43183445,3891.39770212],
                 [60,1138.38944853,4405.72066305],
                 [61,1283.66164645,4976.35474767],
                 [62,1444.52539102,5608.05780766],
                 [63,1622.34606982,6305.81807523],
                 [64,1818.58173473,7074.83926471],
                 [65,2034.78731509,7920.51931465],
                 [66,2272.61897728,8848.42311397],
                 [67,2533.83837364,9864.24691385],
                 [68,2820.31697876,10973.77544704],
                 [69,3134.04032587,12182.82996755],
                 [70,3477.112196,13497.2074148],
                 [71,3851.75863142, 14922.61095331],
                 [72,4260.33188256, 16464.57224325],
                 [73,4705.31401207, 18128.36555941],
                 [74,5189.32029036, 19918.91673526],
                 [75,5715.10213279, 21840.70844038],
                 [76,6285.54983196, 23897.68214918],
                 [77,6903.69443012, 26093.14508576],
                 [78,7572.7090788, 28429.67953379],
                 [79,8295.90985861, 30909.06241659],
                 [80,9076.75550698, 33532.19810836]
                  ]# in units of MHz cm^2/V^2
    n_idx = n-40
    if atom == "Cs":
        alpha = pol_list[n_idx][2]
    elif atom == "Rb":
        alpha = pol_list[n_idx][1]
    else:
        raise ValueError("data not availible. Check state")
    return -0.5 * alpha * (E_field / 1000)**2
	
def lifetimeus(atom, n, l, j, T):
    return atom.getStateLifetime(n, l, j, T, includeLevelsUpTo=n+40, s=0.5)*1e6
	
def get_nearest_blockade(atom, n, r, blockades_dict):
    """
    Return the blockade B for the closest separation r from the blockades_dict.

    Parameters:
        atom (str): Atom species key, e.g. 'Rb' or 'Cs'
        n (int or str): Principal quantum number (will be converted to string key)
        r (float): Separation distance in microns
        blockades_dict (dict): Nested dictionary of the form:
            {
                "Rb": {
                    "40": [(r1, B1), (r2, B2), ...],
                    "41": [...]
                },
                "Cs": {...}
            }

    Returns:
        float: Blockade value B (GHz) corresponding to the closest available r
    """
    # Ensure atom exists
    if atom not in blockades_dict:
        raise KeyError(f"Atom '{atom}' not found in blockades_dict")

    # Ensure n is a string key
    n_key = str(n)
    if n_key not in blockades_dict[atom]:
        raise KeyError(f"n={n} not found for atom '{atom}'")

    # Extract list of (r, B) pairs
    data = blockades_dict[atom][n_key]
    if not data:
        raise ValueError(f"No data for {atom} n={n}")

    # Separate into arrays for convenience
    rs, Bs = zip(*data)

    # Find index of the closest r
    idx = min(range(len(rs)), key=lambda i: abs(rs[i] - r))

    return Bs[idx]
	
if atom_name == "Rb":
    a = Rubidium()
elif atom_name == "Cs":
    a = Caesium()

cases = {
        "atom name": atom_name,
        "separation (um)": atom_separation_um,
        "n": int(atom_n),
        "lifetime": lifetimeus(a, int(atom_n), 1, 1.5, 300), #us
        "ideal_blockade": 2 * np.pi * 1e3 * get_nearest_blockade(atom_name, int(atom_n), atom_separation_um,blockade_dict), #2 * pi * MHz
        "Rabi Rate (MHz)": rabi_rate / (2 * np.pi)
        }

#--- Static parameters and Hamiltonians ---
# Physical parameters (in MHz)
class PulseParameters:
    """
    Container for all static physical and derived pulse parameters, plus initial states and
    a time-grid generator.
    """
    def __init__(self,
                 Omega_Rabi1,
                 blockade_inf: bool,
                 blockade,
                 r_lifetime,
                 Delta1,
                 Stark1,
                 Stark2,
                 resolution,
                 optimizer: bool,
                 r_lifetime2
                 ):
        # raw parameters
        self.Omega_Rabi1  = Omega_Rabi1
        self.blockade_inf = blockade_inf
        self.blockade     = blockade
        self.r_lifetime   = r_lifetime
        self.Delta1       = Delta1
        self.resolution   = resolution
        self.optimizer    = optimizer
        self.Stark1        = Stark1
        self.Stark2        = Stark2
        self.r_lifetime2   = r_lifetime2

        assert self.optimizer in (0, 1), "optimizer must be 0 (not being run) or 1 (will be run)"
        
        # fixed initial state
        self.initial_psi01 = np.array([1, 0], dtype=complex)

        self.recompute()
           
    # Dynamic initial parameters
    def recompute(self):
        """Re-derive normalized / derived parameters"""
        assert self.blockade_inf in (0, 1), "blockade_inf must be 0 (finite blockade) or 1 (infinite blockade)"
        self.normalized_blockade = self.blockade / self.Omega_Rabi1
        
        if self.blockade_inf == 1:
            self.initial_psi11 = self.initial_psi01.copy()
        else:
            self.initial_psi11 = np.array([1, 0, 0, 0], dtype=complex)

        self.decay_rate = (1 / self.r_lifetime) / self.Omega_Rabi1
        self.decay_rate2 = (1 / self.r_lifetime2) / self.Omega_Rabi1
        assert self.optimizer in (0, 1), "optimizer must be 0 (not being run) or 1 (will be run)"
         
                    

    def time_grid(self, pulse_time):
        """
        Build a uniform time grid for a given pulse duration.
        Returns (times, dt).
        """
        times = np.linspace(0, pulse_time, self.resolution)
        dt = times[1] - times[0]
        return times, dt


class Hamiltonians:
    """
    Provides Hamiltonian constructors given pulse parameters.
    """
    def __init__(self, params: PulseParameters):
        self.params = params

    def H11(self, phase_i, omega_scale: float = 1.0):
        """Two-atom Hamiltonian at a given phase (Hermitian)."""
        Omega1 = omega_scale * np.exp(1j * phase_i) / 2
        Delta1 = self.params.Delta1/self.params.Omega_Rabi1
        if self.params.optimizer == 0:
            Stark1  = self.params.Stark1/self.params.Omega_Rabi1
            Stark2  = self.params.Stark2/self.params.Omega_Rabi1
        else:
            Stark1 = 0
            Stark2 = 0
        if self.params.blockade_inf == 1:
            H = self.H01(phase_i, omega_scale=omega_scale)
        else:
            B = self.params.normalized_blockade
            H = np.array([
                [0,               Omega1,          Omega1,          0],
                [np.conj(Omega1), Delta1+Stark1,   0,               Omega1],
                [np.conj(Omega1), 0,               Delta1+Stark2,   Omega1],
                [0,               np.conj(Omega1), np.conj(Omega1), 2*Delta1+Stark1+Stark2+B],
         
            ], complex)
            decay_matrix = np.diag([0, self.params.decay_rate, self.params.decay_rate2, self.params.decay_rate + self.params.decay_rate2])
            H += (-1j * decay_matrix / 2)
        return H

    def H01(self, phase_i, omega_scale: float = 1.0):
        """Single-atom Hamiltonian at a given phase (Hermitian)."""
        Omega1 = omega_scale * np.exp(1j * phase_i) / 2
        Delta1 = self.params.Delta1/self.params.Omega_Rabi1

        H = np.array([
            [0,               Omega1],
            [np.conj(Omega1), Delta1],
        ], complex)
        decay_matrix = np.diag([0, self.params.decay_rate])
        H += (-1j * decay_matrix / 2) * np.eye(H.shape[0])
        return H
    

class FidelityCalculator:
    """
    Computes fidelities for given quantum states.
    """
    @staticmethod
    def bell_state_fidelity(psi01, psi11):
        """Compute Bell state fidelity from single- and two-atom amplitudes."""
        return (1/16) * np.abs(1 + 2 * psi01[0] - psi11[0])**2 #bell state fidelity
        
    

class CosineAnsatz:
    def generate(self, params: PulseParameters, inputs: np.ndarray):
        # unpack
        pulse_time, drive_detuning = inputs[0], inputs[1]
        A1, f1, o1 = inputs[2:]
        
        N = params.resolution
        t_grid = np.linspace(0, pulse_time, N)
        dt = pulse_time / (N-1)
        print(A1,f1,o1,pulse_time,drive_detuning)
        # build each cosine
        cos1 = np.cos((t_grid - pulse_time/2) * (f1 / params.Omega_Rabi1) - o1)
        # cos2 = np.cos((t_grid - pulse_time/2) * (f2 / params.Omega_Rabi1) - o2)
        # cos3 = np.cos((t_grid - pulse_time/2) * (f3 / params.Omega_Rabi1) - o3)
        # cos4 = np.cos((t_grid - pulse_time/2) * (f4 / params.Omega_Rabi1) - o4)

        # assemble phases
        phases = (drive_detuning / params.Omega_Rabi1) * t_grid \
               + A1 * cos1 \
               #+ A2 * cos2 \
               #+ A3 * cos3 \
              # + A4 * cos4 \

        return phases, dt


class GateSimulator:
    """
    Evolves the two-level system under a given phase profile and returns fidelity.
    """
    def __init__(self,
                 params: PulseParameters,
                 hams: Hamiltonians,
                 fidelity_calc: FidelityCalculator
                 ):
        self.params = params
        self.hams   = hams
        self.fid    = fidelity_calc

    def run(self, phases: np.ndarray, dt: float, omega_scale: float = 1.0) -> float:
        psi01 = self.params.initial_psi01.copy()
        psi11 = self.params.initial_psi11.copy()
        for phi in phases:
            U01 = scipy.linalg.expm(-1j*self.hams.H01(phi, omega_scale=omega_scale)*dt)
            U11 = scipy.linalg.expm(-1j*self.hams.H11(phi, omega_scale=omega_scale)*dt)
            psi01 = U01 @ psi01
            psi11 = U11 @ psi11
        # remove global phase
        gp = psi01[0]/np.abs(psi01[0])
        psi01 /= gp
        psi11 /= gp**2
        return [self.fid.bell_state_fidelity(psi01, psi11),gp] #returns [fidelity, rotation_phase]
        
    def run_and_rotate(self, phases: np.ndarray, dt: float, roatation_phase: float, omega_scale: float = 1.0) -> float:
        psi01 = self.params.initial_psi01.copy()
        psi11 = self.params.initial_psi11.copy()
        for phi in phases:
            U01 = scipy.linalg.expm(-1j*self.hams.H01(phi, omega_scale=omega_scale)*dt)
            U11 = scipy.linalg.expm(-1j*self.hams.H11(phi, omega_scale=omega_scale)*dt)
            psi01 = U01 @ psi01
            psi11 = U11 @ psi11
        # remove global phase
        gp = roatation_phase
        psi01 /= gp
        psi11 /= gp**2
        return self.fid.bell_state_fidelity(psi01, psi11)



class CRABOptimizer:
    """
    Wraps a CosineAnsatz and GateSimulator to tune ansatz parameters via SciPy.
    """
    def __init__(self,
                 ansatz: CosineAnsatz,
                 simulator: GateSimulator
                 ):
        self.ans   = ansatz
        self.sim   = simulator

    def loss(self, inputs: np.ndarray) -> float:
        phases, dt = self.ans.generate(self.sim.params, inputs)
        F = self.sim.run(phases, dt)[0]
        return 1 - F

    def optimize(self, init_inputs: np.ndarray):
        def cb(xk):
            inf = self.loss(xk)
            print(f"Infidelity: {inf:.6f}", end="\r", flush=True)
        bounds = [(1e-8, None)] + [(None, None)] * (len(init_inputs) - 1)
        res = minimize(self.loss, init_inputs, callback=cb, bounds=bounds, options={'gtol':1e-6})
        return res.fun, res.x


# Find the optimal parameters
def optimize_gates(rabi, cases):
    #cases is the case dictionary set up before
    
    best_params = []

    B = cases["ideal_blockade"]
    Lifetime = cases["lifetime"]
    Lifetime2 = cases.get("lifetime2", Lifetime)
    resolution = 200
    Stark1 = 0
    Stark2 = 0
    # set up (Omega_Rabi1, blockade_inf, blockade, r_lifetime, delta1, Stark1, Stark2, resolution, optimizer, Lifetime2)
    params        = PulseParameters(rabi, 0, B, Lifetime, 0, Stark1, Stark2, resolution, 1, Lifetime2)
    hams          = Hamiltonians(params)
    fid_calc      = FidelityCalculator()
    ansatz        = CosineAnsatz()
    simulator     = GateSimulator(params, hams, fid_calc)

    # initial guess: [pulse_time, drive_detuning, (A_i,f_i,o_i)] Amplitudes, Frequencies, phi_Offsets
    init = start_params[(atom_name, atom_n, r_idx)]['best_x']
    print("Setting best known parameters to ", init)
    optimizer = CRABOptimizer(ansatz, simulator)
    best_inf, best_x = optimizer.optimize(np.array(init))
    print("That fidelity was", 1-best_inf)
    
            #pulse_time, gamma, A, w, phi
    LP_params = [7.65,0,2*np.pi*0.1122, 1.0431, -0.7318]
    init = LP_params
    print("Setting best known parameters to ", init)
    optimizer = CRABOptimizer(ansatz, simulator)
    a_inf, a = optimizer.optimize(np.array(init))
    print("That fidelity was", 1-a_inf)
    
    pulse_time = best_x[0]
    times_crab = np.linspace(0, pulse_time, params.resolution)   # same grid as best CRAB
    best_phase, dt = ansatz.generate(params, best_x)
    
    #Sanity check
    R = len(best_phase)
    times_grape = np.linspace(0, pulse_time, R)
    dt = times_grape[1]-times_grape[0]
    simulator = GateSimulator(params, hams, fid_calc)
    infidelity_test = 1 - simulator.run(best_phase, dt)[0]
    print(f"\nInfidelity via GateSimulator: {infidelity_test:.6e}")

    print("\nOptimal pulse time: ", pulse_time, "Omega * t")
    print("Converged parameters ", best_x)
    
    params.optimizer = 0
    #params.r_lifetime = Lifetime
    params.Stark1 = Stark1
    params.Stark2 = Stark2
    params.recompute()
    infid = 1 - simulator.run(best_phase, dt)[0]
    print(f"Stark1 = {Stark1}, Stark2 = {Stark2}, τ = {Lifetime}  →  infidelity = {infid:.4e}")
    print("________________________________________________")

    cases["best_x"] = best_x
    cases["rotation_phase"] = simulator.run(best_phase, dt)[1]
    cases["F_base"] = infid

#optimize_gates(rabi_rate, cases)

def simulate_gate_F(B, best_atom_params, roatation_phase, stark1, stark2, lifetime, lifetime2, delta_dc: float = 0.0, omega_scale: float = 1.0):
    #Simulates the proper gate fildelity, considering the rotation to the Bell state
    Rabi = rabi_rate

    resolution = 200
    
    # set up (Omega_Rabi1, blockade_inf, blockade, r_lifetime, delta1, Stark1, Stark2, resolution, optimizer, Lifetime2)
    params = PulseParameters(Rabi, 0, B, lifetime, delta_dc, stark1, stark2, resolution, 0, lifetime2)
    hams = Hamiltonians(params)
    fid_calc = FidelityCalculator()
    simulator = GateSimulator(params, hams, fid_calc)
    ansatz = CosineAnsatz()

    pulse_time = best_atom_params[0]
    times_crab = np.linspace(0, pulse_time, params.resolution)
    best_phase, _ = ansatz.generate(params, best_atom_params)
    R = len(best_phase)
    # print(R)
    times_grape = np.linspace(0, pulse_time, R)
    dt = times_grape[1]-times_grape[0]
    # print('dt', len(times_grape), best_phase, len(best_phase))
    np.save('phases', best_phase)
    np.save('time', times_grape)
    fid, gp =  simulator.run(best_phase, dt, omega_scale=omega_scale)
    infid = 1-fid
    return infid
	

#Define functions that give atom distributions in the trap
def sigma_r_um(atom_T_uK, U_trap_max_uK, w0_um):
    return (0.5) * np.sqrt(atom_T_uK / U_trap_max_uK)
def sigma_z_um(atom_T_uK, U_trap_max_uK, w0_um, wavelength_um):
    zR = np.pi * w0_um**2 / wavelength_um
    return zR * np.sqrt(atom_T_uK / (2 * U_trap_max_uK))
	
#Actually do the Monte Carlo
def monte_carlo(shots, cases, temps=(5, 10), E_fields=(5, 10), detuning_sigmas_MHz=(), rabi_frac_sigmas=()):
    # temps: atom temperature(s) in uK (position-induced blockade fluctuations)
    # E_fields: RMS DC E-field noise in mV/cm (DC Stark shift)
    # detuning_sigmas_MHz: RMS additional common-mode detuning noise in MHz (laser/B-field drift, Doppler, etc.)
    # rabi_frac_sigmas: RMS fractional Rabi noise (dimensionless), e.g. 0.01 for 1% shot-to-shot
    
    best_atom_params = cases["best_x"]
    lifetime = cases["lifetime"]
    lifetime2 = cases.get("lifetime2", lifetime)
    rotation_phase = cases["rotation_phase"]
    
    #set up the dictionary for the results to live in
    cases["F_pos"] = {}
    cases["F_E"] = {}
    cases["F_detuning"] = {}
    cases["F_rabi"] = {}
    for T_atom in temps:
        cases["F_pos"][str(T_atom)] = []
    for E_field in E_fields:
        cases["F_E"][str(E_field)] = []
    for sMHz in detuning_sigmas_MHz:
        cases["F_detuning"][str(sMHz)] = []
    for srel in rabi_frac_sigmas:
        cases["F_rabi"][str(srel)] = []
    print("Running the Monte Carlo")
    for i in range(shots):
        #Monte Carlo on the position variation
        for T_atom in temps:
            sep = sample_pair_distances(n_samples=1,sigma_r=sigma_r_um(atom_T_uK=T_atom, U_trap_max_uK=500, w0_um=1.),sigma_z=sigma_z_um(atom_T_uK=T_atom, U_trap_max_uK=500, w0_um=1., wavelength_um=1.064),x_offset=atom_separation_um,rng=None)[0][0]
            B = 2 * np.pi * 1e3 * get_nearest_blockade(atom_name, int(atom_n), sep, blockade_dict)
            stark1 = 0
            stark2 = 0
            F_single = simulate_gate_F(B, best_atom_params, rotation_phase, stark1, stark2, lifetime, lifetime2)
            cases["F_pos"][str(T_atom)].append(F_single)
        
        #Monte Carlo on the Stark        
        for E_field in E_fields:
            B = cases["ideal_blockade"]
            E = sample_gaussian(E_field)
            stark1 = 2 * np.pi * light_shift_MHz(atom_name, atom_n, E)
            stark2 = stark1
            F_single = simulate_gate_F(B, best_atom_params, rotation_phase, stark1, stark2, lifetime, lifetime2)
            cases["F_E"][str(E_field)].append(F_single)

        # Monte Carlo on additional common-mode detuning (shot-to-shot DC)
        for sMHz in detuning_sigmas_MHz:
            B = cases["ideal_blockade"]
            stark1 = 0
            stark2 = 0
            delta_dc = 2 * np.pi * sample_gaussian(sMHz)  # MHz -> rad/us (since 1 MHz = 1/us)
            F_single = simulate_gate_F(B, best_atom_params, rotation_phase, stark1, stark2, lifetime, lifetime2,
                                      delta_dc=delta_dc, omega_scale=1.0)
            cases["F_detuning"][str(sMHz)].append(F_single)

        # Monte Carlo on shot-to-shot Rabi fluctuations (DC)
        for srel in rabi_frac_sigmas:
            B = cases["ideal_blockade"]
            stark1 = 0
            stark2 = 0
            eps = sample_gaussian(srel)
            omega_scale = 1.0 + eps
            F_single = simulate_gate_F(B, best_atom_params, rotation_phase, stark1, stark2, lifetime, lifetime2,
                                      delta_dc=0.0, omega_scale=omega_scale)
            cases["F_rabi"][str(srel)].append(F_single)


def convert_to_experiment_params(A,w,phi,gamma,T,Omega):
    A_exp = A
    w_exp = w/Omega
    phi_exp = w*T/Omega/2+phi
    gamma_exp = gamma/Omega
    return A_exp, w_exp, phi_exp, gamma_exp

best_atom_params = start_params[(atom_name, atom_n,r_idx)]["best_x"]
lifetime = cases["lifetime"]
lifetime2 = cases.get("lifetime2", lifetime)
rotation_phase = start_params[(atom_name, atom_n,r_idx)]["rotation_phase"]

# B = 2 * np.pi * 1e3 * get_nearest_blockade(atom_name, int(atom_n), 2.5, blockade_dict)
# F_single = simulate_gate_F(B, best_atom_params, rotation_phase, 0, 0, lifetime, lifetime2)
# print("With parameters", best_atom_params, "\n Fidelity =", 1-F_single)
#pulse_time, gamma, A, w, phi
[pulse_time, gamma, A, w, phi]= [7.65,0,2*np.pi*0.1122, 1.0431, -0.7318]
best_atom_params = [pulse_time, gamma*rabi_rate, A, w*rabi_rate, phi-w*pulse_time/2]
B = 1000000
F_single = simulate_gate_F(B, best_atom_params, rotation_phase, 0, 0, lifetime, lifetime2)
print("With parameters", best_atom_params, "\n Fidelity =", 1-F_single)
