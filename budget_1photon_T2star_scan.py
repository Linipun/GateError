"""
1-photon CZ gate error budget, parametrised by a MEASURED T2* instead of by
individual frequency-noise sources.

This is budget_1photon_scan.py with every *detuning* (frequency) noise channel
    - laser frequency noise (phase-noise PSD)
    - DC E-field fluctuation (quadratic Stark)
    - broadband E-field fluctuation
    - DC B-field fluctuation (Zeeman)
    - Doppler shift from atomic motion
replaced by ONE channel derived from the Ramsey coherence time T2* of the
|1> - |r> transition.  Every non-detuning channel (finite blockade / TO,
intensity noise, Rabi inhomogeneity, blockade fluctuation from atomic motion,
Rydberg decay, mj leakage) is untouched and computed exactly as before.

Noise model
-----------
'gaussian' (default): quasi-static inhomogeneous dephasing.  The detuning is
    constant during one gate but Gaussian-distributed shot to shot,
        C(t) = <e^{i d t}> = exp(-sigma_d^2 t^2 / 2) == exp(-(t/T2*)^2)
        =>  sigma_d = sqrt(2) / T2*        [rad/us, T2* in us]
    This is what a Gaussian-decaying Ramsey fringe measures, and it is the
    right model when the noise is slow compared with the gate (Doppler, DC
    field drift, slow laser drift).  The gate infidelity is then the average of
    the exact gate simulation over that Gaussian, done by Gauss-Hermite
    quadrature (exact to machine precision for the smooth quadratic-in-delta
    infidelity, and far cheaper than Monte Carlo).

'white': Markovian dephasing, C(t) = exp(-t/T2*), i.e. a flat detuning PSD
    S_delta = 4/T2* [rad^2/us] (one-sided).  Fast noise does NOT act quasi-
    statically, so this branch integrates the universal linear-response
    function I(omega) over the band instead of averaging the simulation.

    CAVEAT (checked numerically, see below): linear_response.I(omega) is the
    HAAR-AVERAGED gate infidelity response, while every Monte-Carlo channel in
    this budget (and in budget_1photon_scan.py) uses the Bell-state fidelity
    of budget_monte_carlo.Hamiltonians.  The two metrics differ by an O(1),
    Omega-dependent factor -- measured here as I(0)/Omega^2 divided by the
    exact d^2(infidelity)/d(nu)^2:  0.89 at Omega/2pi = 1.5 MHz and 2.6 at
    10 MHz.  So the 'white' branch takes only the SHAPE of I(omega) from the
    linear response and normalises it at omega -> 0 to the exact quasi-static
    curvature of this budget's own metric (t2_white_calibrate=True).  The raw,
    uncalibrated linear-response number is still reported for reference.

Correlation between the two atoms
---------------------------------
't2_correlation':
    'common'      - the same detuning on both atoms (shared laser / global
                    field drift).  |rr> then sees 2*delta; errors add
                    coherently.  This is what budget_1photon_scan.py assumed.
    'independent' - each atom draws its own detuning (Doppler, local field
                    gradients, per-site light shifts).
Both are implemented exactly.  For this gate the choice barely matters: the
two-atom cross-correlation term turns out to be small, so 'common' and
'independent' give the same channel to within a few percent (7.73e-4 vs
7.43e-4 at Omega/2pi = 1.5 MHz, T2* = 10 us).  'common' is the default.

Usage:  python budget_1photon_T2star_scan.py "70, 4.0"
        python budget_1photon_T2star_scan.py "70, 4.0, 15"     # T2* = 15 us
"""

from budget_monte_carlo import *
from linear_response import (build_Oseq, response_G13, isometry_haar_full,
                             propagate_U, O_nu, idx, dim)
import numpy as np
import matplotlib.pyplot as plt
from arc import *
import json
import scipy.optimize as opt

import sys

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


result = 'result_T2star'
os.makedirs(result, exist_ok=True)


#### config #######
atom_name = 'Cs'
arg = eval('[' + sys.argv[1] + ']')
l = 1
j = 3/2
mj = 3/2
pulse_time = 7.65  # Omega_Rabi
resolution = 200   # number of phase steps in the pulse
lambda_rydberg = 0.319  # um
# Effective detuning of the nP3/2 mj=1/2 leakage state from the mj=3/2 gate state, in rad/us.
# Not a hyperfine splitting: a pi-polarised microwave resonantly dresses mj=1/2 with (n+1)S1/2
# mj=1/2 (mj=3/2 is untouched -- S1/2 has no mj=3/2 sublevel) and pushes it out of resonance.
# NOTE this must be the effective detuning Delta_eff = [sum_k |<P|k>|^2 / delta_k^2]^(-1/2)
# over the *dressed* branches k, NOT Omega_mw/2.
# Set to None for the undressed (no-microwave) limit, i.e. bare Zeeman splitting only.
HF_split = 2000*np.pi*2  # rad/us
HF_split = None
T_atom = 15        # uK
lambda_trap = 1.064  # um
w0_trap = 1.2      # um
num_samples = 10000  # motional (blockade / Rabi inhomogeneity) Monte-Carlo samples
rin_strength = 1e-3  # integrated relative intensity noise <(dI/I)^2>

# ---- the T2* channel (replaces ALL frequency-noise channels) ----
T2_star_us = 10.0            # measured Ramsey T2* of the |1>-|r> transition [us]
t2_model = 'gaussian'        # 'gaussian' (quasi-static) | 'white' (Markovian)
t2_correlation = 'common'    # 'common' | 'independent'
t2_average = 'gauss_hermite'  # 'gauss_hermite' | 'mc'   (gaussian model only)
t2_gh_nodes = 11             # Gauss-Hermite nodes per atom (converged by n=5)
t2_mc_samples = 4000         # only used if t2_average == 'mc'
t2_white_fmax_over_Omega = 10.0  # integrate I(omega) out to this multiple of Omega/2pi
t2_white_points = 200        # frequency grid points for the 'white' band integral
t2_white_calibrate = True    # normalise I(omega) at DC to this budget's own metric
#### config #######


### parameters ####
n = int(arg[0])
atom_d = arg[1]  # um
Bz = 10          # G
w0_rydberg = 7.5  # um
trap_depth = 450  # uK
if len(arg) > 2:  # optional CLI override of T2*
    T2_star_us = float(arg[2])
### parameters ####


f_Rabis = np.linspace(2, 40, 20)
f_Rabis = np.array([1.5])


if atom_name == "Rb":
    atom = Rubidium()
elif atom_name == "Cs":
    atom = Caesium()
blockade_mrad = find_blockade_Mrad(atom_name, n, atom_d)
print('Blockade:', blockade_mrad/2/np.pi, 'MHz')
R_lifetime = atom.getStateLifetime(n=n, l=l, j=j, temperature=300, includeLevelsUpTo=n+20, s=0.5)*1e6
m_atom = atom.mass
if HF_split is None:
    HF_split = (atom.getZeemanEnergyShift(l=1, j=3/2, mj=3/2, magneticFieldBz=Bz/10000) -
                atom.getZeemanEnergyShift(l=1, j=3/2, mj=1/2, magneticFieldBz=Bz/10000))/hbar/1e6


# ---------------------------------------------------------------------------
# T2* -> detuning statistics
# ---------------------------------------------------------------------------
# Quasi-static Gaussian: C(t) = exp(-sigma_d^2 t^2/2) = exp(-(t/T2*)^2)
sigma_delta = np.sqrt(2.0) / T2_star_us      # rad/us, per atom
sigma_nu = sigma_delta / (2*np.pi)           # MHz  (units the response function wants)
# Markovian white: C(t) = exp(-t/T2*)  <=>  one-sided S_delta = 4/T2* [rad^2/us]
S_nu_white = 4.0 / ((2*np.pi)**2 * T2_star_us)   # MHz^2/MHz, one-sided

print(f'T2* = {T2_star_us} us  ({t2_model}, {t2_correlation})')
if t2_model == 'gaussian':
    print(f'  -> quasi-static detuning sigma = {sigma_delta:.4f} rad/us '
          f'= 2pi x {sigma_nu*1e3:.1f} kHz  (per atom)')
else:
    print(f'  -> white detuning PSD S_nu = {S_nu_white*1e6:.3f} Hz^2/Hz (one-sided)')

# Diagnostic only: what T2* would the Doppler shift alone give at this temperature?
# k_eff = 2pi/lambda for the single 319 nm photon; sigma_delta_doppler = k*sqrt(kB T/m).
doppler_sigma_rad_us = 2*np.pi/(lambda_rydberg*1e-6)*np.sqrt(kb*(T_atom*1e-6)/m_atom)/1e6
T2_doppler_only = np.sqrt(2.0)/doppler_sigma_rad_us
print(f'  (for reference: Doppler alone at {T_atom} uK would give T2* = {T2_doppler_only:.1f} us)')


# ---------------------------------------------------------------------------
# Independent (uncorrelated) per-atom detunings
# ---------------------------------------------------------------------------
class IndepDetuningHamiltonians(Hamiltonians):
    """
    Two-atom Hamiltonian with an *independent* Rydberg detuning on each atom.

    The base class only carries one common Delta1, but its Stark1/Stark2 slots
    already enter the two-atom diagonal exactly as per-atom detunings would:
        diag(H11) = [0, Delta1+Stark1, Delta1+Stark2, 2*Delta1+Stark1+Stark2]
    so setting Delta1 = 0, Stark1 = d1, Stark2 = d2 gives the right |r0>, |0r>,
    |rr> energies.  Only the single-excitation sectors need overriding, because
    the base class propagates both of them with the same Delta1.

    Convention (matches asym_return_fidel): the atom driven by Omega1 is the one
    carrying Stark1 = d1 and whose single-atom sector is psi01.
    """

    def __init__(self, *, delta1, delta2, **kwargs):
        kwargs['Delta1'] = 0.0
        kwargs['Stark1'] = delta1
        kwargs['Stark2'] = delta2
        super().__init__(**kwargs)
        self.delta1 = delta1
        self.delta2 = delta2

    def H01_atom(self, phase_i, delta, omega_scale: float = 1.0):
        Omega1 = omega_scale * np.exp(1j * phase_i) / 2
        d = delta / self.Omega_Rabi1
        H = np.array([
            [0, Omega1],
            [np.conj(Omega1), d],
        ], complex)
        H += np.diag([0, -1j * self.decay_rate / 2])
        return H

    def return_fidel_indep(self, phases=None, dt=None):
        psi01 = self.initial_psi01.copy()
        psi10 = self.initial_psi01.copy()
        psi11 = self.initial_psi11.copy()
        for phi in phases:
            psi01 = scipy.linalg.expm(-1j * self.H01_atom(phi, self.delta1) * dt) @ psi01
            psi10 = scipy.linalg.expm(-1j * self.H01_atom(phi, self.delta2) * dt) @ psi10
            psi11 = scipy.linalg.expm(-1j * self.H11(phi) * dt) @ psi11
        gp1 = psi01[0] / np.abs(psi01[0])
        gp2 = psi10[0] / np.abs(psi10[0])
        psi01 /= gp1
        psi10 /= gp2
        psi11 /= gp1
        psi11 /= gp2
        return self.bell_state_fidelity(psi10, psi01, psi11), (gp1, gp2)


# ---------------------------------------------------------------------------
# Per-atom detuning response operators (for the 'independent' linear response)
# ---------------------------------------------------------------------------
def O_nu_atom(which: int) -> np.ndarray:
    """
    Detuning-noise operator for ONE atom, same normalisation as
    linear_response.O_nu (a factor -2pi per Rydberg excitation, so the noise
    variable is a frequency in MHz).  O_nu_atom(1) + O_nu_atom(2) == O_nu().
    """
    O = np.zeros((dim, dim), dtype=complex)
    states = ["r0", "r1", "rr"] if which == 1 else ["0r", "1r", "rr"]
    for k in states:
        O[idx[k], idx[k]] += -2 * np.pi
    return O


def build_Oseq_op(phases, dt, B, Oinst) -> np.ndarray:
    """Heisenberg sequence for a fixed (time-independent) noise operator."""
    Us = propagate_U(phases=phases, dt=dt, B=B)
    Udag = np.conjugate(np.swapaxes(Us, 1, 2))
    return np.einsum("tij,jk,tkl->til", Udag, Oinst, Us)


## Linear Response ####
S_haar = isometry_haar_full()  # D=4

TO_1 = []
RIN_1photon = []
T2_1photon = []          # the single channel that replaces all frequency noises
T2_1photon_lr = []       # linear-response cross-check of the same channel
decay_1photon = []
infids_motion_blockade1 = []
infids_motion_rabi1 = []
leakage1 = []
scattering1 = []
t2_lr_coeff = []         # I(0)/Omega^2 [1/MHz^2], linear response (Haar metric)
t2_lr_integral = []      # int I(w)/Omega^2 df [1/MHz], for the white model
t2_c2_exact = []         # exact d^2(infid)/d(nu)^2 [1/MHz^2], this budget's metric
t2_scale_A = []          # infid_T2 = t2_scale_A / T2*^t2_scale_power
t2_scale_power = 1 if t2_model == 'white' else 2

Omega_Rabis = 2 * np.pi * f_Rabis
sigma_r = sigma_r_um(T_atom, trap_depth, w0_trap)
sigma_z = sigma_z_um(T_atom, trap_depth, w0_trap, lambda_trap)
ds, c1, c2, = sample_pair_distances(
    n_samples=num_samples,
    sigma_r=sigma_r,
    sigma_z=sigma_z,
    x_offset=atom_d,
    rng=None
)
x1 = c1['x']
y1 = c1['y']
z1 = c1['z']
x2 = c2['x']
y2 = c2['y']
z2 = c2['z']
rabi = np.sqrt(relative_gaussian_beam_intensity(0, 0, 0-atom_d/2, w0_rydberg, lambda_rydberg))
rabis1 = np.sqrt(relative_gaussian_beam_intensity(x1, z1, y1-atom_d/2, w0_rydberg, lambda_rydberg))/rabi
rabis2 = np.sqrt(relative_gaussian_beam_intensity(x2-atom_d, z2, y2+atom_d/2, w0_rydberg, lambda_rydberg))/rabi

# Gauss-Hermite nodes/weights for <f>_{N(0,sigma^2)} = (1/sqrt(pi)) sum_i w_i f(sqrt(2) sigma x_i)
gh_x, gh_w = np.polynomial.hermite.hermgauss(t2_gh_nodes)

for Omega_Rabi in Omega_Rabis:
    print('Omega:', Omega_Rabi / 2 / np.pi)
    H_gen1 = Hamiltonians(Omega_Rabi1=Omega_Rabi, blockade_inf=False, blockade=blockade_mrad, r_lifetime=R_lifetime,
                          Delta1=0,
                          Stark1=0, Stark2=0, resolution=resolution, r_lifetime2=R_lifetime, pulse_time=pulse_time)
    PhaseGuess = [2 * np.pi * 0.1122, 1.0431, -0.7318, 0]
    time, phase_guess, dt = phase_cosine_generate(*PhaseGuess, H_gen1.pulse_time, H_gen1.resolution)
    fid1, global_phi = H_gen1.return_fidel(phases=phase_guess, dt=dt)
    opt_out = opt.minimize(fun=fid_optimize, x0=PhaseGuess, args=(H_gen1))
    phase_params1 = opt_out.x
    infid_TO_theory = opt_out.fun
    H_gen1 = Hamiltonians(Omega_Rabi1=Omega_Rabi, blockade_inf=False, blockade=blockade_mrad, r_lifetime=1e10, Delta1=0,
                          Stark1=0, Stark2=0, resolution=resolution, r_lifetime2=1e10, pulse_time=pulse_time)
    time, phase, dt = phase_cosine_generate(*phase_params1, H_gen1.pulse_time, H_gen1.resolution)
    fid1, global_phi = H_gen1.return_fidel(phases=phase, dt=dt)
    infid_TO1 = 1 - fid1
    TO_1.append(infid_TO1)

    o_f = build_Oseq(phases=phase, dt=dt, B=blockade_mrad/Omega_Rabi, is_intensity=False)
    o_I = build_Oseq(phases=phase, dt=dt, B=blockade_mrad/Omega_Rabi, is_intensity=True)

    Ii = response_G13(o_I, S_haar, 0, dt=dt)
    RIN_contribution = Ii * rin_strength
    RIN_1photon.append(RIN_contribution)

    # ------------------------------------------------------------------
    # T2* channel
    # ------------------------------------------------------------------
    # Linear-response coefficients.  The noise variable of O_nu is a frequency
    # in MHz (O_nu carries the 2pi), and time is in units of 1/Omega, so
    # I(omega)/Omega^2 multiplied by a detuning variance in MHz^2 is an
    # infidelity.  For 'common' noise the operator is O_nu (both atoms, |rr>
    # weighted twice); for 'independent' noise the two atoms are uncorrelated,
    # so their variances - not their operators - add.
    if t2_correlation == 'independent':
        o_f1 = build_Oseq_op(phases=phase, dt=dt, B=blockade_mrad/Omega_Rabi, Oinst=O_nu_atom(1))
        o_f2 = build_Oseq_op(phases=phase, dt=dt, B=blockade_mrad/Omega_Rabi, Oinst=O_nu_atom(2))
        I0 = (response_G13(o_f1, S_haar, 0, dt=dt)
              + response_G13(o_f2, S_haar, 0, dt=dt)) / Omega_Rabi ** 2
        o_f_list = [o_f1, o_f2]
    else:
        I0 = response_G13(o_f, S_haar, 0, dt=dt) / Omega_Rabi ** 2
        o_f_list = [o_f]
    t2_lr_coeff.append(I0)

    # band integral of the same response, needed for the 'white' model
    f_max = t2_white_fmax_over_Omega * Omega_Rabi / (2 * np.pi)   # MHz
    fs_grid = np.linspace(f_max / t2_white_points, f_max, t2_white_points)
    df_grid = fs_grid[1] - fs_grid[0]
    J = 0.0
    for fE in fs_grid:
        omg = 2 * np.pi * fE / Omega_Rabi          # normalised noise frequency omega/Omega
        J += sum(response_G13(of, S_haar, omg, dt=dt) for of in o_f_list) / Omega_Rabi ** 2 * df_grid
    t2_lr_integral.append(J)

    # --- exact quasi-static average of the full gate simulation ---
    def _infid_common(d):
        H = Hamiltonians(Omega_Rabi1=Omega_Rabi, blockade_inf=False, blockade=blockade_mrad,
                         r_lifetime=10e9, r_lifetime2=10e9, Delta1=d, Stark1=0, Stark2=0,
                         resolution=resolution, pulse_time=pulse_time)
        return 1 - H.return_fidel(phases=phase, dt=dt)[0]

    def _infid_indep(d1, d2):
        H = IndepDetuningHamiltonians(delta1=d1, delta2=d2,
                                      Omega_Rabi1=Omega_Rabi, blockade_inf=False,
                                      blockade=blockade_mrad, r_lifetime=10e9, r_lifetime2=10e9,
                                      resolution=resolution, pulse_time=pulse_time)
        return 1 - H.return_fidel_indep(phases=phase, dt=dt)[0]

    eps_d = 0.002 * Omega_Rabi                 # small probe detuning [rad/us]
    eps_nu = eps_d / (2 * np.pi)               # in MHz, the response variable
    if t2_correlation == 'independent':
        if t2_average == 'mc':
            rng = np.random.default_rng()
            d1s = rng.normal(0.0, sigma_delta, t2_mc_samples)
            d2s = rng.normal(0.0, sigma_delta, t2_mc_samples)
            ws = np.full(t2_mc_samples, 1.0 / t2_mc_samples)
        else:
            nodes = np.sqrt(2.0) * sigma_delta * gh_x
            d1s = np.repeat(nodes, t2_gh_nodes)
            d2s = np.tile(nodes, t2_gh_nodes)
            ws = np.outer(gh_w, gh_w).ravel() / np.pi
        infid_avg = sum(w * _infid_indep(d1, d2) for d1, d2, w in zip(d1s, d2s, ws))
        # small-detuning curvature d^2(infid)/d(nu^2), summed over the two
        # (uncorrelated) atoms -- this is the exact analogue of I(0)/Omega^2
        c2_curv = ((_infid_indep(eps_d, 0) + _infid_indep(-eps_d, 0) - 2 * infid_TO1)
                   + (_infid_indep(0, eps_d) + _infid_indep(0, -eps_d) - 2 * infid_TO1)) / (2 * eps_nu ** 2)
    else:
        if t2_average == 'mc':
            rng = np.random.default_rng()
            dsamp = rng.normal(0.0, sigma_delta, t2_mc_samples)
            ws = np.full(t2_mc_samples, 1.0 / t2_mc_samples)
        else:
            dsamp = np.sqrt(2.0) * sigma_delta * gh_x
            ws = gh_w / np.sqrt(np.pi)
        infid_avg = sum(w * _infid_common(d) for d, w in zip(dsamp, ws))
        c2_curv = (_infid_common(eps_d) + _infid_common(-eps_d) - 2 * infid_TO1) / (2 * eps_nu ** 2)
    infid_qs = infid_avg - infid_TO1                 # quasi-static (gaussian) result
    c2_anchor = infid_qs / sigma_nu ** 2             # passes exactly through this point
    infid_T2_lr = I0 * sigma_nu ** 2                 # raw linear response, Haar metric

    if t2_model == 'white':
        # Markovian: fast noise cannot be averaged quasi-statically, so use the
        # band integral of the universal response against the flat PSD, with
        # the DC value renormalised to this budget's own fidelity metric.
        cal = (c2_curv / I0) if t2_white_calibrate else 1.0
        infid_T2 = J * S_nu_white * cal
        # infid(T2*) = A/T2*, from S_nu = 4/((2pi)^2 T2*)
        scale_A = J * cal * 4.0 / (2 * np.pi) ** 2
    else:
        infid_T2 = infid_qs
        # infid(T2*) = A/T2*^2, from sigma_nu^2 = 1/(2 pi^2 T2*^2)
        scale_A = c2_anchor / (2 * np.pi ** 2)

    T2_1photon.append(infid_T2)
    T2_1photon_lr.append(infid_T2_lr)
    t2_scale_A.append(scale_A)
    t2_c2_exact.append(c2_curv)
    print(f'  T2* channel: {infid_T2:.3e}   '
          f'[exact curvature {c2_curv:.4g}/MHz^2 vs linear-response {I0:.4g}/MHz^2, '
          f'metric factor {I0/c2_curv:.2f}]')

    infids_decay = (2.95 / (Omega_Rabi)) / R_lifetime
    decay_1photon.append(infids_decay)

    # ------------------------------------------------------------------
    # Motional channels (unchanged): blockade fluctuation + Rabi inhomogeneity
    # ------------------------------------------------------------------
    infids_s1 = []
    infids_blockade = []
    infids_rabi = []
    for rabi1, rabi2, d in zip(rabis1, rabis2, ds):
        blockade = find_blockade_Mrad(atom_name, n, d)
        # total
        H_gen = Hamiltonians(Omega_Rabi1=Omega_Rabi, blockade_inf=False, blockade=blockade, r_lifetime=10e9, Delta1=0,
                             Stark1=0, Stark2=0, resolution=resolution, r_lifetime2=10e9, pulse_time=pulse_time)
        fid, global_phi = H_gen.asym_return_fidel(phases=phase, dt=dt, omega1_scale=rabi1, omega2_scale=rabi2)
        infids_s1.append(1 - fid)

        H_gen_block = Hamiltonians(Omega_Rabi1=Omega_Rabi, blockade_inf=False, blockade=blockade, r_lifetime=10e9,
                                   Delta1=0,
                                   Stark1=0, Stark2=0, resolution=resolution, r_lifetime2=10e9, pulse_time=pulse_time)
        fid, global_phi = H_gen_block.asym_return_fidel(phases=phase, dt=dt, omega1_scale=1, omega2_scale=1)
        infids_blockade.append(1 - fid)

        H_gen_rabi = Hamiltonians(Omega_Rabi1=Omega_Rabi, blockade_inf=False, blockade=blockade_mrad, r_lifetime=10e9,
                                  Delta1=0,
                                  Stark1=0, Stark2=0, resolution=resolution, r_lifetime2=10e9, pulse_time=pulse_time)
        fid, global_phi = H_gen_rabi.asym_return_fidel(phases=phase, dt=dt, omega1_scale=rabi1, omega2_scale=rabi2)
        infids_rabi.append(1 - fid)

    infids_s1 = np.asarray(infids_s1)
    infids_motion = np.mean(infids_s1) - infid_TO1

    infids_blockade = np.asarray(infids_blockade)
    infids_motion_blockade = np.mean(infids_blockade) - infid_TO1
    infids_motion_blockade_std = infids_blockade.std(ddof=1)

    infids_rabi = np.asarray(infids_rabi)
    infids_motion_rabi = np.mean(infids_rabi) - infid_TO1
    infids_motion_rabi_std = infids_rabi.std(ddof=1)

    infids_motion_rabi1.append(infids_motion_rabi)
    infids_motion_blockade1.append(infids_motion_blockade)

    H_gen1 = LeakageHamiltonians(Omega_Rabi1=Omega_Rabi, blockade_inf=False,
                                 blockade=blockade_mrad, r_lifetime=10e9,
                                 Delta1=0,
                                 Stark1=0, Stark2=0, resolution=resolution, r_lifetime2=10e9,
                                 pulse_time=pulse_time,
                                 mj12_split=HF_split)
    fid_with_leak, global_phi = H_gen1.return_fidel(phases=phase, dt=dt)
    leakage_mj = (1 - fid_with_leak) - infid_TO1
    leakage1.append(leakage_mj)

    # Off-resonant / photoionization scattering is NOT modelled; this is a placeholder, not a computed bound.
    scattering1.append(np.nan)
    total = (infid_TO1 + infids_motion_blockade + infids_motion_rabi + leakage_mj
             + infid_T2 + RIN_contribution + infids_decay)
    print('total error:', total)

RIN_1photon = np.array(RIN_1photon)
T2_1photon = np.array(T2_1photon)
T2_1photon_lr = np.array(T2_1photon_lr)
infids_motion_blockade1 = np.array(infids_motion_blockade1)
infids_motion_rabi1 = np.array(infids_motion_rabi1)
decay_1photon = np.array(decay_1photon)
leakage1 = np.array(leakage1)
TO_1 = np.array(TO_1)
t2_lr_coeff = np.array(t2_lr_coeff)
t2_lr_integral = np.array(t2_lr_integral)
t2_c2_exact = np.array(t2_c2_exact)
t2_scale_A = np.array(t2_scale_A)

# TO_1 is the intrinsic finite-blockade error; it is the baseline subtracted from every other
# channel, so it must be added back here. Scattering is not modelled and is excluded.
sum_1photon = (TO_1 + RIN_1photon + infids_motion_blockade1 + infids_motion_rabi1
               + decay_1photon + leakage1 + T2_1photon)
print('min infid:', min(sum_1photon), '(scattering not included)')

# What T2* would each target infidelity require, from this channel alone?
# infid_T2 = A / T2*^p  with p = 2 (gaussian) or 1 (white), A anchored on the
# exact simulation -- verified to reproduce the full calculation to <0.5% for
# T2* >= 5 us and to 3% at T2* = 2 us (where the quadratic-in-delta expansion
# starts to break down).
for target in (1e-3, 1e-4):
    req = (t2_scale_A / target) ** (1.0 / t2_scale_power)
    print(f'T2* needed for a {target:.0e} T2*-channel error: '
          + ', '.join(f'{f:.1f} MHz -> {r:.1f} us' for f, r in zip(f_Rabis, req)))
# and the T2* at which this channel alone exceeds everything else combined
rest_tot = sum_1photon - T2_1photon
cross = (t2_scale_A / rest_tot) ** (1.0 / t2_scale_power)
print('T2* at which the T2* channel equals the rest of the budget: '
      + ', '.join(f'{f:.1f} MHz -> {r:.1f} us' for f, r in zip(f_Rabis, cross)))

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(f_Rabis, RIN_1photon, c="#ff4da6", linewidth=2, label="RIN")
ax.plot(f_Rabis, infids_motion_blockade1, c="#2ecc71", linewidth=2, label="$\delta V$")
ax.plot(f_Rabis, infids_motion_rabi1, linewidth=2, label='$\delta \Omega$')
ax.plot(f_Rabis, decay_1photon, c="#7f8c8d", linewidth=2, label='$\gamma$')
ax.plot(f_Rabis, TO_1, c="#b06ae2", linewidth=2, label='TO')
ax.plot(f_Rabis, leakage1, linewidth=2, label='leakage')
ax.plot(f_Rabis, T2_1photon, c="#4e63ff", linewidth=2, label='$T_2^*$ = %g $\mu$s' % T2_star_us)
ax.plot(f_Rabis, sum_1photon, c="k", linewidth=4, label='$\Sigma$')
ax.axhline(1e-3, c='k', linestyle=":")
ax.set_ylabel("Infidelity", fontsize=14)
ax.set_xlabel("$\Omega/ 2\pi$ [MHz] ", fontsize=14)
ax.tick_params(labelsize=12)
ax.set_yscale('log')
ax.set_title('$1 \gamma$ gate, $T_2^*$ model', fontsize=16)
ax.set_ylim([1e-9, max(1e-3, 2 * float(np.max(sum_1photon)))])
ax.legend(fontsize=12)
fig.savefig(os.path.join(result, 'Error_vs_rabi_T2star.pdf'), bbox_inches='tight')

# Second figure: how the T2* channel (and the total) scale with T2*.
# infid_T2 = A/T2*^p, anchored on the exact simulation; everything else in the
# budget is T2*-independent, so the total follows for free.
T2_grid = np.logspace(np.log10(max(0.2, T2_star_us/50)), np.log10(T2_star_us*20), 200)
fig2, ax2 = plt.subplots(figsize=(7, 5))
rest = sum_1photon - T2_1photon
for i, f_R in enumerate(f_Rabis):
    curve = t2_scale_A[i] / T2_grid ** t2_scale_power
    p = ax2.plot(T2_grid, curve, linewidth=2, label='$T_2^*$ channel, $\Omega/2\pi$=%g MHz' % f_R)
    ax2.plot(T2_grid, curve + rest[i], linewidth=2, linestyle='--', c=p[0].get_color(),
             label='total, $\Omega/2\pi$=%g MHz' % f_R)
    ax2.plot([T2_star_us], [T2_1photon[i]], 'o', c=p[0].get_color(), ms=7)
ax2.axhline(1e-3, c='k', linestyle=":")
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlabel('$T_2^*$ [$\mu$s]', fontsize=14)
ax2.set_ylabel('Infidelity', fontsize=14)
ax2.set_title('Error vs $T_2^*$ (markers = full simulation)', fontsize=14)
ax2.legend(fontsize=11)
fig2.savefig(os.path.join(result, 'Error_vs_T2star.pdf'), bbox_inches='tight')


# -----------------------------
# Save config + raw scan data
# -----------------------------
def _to_jsonable(x):
    """Convert numpy/pandas types to JSON-serializable python types."""
    try:
        import numpy as _np
        if isinstance(x, (_np.integer,)):
            return int(x)
        if isinstance(x, (_np.floating,)):
            return float(x)
        if isinstance(x, (_np.ndarray,)):
            return x.tolist()
    except Exception:
        pass
    return x


config = dict(
    atom_name=atom_name,
    n=int(n), l=l, j=float(j), mj=float(mj),
    atom_d_um=float(atom_d),
    Bz_G=float(Bz),
    pulse_time=float(pulse_time),
    resolution=int(resolution),
    w0_rydberg_um=float(w0_rydberg),
    lambda_rydberg_um=float(lambda_rydberg),
    mj_leak_split_MHz=float(HF_split) / (2 * np.pi) if HF_split is not None else None,  # HF_split is rad/us
    T_atom_uK=float(T_atom),
    trap_depth_uK=float(trap_depth),
    lambda_trap_um=float(lambda_trap),
    w0_trap_um=float(w0_trap),
    num_samples=int(num_samples),
    RIN_strength=rin_strength,
    T2_star_us=float(T2_star_us),
    t2_model=t2_model,
    t2_correlation=t2_correlation,
    t2_average=t2_average,
    t2_gh_nodes=int(t2_gh_nodes),
    t2_mc_samples=int(t2_mc_samples),
    t2_white_fmax_over_Omega=float(t2_white_fmax_over_Omega),
    t2_white_calibrate=bool(t2_white_calibrate),
    f_Rabi_scan_MHz=dict(start=float(f_Rabis[0]), stop=float(f_Rabis[-1]), num=int(len(f_Rabis))),
    derived=dict(
        blockade_mrad=float(blockade_mrad),
        blockade_MHz=float(blockade_mrad/2/np.pi),
        R_lifetime_us=float(R_lifetime),
        sigma_delta_rad_per_us=float(sigma_delta),
        sigma_nu_MHz=float(sigma_nu),
        S_nu_white_MHz=float(S_nu_white),
        T2_star_doppler_only_us=float(T2_doppler_only),
    ),
)

raw_fig2 = dict(
    x=dict(name="Omega_over_2pi_MHz", values=_to_jsonable(f_Rabis)),
    y=dict(
        T2star=_to_jsonable(T2_1photon),
        T2star_linear_response=_to_jsonable(T2_1photon_lr),
        RIN=_to_jsonable(RIN_1photon),
        blockade=_to_jsonable(infids_motion_blockade1),
        rabi=_to_jsonable(infids_motion_rabi1),
        decay=_to_jsonable(decay_1photon),
        leakage=_to_jsonable(np.array(leakage1)),
        TO=_to_jsonable(TO_1),
        scattering=None,  # not modelled
        total=_to_jsonable(sum_1photon),
    ),
    t2_scaling=dict(
        # infid_T2(T2*) = A / T2*^power   (A anchored on the exact simulation)
        A=_to_jsonable(t2_scale_A),
        power=int(t2_scale_power),
        c2_exact_per_MHz2=_to_jsonable(t2_c2_exact),   # exact d2(infid)/d(nu)2
        I0_linear_response_per_MHz2=_to_jsonable(t2_lr_coeff),  # Haar metric, differs by O(1)
        J_integral_per_MHz=_to_jsonable(t2_lr_integral),
        budget_without_T2star=_to_jsonable(sum_1photon - T2_1photon),
    ),
)

out = dict(
    meta=dict(
        script=os.path.basename(__file__) if '__file__' in globals() else 'budget_1photon_T2star_scan.py',
    ),
    config={k: _to_jsonable(v) for k, v in config.items()},
    raw_fig2=raw_fig2,
)

out_json = os.path.join(result, f"scan_1photon_T2star_n{int(n)}_config_and_raw.json")
with open(out_json, "w") as f:
    json.dump(out, f, indent=2)

print(f"Saved config + raw scan data to: {out_json}")
