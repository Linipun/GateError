"""budget_1photon_scan.py with the mj=1/2 level in the loop everywhere.

Same input and same output as budget_1photon_scan.py:

    python budget_1photon_scan_leakage.py "70, 3.0"        # n, atom_d [um]

writes result/scan_1photon_leakage_n<n>_config_and_raw.json with the same
config / raw_fig2 structure and the same channel keys, plus the same figure.
Drop it into the notebook's transform_result() unchanged.

The difference is that LeakageHamiltonians (9 levels: both atoms with |1>, r and the mj=1/2
level r') is used for EVERY channel, not only for the leakage one:

  * the pulse phases are optimized against the 9-level Hamiltonian, so the gate is designed
    knowing the leak level exists;
  * the motional (blockade, Rabi) and quasi-static detuning channels are evaluated in the
    9-level model and baselined against the 9-level gate error.

The two reported channels keep their original meanings, so TO + leakage is the 9-level gate
error and the channels still sum to the total:

    TO      = the pulse's error in the 4-level model (intrinsic finite-blockade error)
    leakage = what the mj=1/2 level adds on top of it

Measured at the budget optimum (n=99, d=1.73 um, Omega/2pi=23.5 MHz, 2 GHz dressing), running
9-level throughout leaves the motional and detuning channels unchanged to four significant
figures -- they are differences against a baseline, so the leak level cancels -- and the whole
gain comes from the pulse optimization: total 8.61e-5 -> 8.28e-5.  Near resonance
(splitting/Omega ~ 1) that cancellation fails and this script is the only correct one.

Not 9-level: RIN and vnoise come from linear_response, whose basis has no r', so those two
channels are identical to the 4-level scan.  Given leakage ~ Omega^2/(6 Delta^2), the
correction to them is second order, but it is not computed here.

Runtime is dominated by the Monte-Carlo channels: ~45 min at num_samples = 10000 on 8 threads,
comparable to budget_1photon_scan.py itself (the 9-level propagation is ~4x heavier per
sample, but it is batched here rather than looped).
"""

from budget_monte_carlo import *
import phase_noise
from linear_response import build_Oseq, response_G13, isometry_haar_full
from budget_1photon_optimize import batch_infidelity_9
import numpy as np
import matplotlib.pyplot as plt
from arc import *
import json
import scipy.optimize as opt

import pandas as pd
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


result = 'result'
os.makedirs(result,exist_ok=True)


#### config #######
atom_name = 'Cs'
arg = eval('['+sys.argv[1]+']')
l = 1
j = 3/2
mj = 3/2
pulse_time= 7.65 #Omega_Rabi
resolution = 200 # number of phase steps in the pulse
lambda_rydberg = 0.319 #um
# Effective detuning of the nP3/2 mj=1/2 leakage state from the mj=3/2 gate state, in rad/us.
# Not a hyperfine splitting: a pi-polarised microwave resonantly dresses mj=1/2 with (n+1)S1/2
# mj=1/2 (mj=3/2 is untouched -- S1/2 has no mj=3/2 sublevel) and pushes it out of resonance.
# NOTE this must be the effective detuning Delta_eff = [sum_k |<P|k>|^2 / delta_k^2]^(-1/2)
# over the *dressed* branches k, NOT Omega_mw/2: nP3/2 sits midway between nS and (n+1)S
# (delta_S - delta_P = 0.49), so the microwave is only ~184 MHz from the P->nS transition and
# the dressing is three-level.  Delta_eff = 2pi*1.5 GHz already needs Omega_mw/2pi ~ 2.8 GHz.
# Set to None for the undressed (no-microwave) limit, i.e. bare Zeeman splitting only.
# NOTE with this script that limit is meaningful: the gate is designed in the 9-level model,
# so it does not fall apart the way the 4-level scan does when the splitting approaches Omega.
HF_split = 2000*np.pi*2 # rad/us
# HF_split = None

alpha_dc = 700 #MHz (V/cm)^-2
alpha_dc = None

T_atom = 1 #uK
lambda_trap = 1.064 #um
w0_trap = 1.064 #um
edc_fluc = 1e-3 #V/cm
edc_zero = 0 #V/m

bdc_fluc = 1e-3 #G

num_samples =10000
rin_strength = 1e-4

f_hz_hz2 = 220
f_range = 1e5 # Hz

efield_fluc_on = False   # add a FLUCTUATING (broadband, no-DC-bias) E-field channel?
edc_fluc_fast  = 12.5e-3   # V/cm, RMS of the broadband E-field noise
efield_range   = 2e6    # Hz, bandwidth of the E-field noise; quadratic Stark -> detuning noise up to 2*efield_range
#### config #######


### parameters ####
n =  int(arg[0])
atom_d = arg[1] #um
Omega_Rabi= 10*2*np.pi  #MHz
Bz = 10 #G
w0_rydberg = 7.5 #um
trap_depth = 1000 #uK
### parameters ####


f_Rabis = np.linspace(2, 40, 20)
# f_Rabis = np.array([1.5])


if atom_name == "Rb":
    atom = Rubidium()
elif atom_name == "Cs":
    atom = Caesium()
blockade_mrad = find_blockade_Mrad(atom_name, n, atom_d)
print('Blockade:', blockade_mrad/2/np.pi , 'MHz')
R_lifetime = atom.getStateLifetime(n=n,l=l,j=j,temperature=300, includeLevelsUpTo=n+20,s=0.5)*1e6
m_atom = atom.mass
if HF_split is None:
    HF_split = (atom.getZeemanEnergyShift(l=1, j=3/2, mj=3/2, magneticFieldBz=Bz/10000)-
                atom.getZeemanEnergyShift(l=1, j=3/2, mj=1/2, magneticFieldBz=Bz/10000))/hbar/1e6
print('mj=1/2 splitting:', HF_split/2/np.pi, 'MHz')
if alpha_dc is None:
    calc = StarkMap(atom)
    calc.defineBasis(n=n, l=1, j=1.5, mj=1.5, nMin=n - 20, nMax=n + 30, maxL=5, Bz=Bz / 10000)
    calc.diagonalise(np.linspace(0, 60, 600))
    alpha_dc = calc.getPolarizability(debugOutput=True)


## Linear Response ####
S_haar = isometry_haar_full()  # D=4

TO_1 = []
v_1photon = []
RIN_1photon = []
E_1photon = []
B_1photon = []
doppler_1photon = []
decay_1photon = []
infids_motion_blockade1 = []
infids_motion_rabi1 = []
leakage1 = []
scattering1 = []
efield_fluc_1photon = []

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
blockades = find_blockade_Mrad(atom_name, n, ds)   # np.interp is vectorised

for Omega_Rabi in Omega_Rabis:
    print('Omega:', Omega_Rabi / 2 / np.pi)
    # --- pulse phases optimized against the 9-LEVEL Hamiltonian (with decay, as in the scan)
    H_gen1 = LeakageHamiltonians(Omega_Rabi1=Omega_Rabi, blockade_inf=False,
                                 blockade=blockade_mrad, r_lifetime=R_lifetime,
                                 r_lifetime2=R_lifetime, Delta1=0, Stark1=0, Stark2=0,
                                 resolution=resolution, pulse_time=pulse_time,
                                 mj12_split=HF_split)
    PhaseGuess = [2 * np.pi * 0.1122, 1.0431, -0.7318, 0]
    opt_out = opt.minimize(fun=fid_optimize, x0=PhaseGuess, args=(H_gen1))
    phase_params1 = opt_out.x

    # --- the same pulse in the 4-level model: the intrinsic finite-blockade error
    H_gen_4 = Hamiltonians(Omega_Rabi1=Omega_Rabi, blockade_inf=False, blockade=blockade_mrad,
                           r_lifetime=1e10, Delta1=0, Stark1=0, Stark2=0,
                           resolution=resolution, r_lifetime2=1e10, pulse_time=pulse_time)
    time, phase, dt = phase_cosine_generate(*phase_params1, H_gen_4.pulse_time,
                                            H_gen_4.resolution)
    fid1, global_phi = H_gen_4.return_fidel(phases=phase, dt=dt)
    infid_TO1 = 1 - fid1
    TO_1.append(infid_TO1)

    # --- and in the 9-level model: the baseline every other channel is measured against
    H_gen_9 = LeakageHamiltonians(Omega_Rabi1=Omega_Rabi, blockade_inf=False,
                                  blockade=blockade_mrad, r_lifetime=10e9, r_lifetime2=10e9,
                                  Delta1=0, Stark1=0, Stark2=0, resolution=resolution,
                                  pulse_time=pulse_time, mj12_split=HF_split)
    fid_with_leak, global_phi = H_gen_9.return_fidel(phases=phase, dt=dt)
    infid_base9 = 1 - fid_with_leak
    leakage_mj = infid_base9 - infid_TO1
    leakage1.append(leakage_mj)

    o_f = build_Oseq(phases=phase, dt=dt, B=blockade_mrad/Omega_Rabi, is_intensity=False)
    o_I = build_Oseq(phases=phase, dt=dt, B=blockade_mrad/Omega_Rabi, is_intensity=True)

    Ii = response_G13(o_I, S_haar, 0, dt=dt)
    RIN_contribution = Ii * rin_strength
    RIN_1photon.append(RIN_contribution)

    If_1 = response_G13(o_f, S_haar, 0, dt=dt) / Omega_Rabi ** 2
    v_contribution = If_1 * f_hz_hz2 * f_range / 1e6 / 1e6
    v_1photon.append(v_contribution)

    # Fluctuating BROADBAND E-field, NO DC bias -> quadratic Stark: delta(t) = 1/2 alpha E(t)^2.
    if efield_fluc_on:
        dnu_rms = 0.5 * alpha_dc * edc_fluc_fast ** 2 * 1e6
        f_hi = 2.0 * efield_range
        fs_E = np.linspace(f_hi / 40, f_hi, 40)
        S_delta = dnu_rms ** 2 / f_hi
        dfE = fs_E[1] - fs_E[0]
        efield_fluc_contribution = 0.0
        for fE in fs_E:
            omg = 2 * np.pi * fE / 1e6 / Omega_Rabi
            I_w = response_G13(o_f, S_haar, omg, dt=dt) / Omega_Rabi ** 2
            efield_fluc_contribution += I_w * S_delta * dfE / 1e6 / 1e6
    else:
        efield_fluc_contribution = 0.0
    efield_fluc_1photon.append(efield_fluc_contribution)

    infids_decay = (2.95 / (Omega_Rabi)) / R_lifetime
    decay_1photon.append(infids_decay)

    # k_eff = 2*pi/lambda of the Rydberg beam (single 319 nm photon), in rad/s
    doppler_shift = 2 * np.pi / (lambda_rydberg * 1e-6) * np.sqrt(kb * (T_atom * 1e-6) / m_atom)
    delta_edc = abs(-1 / 2 * alpha_dc * 1e6 * ((edc_zero + edc_fluc) ** 2 - edc_zero ** 2)) * 2 * np.pi
    delta_bdc = atom.getZeemanEnergyShift(l=1, j=3 / 2, mj=3 / 2, magneticFieldBz=bdc_fluc / 10000) / hbar
    total_shift = np.sqrt(delta_bdc ** 2 + delta_edc ** 2 + doppler_shift ** 2)
    detunings = total_shift / 1e6

    # --- quasi-static detuning, 9-level.  Note the leak level's detuning is Delta1 + mj12_split,
    # which LeakageHamiltonians.H01 only got right after the fix; at Delta1 = 0 it never showed.
    d_samples = np.random.default_rng().normal(0.0, detunings, size=num_samples)
    infids_s = batch_infidelity_9(phase, dt, Omega_Rabi, blockade_mrad, d_samples, 1.0, 1.0,
                                  HF_split, 10e9, 10e9)
    infids_detuning = np.mean(infids_s) - infid_base9
    infids_detiuning_std = infids_s.std(ddof=1)

    infids_bdc = delta_bdc ** 2 / total_shift ** 2 * infids_detuning
    infids_edc = delta_edc ** 2 / total_shift ** 2 * infids_detuning
    infids_doppler = doppler_shift ** 2 / total_shift ** 2 * infids_detuning
    E_1photon.append(infids_edc)
    B_1photon.append(infids_bdc)
    doppler_1photon.append(infids_doppler)

    # --- motion, 9-level: blockade spread and Rabi inhomogeneity
    infids_blockade = batch_infidelity_9(phase, dt, Omega_Rabi, blockades, 0.0, 1.0, 1.0,
                                         HF_split, 10e9, 10e9)
    infids_rabi = batch_infidelity_9(phase, dt, Omega_Rabi, blockade_mrad, 0.0, rabis1, rabis2,
                                     HF_split, 10e9, 10e9)
    infids_motion_blockade = np.mean(infids_blockade) - infid_base9
    infids_motion_rabi = np.mean(infids_rabi) - infid_base9
    infids_motion_blockade1.append(infids_motion_blockade)
    infids_motion_rabi1.append(infids_motion_rabi)

    # Off-resonant / photoionization scattering is NOT modelled; placeholder, not a bound.
    scattering1.append(np.nan)
    total = (infid_TO1 + leakage_mj + infids_motion_blockade + infids_motion_rabi
             + infids_detuning + v_contribution + RIN_contribution + infids_decay
             + efield_fluc_contribution)
    print('  gate error (9-level):', infid_base9, ' of which leakage:', leakage_mj)
    print('  total error:', total)

v_1photon = np.array(v_1photon)
RIN_1photon = np.array(RIN_1photon)
infids_motion_blockade1 = np.array(infids_motion_blockade1)
infids_motion_rabi1 = np.array(infids_motion_rabi1)
E_1photon = np.array(E_1photon)
B_1photon = np.array(B_1photon)
doppler_1photon = np.array(doppler_1photon)
decay_1photon = np.array(decay_1photon)
leakage1 = np.array(leakage1)
TO_1 = np.array(TO_1)
efield_fluc_1photon = np.array(efield_fluc_1photon)

# TO_1 + leakage1 is the 9-level gate error, the baseline subtracted from every other channel,
# so both must be added back here.  Scattering is not modelled and is excluded.
sum_1photon = (TO_1+leakage1+v_1photon+RIN_1photon+infids_motion_blockade1+infids_motion_rabi1
               +decay_1photon+E_1photon+B_1photon+doppler_1photon+efield_fluc_1photon)
print('min infid:', min(sum_1photon), '(scattering not included)')

fig, ax = plt.subplots(figsize=(7,5))
ax.plot(f_Rabis, v_1photon, c="#4e63ff", linewidth=2, label="$v$")
ax.plot(f_Rabis, RIN_1photon, c="#ff4da6", linewidth=2, label="RIN")
ax.plot(f_Rabis, infids_motion_blockade1, c="#2ecc71", linewidth=2, label="$\delta V$")
ax.plot(f_Rabis, infids_motion_rabi1, linewidth=2, label='$\delta \Omega$')
ax.plot(f_Rabis, decay_1photon, c="#7f8c8d", linewidth=2, label='$\gamma$')
ax.plot(f_Rabis, TO_1, c="#b06ae2", linewidth=2, label='TO')
ax.plot(f_Rabis, sum_1photon, c="k", linewidth=4, label='$\Sigma$')
ax.plot(f_Rabis, leakage1, linewidth=2, label='leakage')
ax.plot(f_Rabis, E_1photon, linewidth=2, label='E')
ax.plot(f_Rabis, efield_fluc_1photon, linewidth=2, linestyle='--', label='E-fluc')
ax.plot(f_Rabis, B_1photon, linewidth=2, label='B')
ax.plot(f_Rabis, doppler_1photon, linewidth=2, label='doppler')
ax.axhline(1e-3, c='k', linestyle=":")
ax.set_ylabel("Infidelity", fontsize=14)
ax.set_xlabel("$\Omega/ 2\pi$ [MHz] ", fontsize=14)
ax.tick_params(labelsize=12)
ax.set_yscale('log')
ax.set_title('$1 \gamma$ gate, 9-level', fontsize=16)
ax.set_ylim([1e-9, 1e-3])
ax.legend(fontsize=14)
fig.savefig(os.path.join(result, 'Error_vs_rabi_leakage.pdf'), bbox_inches='tight')


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
    try:
        import pandas as _pd
        if isinstance(x, (_pd.Timestamp,)):
            return x.isoformat()
    except Exception:
        pass
    return x

config = dict(
    atom_name=atom_name,
    model='9-level (LeakageHamiltonians) throughout: pulse optimization, motion and detuning',
    n=int(n), l=l, j=float(j), mj=float(mj),
    atom_d_um=float(atom_d),
    Bz_G=float(Bz),
    pulse_time=float(pulse_time),
    resolution=int(resolution),
    w0_rydberg_um=float(w0_rydberg),
    lambda_rydberg_um=float(lambda_rydberg),
    mj_leak_split_MHz=float(HF_split) / (2 * np.pi) if HF_split is not None else None,
    alpha_dc_MHz_per_Vcm2=float(alpha_dc) if alpha_dc is not None else None,
    T_atom_uK=float(T_atom),
    trap_depth_uK=float(trap_depth),
    lambda_trap_um=float(lambda_trap),
    w0_trap_um=float(w0_trap),
    edc_fluc_V_per_cm=float(edc_fluc),
    edc_fluc_fast_V_per_cm=float(edc_fluc_fast),
    efield_range_Hz=float(efield_range),
    efield_fluc_on=bool(efield_fluc_on),
    edc_zero_V_per_m=float(edc_zero),
    bdc_fluc_G=float(bdc_fluc),
    num_samples=int(num_samples),
    phase_noise_csv=f_hz_hz2,
    RIN_strength=rin_strength,
    f_Rabi_scan_MHz=dict(start=float(f_Rabis[0]), stop=float(f_Rabis[-1]), num=int(len(f_Rabis))),
    derived=dict(
        blockade_mrad=float(blockade_mrad),
        blockade_MHz=float(blockade_mrad/2/np.pi),
        R_lifetime_us=float(R_lifetime),
    ),
)

raw_fig2 = dict(
    x=dict(name="Omega_over_2pi_MHz", values=_to_jsonable(f_Rabis)),
    y=dict(
        vnoise=_to_jsonable(v_1photon),
        RIN=_to_jsonable(RIN_1photon),
        blockade=_to_jsonable(infids_motion_blockade1),
        efield=_to_jsonable(E_1photon),
        efield_fluc=_to_jsonable(efield_fluc_1photon),
        bfield=_to_jsonable(B_1photon),
        doppler=_to_jsonable(doppler_1photon),
        rabi=_to_jsonable(infids_motion_rabi1),
        decay=_to_jsonable(decay_1photon),
        leakage=_to_jsonable(np.array(leakage1)),
        TO=_to_jsonable(TO_1),
        scattering=None,  # not modelled
        total=_to_jsonable(sum_1photon),
    ),
)

out = dict(
    meta=dict(
        script=os.path.basename(__file__) if '__file__' in globals() else 'budget_1photon_scan_leakage.py',
    ),
    config={k: _to_jsonable(v) for k, v in config.items()},
    raw_fig2=raw_fig2,
)

out_json = os.path.join(result,f"scan_1photon_leakage_n{int(n)}_config_and_raw.json")
with open(out_json, "w") as f:
    json.dump(out, f, indent=2)

print(f"Saved config + raw scan data to: {out_json}")
