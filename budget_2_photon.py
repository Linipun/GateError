import numpy as np
import matplotlib.pyplot as plt
from arc import *
import pickle
import scipy.optimize as opt
import scipy
from budget_monte_carlo import *
from linear_response_2photon import response_2photon, build_Oseq_2photon, O2photon_I1, O2photon_I2, O2photon_nu1, O2photon_nu2
from linear_response import build_Oseq, response_G13, isometry_haar_full
import phase_noise #import procData, p0dict_638
import pandas as pd
from scipy.integrate import solve_ivp
import sys
import json
from datetime import datetime

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
n=63

Omega_Rabi = 8*2*np.pi


atom_d = 2.8 #um
# Omega_Rabi = 5*2*np.pi #MHz*2pi

inter_detuning = 5000*2*np.pi #MHz*2pi
intermediate_n = 7
intermediate_l = 1
intermediate_j = 1/2

Bz = 10 #G
pulse_time= 7.65 #Omega_Rabi
resolution = 200 # number of phase steps in the pulse


w0_rydberg = 10 #um
lambda_rydberg = 0.459 #um

w0_rydberg1 = 10 #um
lambda_rydberg1 = 1.038 #um
# HF_split = 500*np.pi*2 # MHz

T_atom = 5 #uK
trap_depth = 1000 #uK
lambda_trap = 1.064 #um
w0_trap = 1.2 #um

edc_fluc = 1e-3 #V/cm
edc_zero = 0 #V/m
bdc_fluc = 1e-3 #G

num_samples =100

pol_dc = 282


phase_noise_csv = "638_20MHz-2-2-2026.csv"
RIN_csv_path = '319_Intensity_0.442VDC.csv'
RIN_background_csv_path = 'UV_intensity_background.csv'
intensity_DC_V = 0.442

# Blockade
with open('6363_blockade.pkl', 'rb') as file:
    # Load the object from the file
    blockade2_dict = pickle.load(file)
def find_blockade_Mrad_2photon(atom_name, n, d):
    # blockades = blockade2_dict[atom_name][str(n)]
    b_r = blockade2_dict['r']
    b_values = blockade2_dict['blockade_MHz']
    b = np.interp(d, b_r, b_values)
    return b * 2 * np.pi


if atom_name == "Rb":
    atom = Rubidium()
    n_g = 5
    w_qubit = 9192631770 * 2 * np.pi
elif atom_name == "Cs":
    atom = Caesium()
    n_g = 6
    w_qubit = 6.8e9 * 2 * np.pi


blockade_mrad = find_blockade_Mrad_2photon(atom_name, n, atom_d)
print('Blockade:', blockade_mrad/2/np.pi , 'MHz')
R_lifetime = atom.getStateLifetime(n=n,l=0,j=1/2,temperature=300, includeLevelsUpTo=n+20,s=0.5)*1e6
tau_7p = atom.getStateLifetime(n=intermediate_n, l=intermediate_l, j=intermediate_j, temperature=300,
                               includeLevelsUpTo=n + 20, s=0.5) * 1e6
m_atom = atom.mass


f_Rabis = np.linspace(0.1, 2, 50)
Omega_Rabis = 2 * np.pi * f_Rabis

# --- Collect scan results for saving ---
scan_results = {
    'Omega_Rabi_MHz': [],
    'total_error': [],
    'breakdown': {
        'theoretical_limit': [],
        'detuning_total': [],
        'detuning_E_field': [],
        'detuning_B_field': [],
        'detuning_doppler': [],
        'motion_total': [],
        'motion_blockade_only': [],
        'motion_rabi_only': [],
        'intermediate_scattering': [],
        'rydberg_decay': [],
        'laser_phase_noise': [],
        'laser_intensity_RIN': [],
    },
}

for Omega_Rabi in Omega_Rabis:
    #### 2 photons gate ###
    Omega1_0 = np.sqrt(2*Omega_Rabi*inter_detuning)
    Omega2_0 = np.sqrt(2*Omega_Rabi*inter_detuning)
    Delta = inter_detuning#

    v_photon1 = atom.getTransitionFrequency(n1=n_g, l1=0, j1=1/2, n2=intermediate_n, l2=intermediate_l,
                                            j2=intermediate_j, s=0.5)
    v_photon1 += inter_detuning*1e6/2/np.pi
    v_photon2 = atom.getTransitionFrequency(n1=intermediate_n, l1=intermediate_l, j1=intermediate_j,
                                            n2=n, l2=0, j2=1/2, s=0.5)
    v_photon2 -= inter_detuning*1e6/2/np.pi

    alpha_g_gen = DynamicPolarizability(atom, n=n_g, l=0, j=1/2, s=0.5)
    alpha_g_gen.defineBasis(n_g, 9)
    alpha_r_gen = DynamicPolarizability(atom, n=n, l=0, j=1/2, s=0.5)
    alpha_r_gen.defineBasis(n_g, n+20)

    alpha_r_1 = alpha_r_gen.getPolarizability(c/(v_photon1), units='SI', accountForStateLifetime=False, mj=None)[0]
    d1 = atom.getDipoleMatrixElement(n1=n_g, l1=0, j1=1/2, mj1=-1/2, n2=intermediate_n, l2=intermediate_l, j2=intermediate_j,
                                     mj2=1/2,q=1, s=0.5)*bohr_r/hbar*e

    ktilde0_1 = -(1/4/(Delta+w_qubit/1e6)-1/4/Delta)
    ktilder_1 = -(alpha_r_1*2*np.pi*1e6)/4/d1**2+1/4/Delta

    d2 = atom.getDipoleMatrixElement(n1=intermediate_n, l1=intermediate_l, j1=intermediate_j, mj1=1/2,
                                     n2=n, l2=0, j2=1/2, mj2=-1/2,q=-1, s=0.5)*bohr_r/hbar*e
    alpha_1_2 = alpha_g_gen.getPolarizability(c/(v_photon2), units='SI', accountForStateLifetime=False, mj=None)[0]

    ktilde0_2 = 0.0
    ktilder_2 = -1/4/Delta+(alpha_1_2*2*np.pi*1e6)/4/d2**2

    delta1 = (ktilder_1*Omega1_0**2+ktilder_2*Omega2_0**2)
    delta2 = 0

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
    infid_TO_theory = opt_out.fun
    print('Infidelity after optimizer:', infid_TO_theory)
    print('phase parameter', phase_params)
    H_gen = Hamiltonians(Omega_Rabi1=Omega_Rabi, blockade_inf=False, blockade=blockade_mrad, r_lifetime=1e10, Delta1=0,
                         Stark1=0, Stark2=0, resolution=resolution, r_lifetime2=1e10, pulse_time=pulse_time)
    time, phase, dt = phase_cosine_generate(*phase_params, H_gen.pulse_time, H_gen.resolution)
    fid, global_phi = H_gen.return_fidel(phases=phase, dt=dt)
    infid_TO = 1 - fid
    print(infid_TO)
    plt.plot(time, phase)


    doppler_shift = (1/lambda_rydberg-1/lambda_rydberg1)/1e-6*np.sqrt(kb*(T_atom*1e-6)/m_atom)
    print('doppler shift:', doppler_shift/2/np.pi, 'Hz')

    calc = StarkMap(atom)
    if pol_dc is None:
        calc.defineBasis(n=n, l=0, j=0.5, mj=-0.5, nMin=n-20, nMax=n+30, maxL=5, Bz=Bz/10000)
        calc.diagonalise(np.linspace(0,1,100))
        pol_dc = calc.getPolarizability(debugOutput=True)
    delta_edc = abs(-1/2*pol_dc*1e6*((edc_zero+edc_fluc)**2-edc_zero**2))*2*np.pi
    print('Electric DC fluctuation:', delta_edc/2/np.pi, 'Hz')

    delta_bdc = atom.getZeemanEnergyShift(l=0, j=1/2, mj=-1/2, magneticFieldBz=bdc_fluc/10000)/hbar
    print('Magnetic DC fluctuation:', delta_bdc/2/np.pi, 'Hz')

    total_shift = np.sqrt(delta_bdc**2+ delta_edc**2 + doppler_shift**2)
    print('Total DC detuning fluctuation:', total_shift/2/np.pi, 'Hz')
    detunings = total_shift/1e6/Omega_Rabi
    print('delta/Ω:', detunings)
    print('==============')
    infids_s = []
    for i in range(num_samples):
        d = sample_gaussian(detunings)
        H_gen = Hamiltonians(Omega_Rabi1=Omega_Rabi, blockade_inf=False, blockade=blockade_mrad, r_lifetime=10e9, Delta1=d,
                             Stark1=0, Stark2=0, resolution=resolution, r_lifetime2=10e9, pulse_time=pulse_time)
        fid, global_phi = H_gen.return_fidel(phases=phase, dt=dt)
        infids_s.append(1-fid)
    infids_s = np.asarray(infids_s)
    infids_detuning = np.mean(infids_s)-infid_TO
    infids_detiuning_std = infids_s.std(ddof=1)
    infids_bdc = delta_bdc**2/total_shift**2*infids_detuning
    infids_edc = delta_edc**2/total_shift**2*infids_detuning
    infids_doppler = doppler_shift**2/total_shift**2*infids_detuning

    print(f'total error due to detuning: {infids_detuning} +/- {infids_detiuning_std}')
    print('error due to E field: {}'.format(infids_edc))
    print('error due to B field: {}'.format(infids_bdc))
    print('error due to doppler: {}'.format(infids_doppler))

    infids_s = []
    infids_blockade = []
    infids_rabi = []
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
    infids_s = []
    infids_blockade =[]
    infids_rabi =[]
    for rabi1, rabi2, d in zip(rabis1, rabis2, ds):
        blockade = find_blockade_Mrad_2photon(atom_name, n, d)
        # total
        H_gen = Hamiltonians(Omega_Rabi1=Omega_Rabi, blockade_inf=False, blockade=blockade, r_lifetime=10e9, Delta1=0,
                         Stark1=0, Stark2=0, resolution=resolution, r_lifetime2=10e9, pulse_time=pulse_time)
        fid, global_phi = H_gen.asym_return_fidel(phases=phase, dt=dt, omega1_scale=rabi1, omega2_scale=rabi2)
        infids_s.append(1-fid)

        H_gen_block = Hamiltonians(Omega_Rabi1=Omega_Rabi, blockade_inf=False, blockade=blockade, r_lifetime=10e9, Delta1=0,
                     Stark1=0, Stark2=0, resolution=resolution, r_lifetime2=10e9, pulse_time=pulse_time)
        fid, global_phi = H_gen_block.asym_return_fidel(phases=phase, dt=dt, omega1_scale=1, omega2_scale=1)
        infids_blockade.append(1-fid)

        H_gen_rabi = Hamiltonians(Omega_Rabi1=Omega_Rabi, blockade_inf=False, blockade=blockade_mrad, r_lifetime=10e9, Delta1=0,
                     Stark1=0, Stark2=0, resolution=resolution, r_lifetime2=10e9, pulse_time=pulse_time)
        fid, global_phi = H_gen_rabi.asym_return_fidel(phases=phase, dt=dt, omega1_scale=rabi1, omega2_scale=rabi2)
        infids_rabi.append(1-fid)
    infids_s = np.asarray(infids_s)
    infids_motion = np.mean(infids_s) - infid_TO
    infids_motion_std = infids_s.std(ddof=1)

    infids_blockade = np.asarray(infids_blockade)
    infids_motion_blockade = np.mean(infids_blockade) - infid_TO
    infids_motion_blockade_std = infids_blockade.std(ddof=1)

    infids_rabi = np.asarray(infids_rabi)
    infids_motion_rabi = np.mean(infids_rabi) - infid_TO
    infids_motion_rabi_std = infids_rabi.std(ddof=1)

    print(f'total error due to atom motion: {infids_motion} +/- {infids_motion_std}')
    print(f'total error due to atom motion(blockade): {infids_motion_blockade} +/- {infids_motion_blockade_std}')
    print(f'total error due to atom motion(rabi): {infids_motion_rabi} +/- {infids_motion_rabi_std}')

    # delta_mj = atom.getZeemanEnergyShift(l=1, j=3/2, mj=3/2, magneticFieldBz=Bz/10000)/hbar/1e6 - \
    # atom.getZeemanEnergyShift(l=1, j=3/2, mj=1/2, magneticFieldBz=Bz/10000)/hbar/1e6
    ###Scattering off intermediate state
    def Ht(phases, decay_enabled=True):
        O1 = Omega1_0 / Omega_Rabi * np.exp(-1j * phases) / 2
        d1 = Omega1_0 / Omega_Rabi * np.exp(1j * phases) / 2
        O2 = Omega2_0 / Omega_Rabi / 2
        d2 = Omega2_0 / Omega_Rabi / 2
        # print(np.abs(Omega))

        H0 = np.array([[0, O1, 0],
                       [d1, inter_detuning / Omega_Rabi, O2],
                       [0, d2, 0]])
        decay_rate = 1 / tau_7p / Omega_Rabi
        decay = np.diag([0, -1j * decay_rate / 2, 0])
        if decay_enabled:
            return H0 + decay
        return H0


    psi = np.zeros((3), complex)
    psi[0] = 1
    psi_no = np.copy(psi)
    resolution = 10000
    scatter_time, scatter_phase, scatter_dt = phase_cosine_generate(*phase_params, H_gen.pulse_time, resolution)
    for phi in scatter_phase:
        H0 = Ht(phi)
        H0_no = Ht(phi, decay_enabled=False)
        psi = scipy.linalg.expm(-1j * H0 * scatter_dt) @ psi
        psi_no = scipy.linalg.expm(-1j * H0_no * scatter_dt) @ psi_no
    # scattering_e = 0.95/Omega_Rabi/tau_7p*Omega1_0**2/(inter_detuning+w_qubit)**2 + 0.95/Omega_Rabi/tau_7p*(Omega1_0**2+Omega2_0**2)/(inter_detuning)**2+0.12/Omega_Rabi/tau_7p*(Omega1_0**2-Omega2_0**2)/(inter_detuning)**2
    scattering_e =  abs(psi_no[0] ** 2) - abs(psi[0] ** 2)
    print('error due to scattering:', scattering_e)


    loss_decay = (2.95/(Omega_Rabi))/R_lifetime
    print('error due to Rydberg decay:', loss_decay)

    phase_noise_data = phase_noise.procData(phase_noise_csv, True, "638nm", range=20e6, p0=phase_noise.p0dict_638)
    label = phase_noise_data[1][1]
    vnoise_data = phase_noise_data[1][0]
    vnoise_fs = []
    vnoise_W = []
    for d in vnoise_data:
        if d[0] >=0:
            vnoise_fs.append(d[0]/1e6)
            vnoise_W.append(d[1]*d[0]**2)

    vnoise_fs= np.array(vnoise_fs)
    vnoise_W = np.array(vnoise_W)
    S_haar = isometry_haar_full()   # D=4
    T = 2 * np.pi * 1.215 /Omega_Rabi
    t_real = np.linspace(0.0, T, resolution)
    dt_real = t_real[1] - t_real[0]
    oOseq_nu1 = build_Oseq_2photon(phases=phase, dt=dt_real, B=blockade_mrad, Omega1=Omega1_0, Omega2=Omega2_0, delta1=delta1,
                                delta2=delta2, Delta=Delta, inter_detuning=inter_detuning, n=n, Oinst_func=O2photon_nu1)
    vnoise_contribution = []
    for i in range(len(vnoise_fs)-1):
        deltaf = vnoise_fs[i+1]-vnoise_fs[i]
        If_2p_1 = response_2photon(oOseq_nu1, S_haar, vnoise_fs[i]*2*np.pi, dt_real)
        vnoise_contribution.append((If_2p_1*2)*vnoise_W[i]*deltaf/1e6)
    vnoise_error= np.sum(vnoise_contribution)
    print('error due to laser phase noise:', vnoise_error)

    intensity_noise_csv = pd.read_csv(RIN_csv_path, header=None)
    background_noise_csv = pd.read_csv(RIN_background_csv_path, header=None)
    fs_intensity = intensity_noise_csv[0]
    RIN_db = intensity_noise_csv[1]
    fs_background = background_noise_csv[0]
    bg_db = background_noise_csv[1]
    rbw = fs_intensity[1]-fs_intensity[0]
    carrier_p = (intensity_DC_V**2/50*1e3) #dBm

    bg_w = db_to_w(bg_db)
    raw_RIN_w = db_to_w(RIN_db)
    RIN_db_c = w_to_db(np.where((raw_RIN_w-bg_w)<0,1e-99,raw_RIN_w-bg_w))
    RIN_dbc = (RIN_db_c-w_to_db(carrier_p)-w_to_db(rbw)) #convert to dBc/Hz= db(W_RIN/W_carrier/Hz) = db(W_RIN)-db(W_carrier)-db(Hz)

    # fig , ax = plt.subplots(ncols=2)
    # ax[0].plot(fs_intensity, RIN_dbc)
    # ax[0].plot(fs_intensity, RIN_db)
    RIN_W = db_to_w(RIN_dbc)
    fs_intensity = np.array(fs_intensity)
    RIN_W = np.array(RIN_W)
    RIN_contribution = []
    oOseq_I1 = build_Oseq_2photon(phases=phase, dt=dt_real, B=blockade_mrad, Omega1=Omega1_0, Omega2=Omega2_0, delta1=delta1,
                        delta2=delta2, Delta=Delta, inter_detuning=inter_detuning, n=n, Oinst_func=O2photon_I1)
    oOseq_I2 = build_Oseq_2photon(phases=phase, dt=dt_real, B=blockade_mrad, Omega1=Omega1_0, Omega2=Omega2_0, delta1=delta1,
                        delta2=delta2, Delta=Delta, inter_detuning=inter_detuning, n=n, Oinst_func=O2photon_I2)

    RIN_contribution = []
    # fs  = np.linspace(0,15,500)
    # for f in fs:
    for i in range(len(fs_intensity)-2):
        deltaf = fs_intensity[i+2]-fs_intensity[i+1]
        Ii_2p_1 = response_2photon(oOseq_I1, S_haar, fs_intensity[i+2]*2*np.pi/1e6, dt_real)
        Ii_2p_2 = response_2photon(oOseq_I2, S_haar, fs_intensity[i+2]*2*np.pi/1e6, dt_real)
        RIN_contribution.append((Ii_2p_1+Ii_2p_2)*RIN_W[i+2]*deltaf)

    RIN_error = np.sum(RIN_contribution)
    print('error due to RIN:', RIN_error)


    total_error = loss_decay  + infids_motion + infids_detuning+scattering_e+RIN_error+vnoise_error
    # Store per-point results
    scan_results['Omega_Rabi_MHz'].append(float(Omega_Rabi/(2*np.pi)))
    scan_results['total_error'].append(float(total_error))
    scan_results['breakdown']['theoretical_limit'].append(float(infid_TO))
    scan_results['breakdown']['detuning_total'].append(float(infids_detuning))
    scan_results['breakdown']['detuning_E_field'].append(float(infids_edc))
    scan_results['breakdown']['detuning_B_field'].append(float(infids_bdc))
    scan_results['breakdown']['detuning_doppler'].append(float(infids_doppler))
    scan_results['breakdown']['motion_total'].append(float(infids_motion))
    scan_results['breakdown']['motion_blockade_only'].append(float(infids_motion_blockade))
    scan_results['breakdown']['motion_rabi_only'].append(float(infids_motion_rabi))
    scan_results['breakdown']['intermediate_scattering'].append(float(scattering_e))
    scan_results['breakdown']['rydberg_decay'].append(float(loss_decay))
    scan_results['breakdown']['laser_phase_noise'].append(float(vnoise_error))
    scan_results['breakdown']['laser_intensity_RIN'].append(float(RIN_error))
    print('total error:', total_error)

# --- Save config + raw scan data to JSON ---
config = {
    'atom_name': atom_name,
    'n': int(n),
    'atom_d_um': float(atom_d),
    'Bz_G': float(Bz),
    'pulse_time': float(pulse_time),
    'resolution': int(resolution),
    'num_samples': int(num_samples),
    'intermediate': {
        'inter_detuning_MHz_2pi': float(inter_detuning/(2*np.pi)),
        'intermediate_n': int(intermediate_n),
        'intermediate_l': int(intermediate_l),
        'intermediate_j': float(intermediate_j),
    },
    'rydberg_beams': {
        'w0_rydberg_um': float(w0_rydberg),
        'lambda_rydberg_um': float(lambda_rydberg),
        'w0_rydberg1_um': float(w0_rydberg1),
        'lambda_rydberg1_um': float(lambda_rydberg1),
    },
    'trap': {
        'T_atom_uK': float(T_atom),
        'trap_depth_uK': float(trap_depth),
        'lambda_trap_um': float(lambda_trap),
        'w0_trap_um': float(w0_trap),
    },
    'dc_fluctuations': {
        'edc_fluc_V_per_cm': float(edc_fluc),
        'edc_zero_V_per_m': float(edc_zero),
        'bdc_fluc_G': float(bdc_fluc),
        'pol_dc': None if pol_dc is None else float(pol_dc),
    },
    'noise_files': {
        'phase_noise_csv': str(phase_noise_csv),
        'RIN_csv_path': str(RIN_csv_path),
        'RIN_background_csv_path': str(RIN_background_csv_path),
        'intensity_DC_V': float(intensity_DC_V),
    },
    'derived': {
        'blockade_MHz_2pi': float(blockade_mrad/(2*np.pi)),
        'rydberg_lifetime_us': float(R_lifetime),
    },
    'saved_at': datetime.now().isoformat(),
}

raw = {
    'f_Rabi_MHz': [float(x) for x in f_Rabis],
    'Omega_Rabi_MHz_2pi': scan_results['Omega_Rabi_MHz'],
    'total_error': scan_results['total_error'],
    'breakdown': scan_results['breakdown'],
}

out = {'config': config, 'raw': raw}
out_name = f"scan_2photon_n{n}_config_and_raw.json"
with open(out_name, 'w') as f:
    json.dump(out, f, indent=2)
print(f"Saved config + raw scan data to: {out_name}")
