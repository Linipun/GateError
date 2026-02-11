from budget_monte_carlo import *
import phase_noise #import procData, p0dict_638
from linear_response import build_Oseq, response_G13, isometry_haar_full
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
n =  int(arg[0])
l = 1
j = 3/2
mj = 3/2

atom_d = 2.5 #um
Omega_Rabi= 1*2*np.pi  #MHz
Bz = 10 #G
pulse_time= 7.65 #Omega_Rabi
resolution = 200 # number of phase steps in the pulse

w0_rydberg = 10 #um
lambda_rydberg = 0.319 #um

HF_split = 500*np.pi*2 # MHz
HF_split = None #if mj=1/2 split is from B-field

alpha_dc = 700 #MHz (V/cm)^-2
alpha_dc = None

T_atom = 15 #uK
trap_depth = arg[1] #uK
lambda_trap = 1.064 #um
w0_trap = 1.2 #um

edc_fluc = 10e-3 #V/cm
edc_zero = 0 #V/m

bdc_fluc = 10e-3 #G

num_samples =1000

phase_noise_csv = "638_20MHz-2-2-2026.csv"
RIN_csv_path = '319_Intensity_0.442VDC.csv'
RIN_background_csv_path = 'UV_intensity_background.csv'
intensity_DC_V = 0.442

f_Rabis = np.linspace(0.1, 2, 50)
#### config #######


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

phase_noise_data = phase_noise.procData(phase_noise_csv, True, "638nm", range=20e6, p0=phase_noise.p0dict_638)
label = phase_noise_data[1][1]
vnoise_data = phase_noise_data[1][0]
vnoise_fs = []
vnoise_W = []
for d in vnoise_data:
    if d[0] >= 0:
        vnoise_fs.append(d[0] / 1e6)
        vnoise_W.append(d[1] * d[0] ** 2)

print('fitted frequency noise')
vnoise_fs = np.array(vnoise_fs)
vnoise_W = np.array(vnoise_W)

intensity_noise_csv = pd.read_csv(RIN_csv_path, header=None)
background_noise_csv = pd.read_csv(RIN_background_csv_path, header=None)
fs_intensity = intensity_noise_csv[0]
RIN_db = intensity_noise_csv[1]
fs_background = background_noise_csv[0]
bg_db = background_noise_csv[1]
rbw = fs_intensity[1] - fs_intensity[0]
carrier_p = (intensity_DC_V ** 2 / 50 * 1e3)  # dBm

bg_w = db_to_w(bg_db)
raw_RIN_w = db_to_w(RIN_db)
RIN_db_c = w_to_db(np.where((raw_RIN_w - bg_w) < 0, 1e-99, raw_RIN_w - bg_w))
RIN_dbc = (RIN_db_c - w_to_db(carrier_p) - w_to_db(
    rbw))  # convert to dBc/Hz= db(W_RIN/W_carrier/Hz) = db(W_RIN)-db(W_carrier)-db(Hz)

RIN_W = db_to_w(RIN_dbc)
fs_intensity = np.array(fs_intensity)
RIN_W = np.array(RIN_W)

## Linear Response ####
S_haar = isometry_haar_full()  # D=4
o_f = build_Oseq(phases=phase, dt=dt, B=blockade_mrad, is_intensity=False)
o_I = build_Oseq(phases=phase, dt=dt, B=blockade_mrad, is_intensity=True)
TO_1 = []
v_1photon = []
RIN_1photon = []
decay_1photon = []
infids_motion1 = []
leakage1 = []
scattering1 = []


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

for Omega_Rabi in Omega_Rabis:
    print('Omega:', Omega_Rabi / 2 / np.pi)
    H_gen1 = Hamiltonians(Omega_Rabi1=Omega_Rabi, blockade_inf=False, blockade=blockade_mrad, r_lifetime=R_lifetime,
                          Delta1=0,
                          Stark1=0, Stark2=0, resolution=resolution, r_lifetime2=R_lifetime, pulse_time=pulse_time)
    PhaseGuess = [2 * np.pi * 0.1122, 1.0431, -0.7318, 0]
    time, phase_guess, dt = phase_cosine_generate(*PhaseGuess, H_gen1.pulse_time, H_gen1.resolution)
    # fid_optimize(PhaseGuess, H_gen)
    # H_gen.return_fidel
    fid1, global_phi = H_gen1.return_fidel(phases=phase_guess, dt=dt)
    # print('Infidelity before optimizer:', 1-fid1)
    opt_out = opt.minimize(fun=fid_optimize, x0=PhaseGuess, args=(H_gen1))
    phase_params1 = opt_out.x
    # print(phase_params)
    infid_TO_theory = opt_out.fun
    # print('Infidelity after optimizer:', infid_TO_theory)
    # print('phase parameter', phase_params1)
    H_gen1 = Hamiltonians(Omega_Rabi1=Omega_Rabi, blockade_inf=False, blockade=blockade_mrad, r_lifetime=1e10, Delta1=0,
                          Stark1=0, Stark2=0, resolution=resolution, r_lifetime2=1e10, pulse_time=pulse_time)
    time, phase1, dt = phase_cosine_generate(*phase_params1, H_gen1.pulse_time, H_gen1.resolution)
    fid1, global_phi = H_gen1.return_fidel(phases=phase1, dt=dt)
    infid_TO1 = 1 - fid1
    # print(infid_TO1)
    TO_1.append(infid_TO1)

    vnoise1_contribution = []
    for i in range(len(vnoise_fs) - 1):
        deltaf = vnoise_fs[i + 1] - vnoise_fs[i]
        #
        If_1 = response_G13(o_f, S_haar, vnoise_fs[i] * 2 * np.pi / Omega_Rabi, dt=dt) / Omega_Rabi ** 2
        vnoise1_contribution.append(If_1 * vnoise_W[i] * deltaf / 1e6)

    v_1photon.append(np.sum(vnoise1_contribution))

    Inoise1_contribution = []
    for i in range(len(fs_intensity) - 2):
        deltaf = fs_intensity[i + 2] - fs_intensity[i + 1]
        Ii = response_G13(o_I, S_haar, fs_intensity[i + 2] * 2 * np.pi / 1e6 / Omega_Rabi, dt=dt)
        Inoise1_contribution.append(Ii * RIN_W[i + 2] * deltaf)
    RIN_1photon.append(np.sum(Inoise1_contribution))

    decay_1photon.append((2.95 / (Omega_Rabi)) / R_lifetime)

    infids_s1 = []
    for d in ds:
        blockade1 = find_blockade_Mrad(atom_name, n, d)
        H_gen = Hamiltonians(Omega_Rabi1=Omega_Rabi, blockade_inf=False, blockade=blockade1, r_lifetime=1e10, Delta1=0,
                             Stark1=0, Stark2=0, resolution=resolution, r_lifetime2=1e10, pulse_time=pulse_time)
        fid, global_phi = H_gen.asym_return_fidel(phases=phase1, dt=dt, omega1_scale=1, omega2_scale=1)
        infids_s1.append(1 - fid)

    infids_s1 = np.asarray(infids_s1)
    infids_motion1.append(np.mean(infids_s1) - infid_TO1)

    leakage1.append((Omega_Rabi) ** 2 / ((Omega_Rabi) ** 2 + HF_split ** 2))

    scattering1.append(0)

v_1photon = np.array(v_1photon)
RIN_1photon = np.array(RIN_1photon)
infids_motion1 = np.array(infids_motion1)
decay_1photon = np.array(decay_1photon)
sum_1photon = v_1photon+ RIN_1photon+infids_motion1+decay_1photon+leakage1

fig, ax = plt.subplots(figsize=(14,5), ncols=2)
ax.plot(f_Rabis, v_1photon, c="#4e63ff", linewidth=2)
ax.plot(f_Rabis, RIN_1photon, c="#ff4da6", linewidth=2)
ax.plot(f_Rabis, infids_motion1, c="#2ecc71", linewidth=2)
ax.plot(f_Rabis, decay_1photon, c="#7f8c8d", linewidth=2)
ax.plot(f_Rabis, sum_1photon, c="k", linewidth=4)
ax.plot(f_Rabis, scattering1, c='blue', linewidth=2)
ax.plot(f_Rabis, leakage1, c='blue', linewidth=2)
ax.axhline(1e-3, c='k', linestyle=":")
ax.set_ylabel("Infidelity", fontsize=14)
ax.set_xlabel("$\Omega/ 2\pi$ [MHz] ", fontsize=14)
ax.tick_params(labelsize=12)
ax.set_yscale('log')
ax.set_title('$1 \gamma$ gate', fontsize=16)
ax.set_ylim([1e-7, 1e-2])
fig.savefig(os.path.join(result, 'Error_vs_rabi.pdf'), bbox_inches='tight')


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
    # pandas scalars
    try:
        import pandas as _pd
        if isinstance(x, (_pd.Timestamp,)):
            return x.isoformat()
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
    HF_split_MHz=float(HF_split) if HF_split is not None else None,
    alpha_dc_MHz_per_Vcm2=float(alpha_dc) if alpha_dc is not None else None,
    T_atom_uK=float(T_atom),
    trap_depth_uK=float(trap_depth),
    lambda_trap_um=float(lambda_trap),
    w0_trap_um=float(w0_trap),
    edc_fluc_V_per_cm=float(edc_fluc),
    edc_zero_V_per_m=float(edc_zero),
    bdc_fluc_G=float(bdc_fluc),
    num_samples=int(num_samples),
    phase_noise_csv=str(phase_noise_csv),
    RIN_csv_path=str(RIN_csv_path),
    RIN_background_csv_path=str(RIN_background_csv_path),
    intensity_DC_V=float(intensity_DC_V),
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
        motional=_to_jsonable(infids_motion1),
        decay=_to_jsonable(decay_1photon),
        leakage=_to_jsonable(np.array(leakage1)),
        scattering=_to_jsonable(np.array(scattering1)),
        total=_to_jsonable(sum_1photon),
    ),
)

out = dict(
    meta=dict(
        script=os.path.basename(__file__) if '__file__' in globals() else 'budget_1photon_scan.py',
        saved_at=datetime.datetime.now().isoformat(timespec="seconds"),
    ),
    config={k: _to_jsonable(v) for k, v in config.items()},
    raw_fig2=raw_fig2,
)

out_json = os.path.join(result,f"scan_1photon_n{int(n)}_config_and_raw.json")
with open(out_json, "w") as f:
    json.dump(out, f, indent=2)

print(f"Saved config + raw scan data to: {out_json}")
