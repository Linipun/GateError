"""Laser frequency-noise and intensity-noise (RIN) LINEAR RESPONSE vs Rabi frequency,
for the 1-photon and 2-photon Rydberg gates.

What it plots
-------------
The bare response coefficients themselves -- not an infidelity at some assumed noise
level. Two panels sharing an x-axis of gate Rabi frequency Omega/2pi [MHz]:

  (a) I_nu [MHz^-2]       -- frequency-noise response;  1-F = I_nu * <dnu^2>
  (b) I_I  [dimensionless] -- intensity-noise response;  1-F = I_I * <(dI/I)^2>

Every curve is the response to ONE noise source, so nothing is pre-summed and the
curves are directly comparable: the 2-photon frequency response is per laser, and
the two 2-photon arms get a curve each. For a gate driven by several independently
noisy beams, multiply each curve by that beam's noise power and add.

Both channels enter at second order in the noise amplitude through the same
Eq.-(G13) universal response I(omega) (Appendix G of PRX Quantum 6, 010331), so the
infidelity is linear in the noise power and these coefficients are the whole story:

    infidelity  =  I(omega) * (noise power)

The response is evaluated quasi-statically (omega = 0), matching
budget_1photon_scan.py and budget_2_photon.py: the relevant noise bands are far
below the gate bandwidth Omega/2pi, so I(omega) ~ I(0) across the band.

infid_freq() and infid_rin() convert a curve to an infidelity at a given noise
level. The responses are written to JSON, so --load replots without recomputing.

Physics conventions follow the existing budget scripts exactly
--------------------------------------------------------------
* 1-photon: Cs nP3/2 mj=3/2 at 319 nm; blockade from blockades_symmetric_p.pkl.
  linear_response.build_Oseq works in NORMALIZED units (Rabi == 1, dt in 1/Omega),
  so B is passed as blockade/Omega and the frequency response is divided by Omega^2.
* 2-photon: Cs nS1/2 via 7P1/2 at 459 + 1038 nm; blockade from blockades_symmetric_s.pkl.
  linear_response_2photon works in REAL units (rad/us), so dt_real = dt/Omega and no
  Omega^2 division. Arms are balanced, Omega1 = Omega2 = sqrt(2*Omega*Delta), and
  delta1 compensates the background light shift, as in budget_2_photon.py.
* Both lasers of the 2-photon gate couple to the same effective two-photon detuning
  operator (O2photon_nu2 == O2photon_nu1), so one per-laser curve covers both; for
  independent FM noise on each, multiply by (S_nu1 + S_nu2). The two RIN operators
  differ and are plotted separately.

Usage
-----
    python budget_noise_response_scan.py                      # defaults, n=70, d=2 um
    python budget_noise_response_scan.py --n 60 --atom-d 2.2
    python budget_noise_response_scan.py --load result/noise_response_Cs_n70_d2.json
"""

import argparse
import json
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from arc import Caesium, Rubidium

from budget_monte_carlo import Hamiltonians, fid_optimize, phase_cosine_generate, find_blockade_Mrad
from linear_response import build_Oseq, response_G13, isometry_haar_full
from linear_response_2photon import (build_Oseq_2photon, response_2photon, compute_ktildes,
                                     O2photon_I1, O2photon_I2, O2photon_nu1)

# --- palette -----------------------------------------------------------------
# Hue carries the gate (categorical identity); lightness carries the noise level
# (an ordered magnitude). Blue/orange are the first two slots of the validated
# categorical palette; each ordinal ramp is three lightness steps of one hue, so it
# stays separable under every dichromacy. Checked numerically in OKLab: adjacent
# within-ramp dE ~ 19-20 and cross-hue dE 22-33 (both clear the >=15 floor), and the
# lightest step of each ramp clears the 2:1 ordinal contrast floor on a white page.
RAMP_1PHOTON = ["#86b6ef", "#2a78d6", "#104281"]   # blue   250 / 450 / 650
RAMP_2PHOTON = ["#fe9068", "#cb4801", "#752601"]   # orange, matched lightness steps
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"


# --- 2-photon blockade table -------------------------------------------------
def make_blockade_2photon(atom_name):
    """nS1/2 pair blockade [rad/us] vs separation, interpolated from the pickle.

    Mirrors find_blockade_Mrad_2photon in budget_2_photon.py; the 1-photon nP3/2
    table is the one budget_monte_carlo.find_blockade_Mrad already loads.
    """
    fname = 'Rb-Rb-s-states-90-deg.pkl' if atom_name == 'Rb' else 'blockades_symmetric_s.pkl'
    with open(fname, 'rb') as fh:
        table = pickle.load(fh)

    def blockade(n, d):
        entry = table[(atom_name, atom_name, n, n)]
        return np.interp(d, entry['r'], entry['B']) * 1e3 * 2 * np.pi

    return blockade


PHASE_GUESS = [2 * np.pi * 0.1122, 1.0431, -0.7318, 0.0]


def optimal_phase(Omega_Rabi, blockade_mrad, pulse_time, resolution, r_lifetime, warm_start=None):
    """Re-optimize the 4-parameter time-optimal cosine phase at this Rabi frequency.

    Returns (phases, dt, infid_TO, params). The optimizer runs with the real Rydberg
    lifetime (so the profile is the one actually used), then the profile is
    re-evaluated with decay switched off to isolate the coherent time-optimal
    infidelity -- the two-pass recipe of budget_1photon_scan.py / budget_2_photon.py.

    Unlike those scripts this multi-starts and keeps the best point. A single run from
    the fixed guess lands in different local minima as B/Omega drifts across a scan,
    which shows up as order-of-magnitude jitter in the TO infidelity between adjacent
    Rabi points and, through the phase profile, in the noise responses too. Seeding
    with the previous Rabi point's solution (warm_start) keeps the scan on one branch;
    the fixed guess, plus a re-run from each converged point, covers the case where
    that branch stops being the best one.
    """
    H_opt = Hamiltonians(Omega_Rabi1=Omega_Rabi, blockade_inf=False, blockade=blockade_mrad,
                         r_lifetime=r_lifetime, Delta1=0, Stark1=0, Stark2=0,
                         resolution=resolution, r_lifetime2=r_lifetime, pulse_time=pulse_time)

    starts = [np.asarray(PHASE_GUESS, float)]
    if warm_start is not None:
        starts.insert(0, np.asarray(warm_start, float))

    phase_params, best_f = None, np.inf
    for x0 in starts:
        x = x0
        for _ in range(2):          # restart from its own converged point
            res = opt.minimize(fun=fid_optimize, x0=x, args=(H_opt))
            x = res.x
            if res.fun < best_f:
                best_f, phase_params = res.fun, x

    H_eval = Hamiltonians(Omega_Rabi1=Omega_Rabi, blockade_inf=False, blockade=blockade_mrad,
                          r_lifetime=1e10, Delta1=0, Stark1=0, Stark2=0,
                          resolution=resolution, r_lifetime2=1e10, pulse_time=pulse_time)
    _, phase, dt = phase_cosine_generate(*phase_params, H_eval.pulse_time, H_eval.resolution)
    fid, _ = H_eval.return_fidel(phases=phase, dt=dt)
    return phase, dt, 1.0 - fid, phase_params


def scan(cfg):
    """Compute the bare (noise-level-independent) responses at each Rabi frequency."""
    atom = Rubidium() if cfg['atom_name'] == 'Rb' else Caesium()
    S_haar = isometry_haar_full()                       # D = 4, full logical subspace
    blockade_2p = make_blockade_2photon(cfg['atom_name'])

    n, atom_d = cfg['n'], cfg['atom_d']
    B_1p = find_blockade_Mrad(cfg['atom_name'], n, atom_d)          # nP3/2, rad/us
    B_2p = blockade_2p(n, atom_d)                                   # nS1/2, rad/us
    tau_1p = atom.getStateLifetime(n=n, l=1, j=1.5, temperature=300,
                                   includeLevelsUpTo=n + 20, s=0.5) * 1e6
    tau_2p = atom.getStateLifetime(n=n, l=0, j=0.5, temperature=300,
                                   includeLevelsUpTo=n + 20, s=0.5) * 1e6
    print(f"n = {n}, d = {atom_d} um")
    print(f"  1-photon nP3/2 blockade = {B_1p / 2 / np.pi:.1f} MHz,  lifetime = {tau_1p:.0f} us")
    print(f"  2-photon nS1/2 blockade = {B_2p / 2 / np.pi:.1f} MHz,  lifetime = {tau_2p:.0f} us")

    Delta = cfg['inter_detuning']
    # Omega-independent, so pay the ARC polarizability basis cost once.
    kt0_1, ktr_1, kt0_2, ktr_2 = compute_ktildes(n, Delta, Delta)

    f_Rabis = np.linspace(cfg['f_rabi_min'], cfg['f_rabi_max'], cfg['f_rabi_num'])
    out = {k: [] for k in ('I_nu_1p', 'I_I_1p', 'I_nu_2p', 'I_I1_2p', 'I_I2_2p', 'I_I_2p',
                           'TO_1p', 'TO_2p')}
    warm_1p = warm_2p = None

    for f_Rabi in f_Rabis:
        Omega = 2 * np.pi * f_Rabi
        print(f"  Omega/2pi = {f_Rabi:6.2f} MHz", end="", flush=True)

        # ---- 1-photon -------------------------------------------------------
        phase, dt, TO_1p, warm_1p = optimal_phase(Omega, B_1p, cfg['pulse_time'],
                                                  cfg['resolution'], tau_1p, warm_1p)
        # build_Oseq is written with Rabi == 1, so the blockade must be normalized.
        o_f = build_Oseq(phases=phase, dt=dt, B=B_1p / Omega, is_intensity=False)
        o_I = build_Oseq(phases=phase, dt=dt, B=B_1p / Omega, is_intensity=True)
        # /Omega^2 undoes the normalized time, putting I_nu in (rad/us)^-2 so that
        # multiplying by a mean-square detuning in MHz^2 gives an infidelity.
        I_nu_1p = response_G13(o_f, S_haar, 0, dt=dt) / Omega ** 2
        I_I_1p = response_G13(o_I, S_haar, 0, dt=dt)

        # ---- 2-photon -------------------------------------------------------
        phase2, dt2, TO_2p, warm_2p = optimal_phase(Omega, B_2p, cfg['pulse_time'],
                                                    cfg['resolution'], tau_2p, warm_2p)
        Omega1 = Omega2 = np.sqrt(2 * Omega * Delta)        # balanced arms
        delta1 = ktr_1 * Omega1 ** 2 + ktr_2 * Omega2 ** 2  # cancel the background light shift
        dt_real = dt2 / Omega                               # us; this module uses real units
        kw = dict(phases=phase2, dt=dt_real, B=B_2p, Omega1=Omega1, Omega2=Omega2,
                  delta1=delta1, delta2=0.0, Delta=Delta, inter_detuning=Delta, n=n)
        I_nu_2p = response_2photon(build_Oseq_2photon(Oinst_func=O2photon_nu1, **kw),
                                   S_haar, 0, dt_real)
        # The two arms carry independent intensity noise, so each gets its own response
        # (they add only once both are driven at the same RIN). Kept separate because
        # they are wildly unequal: ktilde_r,2 >> ktilde_r,1, so the 1038 nm arm, whose
        # polarizability shifts |1>, dominates the 2-photon intensity sensitivity.
        I_I1_2p = response_2photon(build_Oseq_2photon(Oinst_func=O2photon_I1, **kw),
                                   S_haar, 0, dt_real)
        I_I2_2p = response_2photon(build_Oseq_2photon(Oinst_func=O2photon_I2, **kw),
                                   S_haar, 0, dt_real)

        for k, v in zip(out, (I_nu_1p, I_I_1p, I_nu_2p, I_I1_2p, I_I2_2p,
                              I_I1_2p + I_I2_2p, TO_1p, TO_2p)):
            out[k].append(float(v))
        print(f"   TO: 1p {TO_1p:.2e} 2p {TO_2p:.2e} |"
              f"  I_nu: 1p {I_nu_1p:.3e} 2p {I_nu_2p:.3e} |"
              f"  I_I: 1p {I_I_1p:.3e} 2p-arm1 {I_I1_2p:.3e} 2p-arm2 {I_I2_2p:.3e}")

    return dict(f_Rabis=f_Rabis.tolist(),
                blockade_1p_MHz=B_1p / 2 / np.pi, blockade_2p_MHz=B_2p / 2 / np.pi,
                lifetime_1p_us=tau_1p, lifetime_2p_us=tau_2p,
                ktilde0_1=kt0_1, ktilder_1=float(ktr_1),
                ktilde0_2=kt0_2, ktilder_2=float(ktr_2),
                **out)


# --- noise-level scalings ----------------------------------------------------
# Both are the (linear) map from a noise power to an infidelity. The 1e12 converts
# Hz^2 to MHz^2, matching the rad/us units the responses are expressed in.
def infid_freq(I_nu, S_nu_hz2_per_hz, f_range_hz, n_lasers):
    """White FM noise of PSD S_nu on each of n_lasers, integrated over f_range."""
    return np.asarray(I_nu) * (n_lasers * S_nu_hz2_per_hz) * f_range_hz / 1e12


def infid_rin(I_I, rin_variance):
    """Integrated RIN variance sigma_I^2/<I>^2 (dimensionless)."""
    return np.asarray(I_I) * rin_variance


def make_figure(data, cfg, path):
    """Plot the bare linear-response coefficients themselves, not an infidelity.

    Every curve is the response to ONE noise source, so the two panels are directly
    comparable and nothing is pre-summed: the 2-photon frequency response is per
    laser, and the two 2-photon arms get one curve each. To turn a curve into an
    infidelity, multiply by that source's noise power via infid_freq / infid_rin;
    independent sources then add.
    """
    f = np.asarray(data['f_Rabis'])

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.2), sharex=True)
    ax_nu, ax_I = axes

    for ax in axes:
        ax.set_yscale('log')
        ax.set_xlabel(r'Rabi frequency  $\Omega/2\pi$  [MHz]')
        ax.grid(True, which='major', color=GRID, linewidth=0.7, zorder=0)
        ax.grid(True, which='minor', color=GRID, linewidth=0.35, alpha=0.6, zorder=0)
        ax.set_axisbelow(True)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        for side in ('left', 'bottom'):
            ax.spines[side].set_color('#c3c2b7')
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.set_xlim(f.min(), f.max())

    def draw(ax, curves, ncol):
        for y, c, ls, lw, lab in curves:
            ax.plot(f, y, color=c, lw=lw, ls=ls, zorder=3, label=lab)
        lo = min(np.min(y) for y, *_ in curves)
        hi = max(np.max(y) for y, *_ in curves)
        ax.set_ylim(lo / 3.0, hi * 3.0)
        ax.legend(ncol=ncol, fontsize=8.5, frameon=False, labelcolor=MUTED,
                  loc='upper center', bbox_to_anchor=(0.5, -0.155),
                  handlelength=2.4, columnspacing=1.4, borderaxespad=0.0)

    # (a) frequency-noise response. Both gates reduce to the same detuning operator,
    # so in the deep-blockade limit the per-laser responses coincide to 3 digits and
    # the curves lie on top of each other -- the 1-photon one is drawn wide so it
    # still reads underneath the dashed 2-photon curve. The 2-photon gate is worse
    # only because two independent lasers each contribute this much.
    draw(ax_nu,
         [(np.asarray(data['I_nu_1p']), RAMP_1PHOTON[1], '-', 4.5, '1-photon (319 nm)'),
          (np.asarray(data['I_nu_2p']), RAMP_2PHOTON[1], '--', 2.0, '2-photon, per laser')],
         ncol=2)
    ax_nu.set_title('(a)  Frequency-noise response  $I_\\nu$', loc='left', color=INK,
                    fontsize=11, fontweight='bold')
    ax_nu.set_ylabel(r'$I_\nu$  [MHz$^{-2}$]   $\;1-F = I_\nu\,\langle\delta\nu^2\rangle$')
    ax_nu.annotate('curves coincide — both gates reduce to\nthe same detuning operator; '
                   'the 2-photon\ngate pays it once per laser',
                   xy=(0.97, 0.93), xycoords='axes fraction', ha='right', va='top',
                   fontsize=8, color=MUTED, style='italic')

    # (b) intensity-noise response, one curve per independently-noisy beam.
    draw(ax_I,
         [(np.asarray(data['I_I_1p']), RAMP_1PHOTON[1], '-', 2.0, '1-photon (319 nm)'),
          (np.asarray(data['I_I1_2p']), RAMP_2PHOTON[0], '--', 2.0, '2-photon, arm 1 (459 nm)'),
          (np.asarray(data['I_I2_2p']), RAMP_2PHOTON[2], '--', 2.0, '2-photon, arm 2 (1038 nm)')],
         ncol=3)
    ax_I.set_title('(b)  Intensity-noise response  $I_I$', loc='left', color=INK,
                   fontsize=11, fontweight='bold')
    ax_I.set_ylabel(r'$I_I$  [dimensionless]   $\;1-F = I_I\,\langle(\delta I/I)^2\rangle$')
    ratio = np.asarray(data['I_I2_2p']).mean() / np.asarray(data['I_I_1p']).mean()
    ax_I.annotate(f'the 1038 nm arm alone is ~{ratio:.0f}$\\times$ the 1-photon\n'
                  'response: its polarizability shifts $|1\\rangle$, so\n'
                  'its RIN is a detuning error, not an amplitude one',
                  xy=(0.97, 0.80), xycoords='axes fraction', ha='right', va='top',
                  fontsize=8, color=MUTED, style='italic')

    sub = (rf"Cs, $n={cfg['n']}$, $d={cfg['atom_d']}\,\mu$m  |  "
           rf"1-photon $n$P$_{{3/2}}$ ($B/2\pi={data['blockade_1p_MHz']:.0f}$ MHz), "
           rf"2-photon $n$S$_{{1/2}}$ ($B/2\pi={data['blockade_2p_MHz']:.0f}$ MHz, "
           rf"$\Delta/2\pi={cfg['inter_detuning'] / 2 / np.pi / 1e3:.0f}$ GHz)  |  "
           rf"time-optimal CZ, $\Omega T={cfg['pulse_time']:g}$  |  quasi-static, $\omega=0$")
    fig.suptitle('Laser-noise linear response vs Rabi frequency', x=0.008, y=1.0,
                 ha='left', va='top', fontsize=13, fontweight='bold', color=INK)
    fig.text(0.008, 0.952, sub, ha='left', va='top', fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    for ext in ('pdf', 'png'):
        fig.savefig(f'{path}.{ext}', dpi=200, bbox_inches='tight', facecolor='white')
    print(f'wrote {path}.pdf and {path}.png')
    return fig


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--atom', default='Cs', choices=['Cs', 'Rb'])
    p.add_argument('--n', type=int, default=70, help='principal quantum number (both gates)')
    # 2.0 um puts both pair potentials on a smooth, strongly blockaded stretch of the
    # tabulated curves (nP3/2 ~740 MHz, nS1/2 ~840 MHz at n=70). The tables carry narrow
    # Forster dips -- n=70 nP3/2 collapses to 99 MHz at 2.5 um and nS1/2 to 111 MHz at
    # 2.78 um -- so check find_blockade_Mrad before moving this.
    p.add_argument('--atom-d', type=float, default=2.0, help='atom separation [um]')
    p.add_argument('--f-rabi-min', type=float, default=2.0, help='min Omega/2pi [MHz]')
    p.add_argument('--f-rabi-max', type=float, default=25.0, help='max Omega/2pi [MHz]')
    p.add_argument('--f-rabi-num', type=int, default=24, help='number of Rabi points')
    p.add_argument('--inter-detuning', type=float, default=5000.0,
                   help='2-photon intermediate-state detuning Delta/2pi [MHz]')
    p.add_argument('--pulse-time', type=float, default=7.65, help='gate duration in 1/Omega')
    p.add_argument('--resolution', type=int, default=200, help='phase-profile time steps')
    p.add_argument('--outdir', default='result')
    p.add_argument('--load', default=None, help='replot from a saved JSON instead of recomputing')
    a = p.parse_args()

    cfg = dict(atom_name=a.atom, n=a.n, atom_d=a.atom_d,
               f_rabi_min=a.f_rabi_min, f_rabi_max=a.f_rabi_max, f_rabi_num=a.f_rabi_num,
               inter_detuning=a.inter_detuning * 2 * np.pi,
               pulse_time=a.pulse_time, resolution=a.resolution)

    os.makedirs(a.outdir, exist_ok=True)
    stem = os.path.join(a.outdir, f'noise_response_{a.atom}_n{a.n}_d{a.atom_d:g}')

    if a.load:
        with open(a.load) as fh:
            saved = json.load(fh)
        data, cfg = saved['data'], saved['config']
    else:
        data = scan(cfg)
        with open(f'{stem}.json', 'w') as fh:
            json.dump(dict(config=cfg, data=data), fh, indent=2)
        print(f'wrote {stem}.json')

    make_figure(data, cfg, stem)
    plt.show()


if __name__ == '__main__':
    main()
