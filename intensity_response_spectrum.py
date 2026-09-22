"""Intensity-noise (RIN) linear response vs NOISE frequency, at two Rabi frequencies.

Companion to budget_noise_response_scan.py, which sweeps the Rabi frequency and
evaluates the response quasi-statically (omega = 0). This one holds the Rabi
frequency fixed -- at 2 and 20 MHz by default -- and sweeps the noise frequency
instead, giving the spectral filter function I_I(omega) that the gate applies to a
RIN spectrum. The 1-photon and 2-photon gates are overlaid.

What the curve means
--------------------
I_I(omega) is the Eq.-(G13) universal response (Appendix G of PRX Quantum 6,
010331) evaluated at noise frequency omega. For an intensity-noise PSD S_RIN(f)
[1/Hz] the gate infidelity is

    1 - F  =  \\int I_I(2 pi f) S_RIN(f) df

so I_I(omega) is the weight the gate gives to RIN at each Fourier frequency. It is
essentially a low-pass: noise slower than the gate acts quasi-statically and gets
the full weight I_I(0), while noise much faster than the gate averages away over
the pulse. The roll-off therefore sits at the gate bandwidth, which is why the two
Rabi frequencies produce the same shape in units of Omega but decade-shifted in Hz.

It is not exactly monotonic -- the 1-photon curve peaks ~2-4% above its DC value
near 2 pi f / Omega ~ 0.3 before falling, and every beam shows lobe structure past
the knee -- so a noise tone parked in a lobe is weighted more than the smooth
roll-off would suggest.

One curve per independently-noisy beam
--------------------------------------
The two 2-photon arms are NOT plotted as a single total, because they are not the
same filter. In units of 2 pi f / Omega the -3 dB points are

    2-photon arm 1 (459 nm)  : 0.48    <- narrowest
    1-photon       (319 nm)  : 0.80
    2-photon arm 2 (1038 nm) : 1.63    <- widest

so arm 1 is actually a tighter filter than the 1-photon gate, while arm 2 is twice
as wide. Arm 2 also carries ~80x arm 1's DC response (its polarizability shifts
|1>, making its RIN a detuning error rather than an amplitude one), so it both
dominates the total and integrates over the widest band -- which is why a total
would have looked like "the 2-photon curve" and hidden the fact that the 459 nm arm
is the best-behaved beam of the three. Summing them is only correct when both arms
carry the same RIN; with per-arm noise levels, weight each curve separately.

Panels
------
  (a) vs normalized noise frequency 2 pi f / Omega -- the universal filter shape.
      The 2 and 20 MHz curves nearly collapse here, which is the point: the filter
      is set by the pulse, not by the absolute frequency.
  (b) vs absolute noise frequency f [MHz], log axis -- where the roll-off actually
      lands in the lab, i.e. which part of a measured RIN spectrum the gate is
      sensitive to. This is the same data as (a) with the x-axis rescaled by
      f_Rabi, so no extra computation.

Curves are one per independently-noisy beam -- the 1-photon gate's 319 nm beam and
each of the two 2-photon arms separately -- so nothing is pre-summed and the three
filters can be compared directly. The JSON also stores their sum as I_2p.

Units, following the two linear_response modules
------------------------------------------------
* 1-photon: linear_response works in NORMALIZED time (dt in 1/Omega), so the noise
  frequency passed to response_G13 is the dimensionless 2 pi f / Omega.
* 2-photon: linear_response_2photon works in REAL time (us), so response_2photon
  takes omega in rad/us, i.e. 2 pi f with f in MHz.
Both are driven from the same normalized grid x = 2 pi f / Omega, so the two gates
are always compared at the same noise frequency.

Usage
-----
    python intensity_response_spectrum.py
    python intensity_response_spectrum.py --f-rabis 2 20 --x-max 4
    python intensity_response_spectrum.py --load result/intensity_spectrum_Cs_n70_d2.json
"""

import argparse
import json
import os

import numpy as np
import matplotlib.pyplot as plt
from arc import Caesium, Rubidium

from budget_monte_carlo import find_blockade_Mrad
from linear_response import build_Oseq, response_G13, isometry_haar_full
from linear_response_2photon import (build_Oseq_2photon, response_2photon, compute_ktildes,
                                     O2photon_I1, O2photon_I2)
from budget_noise_response_scan import (make_blockade_2photon, optimal_phase,
                                        RAMP_1PHOTON, RAMP_2PHOTON, INK, MUTED, GRID)


def spectrum(cfg):
    """I_I vs noise frequency, for each requested Rabi frequency and both gates."""
    atom = Rubidium() if cfg['atom_name'] == 'Rb' else Caesium()
    S_haar = isometry_haar_full()                       # D = 4, full logical subspace
    blockade_2p = make_blockade_2photon(cfg['atom_name'])

    n, atom_d = cfg['n'], cfg['atom_d']
    B_1p = find_blockade_Mrad(cfg['atom_name'], n, atom_d)
    B_2p = blockade_2p(n, atom_d)
    tau_1p = atom.getStateLifetime(n=n, l=1, j=1.5, temperature=300,
                                   includeLevelsUpTo=n + 20, s=0.5) * 1e6
    tau_2p = atom.getStateLifetime(n=n, l=0, j=0.5, temperature=300,
                                   includeLevelsUpTo=n + 20, s=0.5) * 1e6
    print(f"n = {n}, d = {atom_d} um")
    print(f"  1-photon nP3/2 blockade = {B_1p / 2 / np.pi:.1f} MHz")
    print(f"  2-photon nS1/2 blockade = {B_2p / 2 / np.pi:.1f} MHz")

    Delta = cfg['inter_detuning']
    _, ktr_1, _, ktr_2 = compute_ktildes(n, Delta, Delta)

    # x = 2 pi f / Omega: the noise frequency in units of the Rabi frequency. Driving
    # both gates from this shared grid keeps them at the same physical f at every point.
    x = np.linspace(0.0, cfg['x_max'], cfg['x_num'])
    runs = []

    for f_Rabi in cfg['f_rabis']:
        Omega = 2 * np.pi * f_Rabi
        print(f"  Omega/2pi = {f_Rabi:g} MHz ...", end="", flush=True)

        # ---- 1-photon: normalized units, so x is passed straight through ----
        phase, dt, TO_1p, _ = optimal_phase(Omega, B_1p, cfg['pulse_time'],
                                            cfg['resolution'], tau_1p)
        o_I = build_Oseq(phases=phase, dt=dt, B=B_1p / Omega, is_intensity=True)
        I_1p = np.array([response_G13(o_I, S_haar, w, dt=dt) for w in x])

        # ---- 2-photon: real units, so x must be converted to rad/us ----
        phase2, dt2, TO_2p, _ = optimal_phase(Omega, B_2p, cfg['pulse_time'],
                                              cfg['resolution'], tau_2p)
        Omega1 = Omega2 = np.sqrt(2 * Omega * Delta)        # balanced arms
        delta1 = ktr_1 * Omega1 ** 2 + ktr_2 * Omega2 ** 2  # cancel the background light shift
        dt_real = dt2 / Omega                               # us
        kw = dict(phases=phase2, dt=dt_real, B=B_2p, Omega1=Omega1, Omega2=Omega2,
                  delta1=delta1, delta2=0.0, Delta=Delta, inter_detuning=Delta, n=n)
        oseq_I1 = build_Oseq_2photon(Oinst_func=O2photon_I1, **kw)
        oseq_I2 = build_Oseq_2photon(Oinst_func=O2photon_I2, **kw)
        w_real = x * Omega                                  # rad/us
        I1_2p = np.array([response_2photon(oseq_I1, S_haar, w, dt_real) for w in w_real])
        I2_2p = np.array([response_2photon(oseq_I2, S_haar, w, dt_real) for w in w_real])

        runs.append(dict(f_Rabi=float(f_Rabi), TO_1p=float(TO_1p), TO_2p=float(TO_2p),
                         I_1p=I_1p.tolist(), I1_2p=I1_2p.tolist(), I2_2p=I2_2p.tolist(),
                         I_2p=(I1_2p + I2_2p).tolist()))
        print(f"\n      1-photon      : I_I(0) = {I_1p[0]:9.3f}   -3dB at 2pi f/Omega = "
              f"{_half_power(x, I_1p):.2f}")
        print(f"      2p arm1 (459) : I_I(0) = {I1_2p[0]:9.3f}   -3dB at 2pi f/Omega = "
              f"{_half_power(x, I1_2p):.2f}")
        print(f"      2p arm2 (1038): I_I(0) = {I2_2p[0]:9.3f}   -3dB at 2pi f/Omega = "
              f"{_half_power(x, I2_2p):.2f}")

    return dict(x=x.tolist(), runs=runs,
                blockade_1p_MHz=B_1p / 2 / np.pi, blockade_2p_MHz=B_2p / 2 / np.pi)


def _half_power(x, y):
    """First x where the response has fallen to half its DC value (-3 dB), or nan."""
    below = np.flatnonzero(y < 0.5 * y[0])
    if below.size == 0:
        return float('nan')
    i = below[0]
    if i == 0:
        return float(x[0])
    # linear interpolation in log(y) across the crossing
    y0, y1 = np.log(y[i - 1]), np.log(y[i])
    t = (np.log(0.5 * y[0]) - y0) / (y1 - y0)
    return float(x[i - 1] + t * (x[i] - x[i - 1]))


# One entry per independently-noisy beam. Colors match budget_noise_response_scan.py
# so the two figures read as one set: blue = 1-photon, the two 2-photon arms as a light
# and a dark step of the orange ramp.
BEAMS = [('I_1p', RAMP_1PHOTON[1], '1-photon (319 nm)', '1-photon'),
         ('I1_2p', RAMP_2PHOTON[0], '2-photon arm 1 (459 nm)', 'arm 1'),
         ('I2_2p', RAMP_2PHOTON[2], '2-photon arm 2 (1038 nm)', 'arm 2')]

# Rabi frequency is carried by linestyle, not color, since color is spent on the beam.
# The lowest Rabi frequency is drawn wide so that where the curves collapse in panel (a)
# the overlap reads as a result rather than as a missing curve.
RABI_STYLES = [('-', 4.5), ('--', 2.0), (':', 2.0), ('-.', 2.0)]


def make_figure(data, cfg, path):
    x = np.asarray(data['x'])
    runs = data['runs']

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.2))
    ax_n, ax_a = axes

    for ax in axes:
        ax.set_yscale('log')
        ax.set_ylabel(r'Intensity-noise response  $I_I(\omega)$')
        ax.grid(True, which='major', color=GRID, linewidth=0.7, zorder=0)
        ax.grid(True, which='minor', color=GRID, linewidth=0.35, alpha=0.6, zorder=0)
        ax.set_axisbelow(True)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        for side in ('left', 'bottom'):
            ax.spines[side].set_color('#c3c2b7')
        ax.tick_params(colors=MUTED, labelsize=9)

    # Beams outer, Rabi inner: with a 3-column legend (filled column-major) each beam
    # then gets its own column, with its Rabi frequencies stacked underneath it.
    for key, color, name, _short in BEAMS:
        for i, run in enumerate(runs):
            fR = run['f_Rabi']
            ls, lw = RABI_STYLES[i % len(RABI_STYLES)]
            y = np.asarray(run[key])
            lab = rf'{name}, $\Omega/2\pi={fR:g}$ MHz'
            ax_n.plot(x, y, color=color, lw=lw, ls=ls, zorder=3, label=lab)
            # Same data, x rescaled to absolute MHz. Drop x=0, which a log axis cannot show.
            ax_a.plot(x[1:] * fR, y[1:], color=color, lw=lw, ls=ls, zorder=3, label=lab)

    ax_n.set_xlim(0, x.max())
    ax_n.set_xlabel(r'Normalized noise frequency  $2\pi f/\Omega$')
    ax_n.set_title('(a)  Universal filter shape', loc='left', color=INK,
                   fontsize=11, fontweight='bold')
    # -3 dB point of each beam, from the lowest-Rabi run (they agree across runs). The
    # dotted verticals sit in the empty band between the arm-2 curve and the other two;
    # the values are quoted in the annotation rather than labelled on the axis, which
    # would collide with the panel title.
    cuts = []
    for key, color, _name, short in BEAMS:
        xh = _half_power(x, np.asarray(runs[0][key]))
        if np.isfinite(xh):
            ax_n.axvline(xh, color=color, lw=1.0, ls=(0, (1, 2)), zorder=2)
            cuts.append((xh, short))
    cuts.sort()

    ax_n.annotate('the two Rabi frequencies collapse — the filter is\n'
                  'set by the pulse, not the absolute frequency.\n'
                  'The three beams do not: each has its own width.\n'
                  + r'$-3$ dB at $2\pi f/\Omega$ = '
                  + ', '.join(f'{v:.2f} ({nm})' for v, nm in cuts),
                  xy=(0.03, 0.74), xycoords='axes fraction', ha='left', va='top',
                  fontsize=8, color=MUTED, style='italic')

    ax_a.set_xscale('log')
    ax_a.set_xlabel(r'Noise frequency  $f$  [MHz]')
    ax_a.set_title('(b)  Same curves vs absolute noise frequency', loc='left', color=INK,
                   fontsize=11, fontweight='bold')
    ax_a.annotate('a faster gate pushes the roll-off out,\nso it integrates RIN over '
                  'a wider band',
                  xy=(0.03, 0.05), xycoords='axes fraction', ha='left', va='bottom',
                  fontsize=8, color=MUTED, style='italic')

    handles, labels = ax_n.get_legend_handles_labels()
    fig.legend(handles, labels, ncol=len(BEAMS), fontsize=8.5, frameon=False,
               labelcolor=MUTED, loc='lower center', bbox_to_anchor=(0.5, -0.02),
               handlelength=2.6, columnspacing=2.2)

    sub = (rf"Cs, $n={cfg['n']}$, $d={cfg['atom_d']}\,\mu$m  |  "
           rf"1-photon $n$P$_{{3/2}}$ ($B/2\pi={data['blockade_1p_MHz']:.0f}$ MHz), "
           rf"2-photon $n$S$_{{1/2}}$ ($B/2\pi={data['blockade_2p_MHz']:.0f}$ MHz, "
           rf"$\Delta/2\pi={cfg['inter_detuning'] / 2 / np.pi / 1e3:.0f}$ GHz, per beam)  |  "
           rf"time-optimal CZ, $\Omega T={cfg['pulse_time']:g}$")
    fig.suptitle('Intensity-noise response vs noise frequency', x=0.008, y=1.0,
                 ha='left', va='top', fontsize=13, fontweight='bold', color=INK)
    fig.text(0.008, 0.95, sub, ha='left', va='top', fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.05, 1, 0.92))
    for ext in ('pdf', 'png'):
        fig.savefig(f'{path}.{ext}', dpi=200, bbox_inches='tight', facecolor='white')
    print(f'wrote {path}.pdf and {path}.png')
    return fig


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--atom', default='Cs', choices=['Cs', 'Rb'])
    p.add_argument('--n', type=int, default=70)
    p.add_argument('--atom-d', type=float, default=2.0, help='atom separation [um]')
    p.add_argument('--f-rabis', type=float, nargs='+', default=[2.0, 20.0],
                   help='Rabi frequencies Omega/2pi [MHz] to compare, low to high')
    p.add_argument('--x-max', type=float, default=3.0,
                   help='max normalized noise frequency 2 pi f / Omega')
    p.add_argument('--x-num', type=int, default=301, help='noise-frequency grid points')
    p.add_argument('--inter-detuning', type=float, default=5000.0,
                   help='2-photon intermediate-state detuning Delta/2pi [MHz]')
    p.add_argument('--pulse-time', type=float, default=7.65, help='gate duration in 1/Omega')
    p.add_argument('--resolution', type=int, default=200, help='phase-profile time steps')
    p.add_argument('--outdir', default='result')
    p.add_argument('--load', default=None, help='replot from a saved JSON')
    a = p.parse_args()

    cfg = dict(atom_name=a.atom, n=a.n, atom_d=a.atom_d, f_rabis=a.f_rabis,
               x_max=a.x_max, x_num=a.x_num,
               inter_detuning=a.inter_detuning * 2 * np.pi,
               pulse_time=a.pulse_time, resolution=a.resolution)

    os.makedirs(a.outdir, exist_ok=True)
    stem = os.path.join(a.outdir, f'intensity_spectrum_{a.atom}_n{a.n}_d{a.atom_d:g}')

    if a.load:
        with open(a.load) as fh:
            saved = json.load(fh)
        data, cfg = saved['data'], saved['config']
    else:
        data = spectrum(cfg)
        with open(f'{stem}.json', 'w') as fh:
            json.dump(dict(config=cfg, data=data), fh, indent=2)
        print(f'wrote {stem}.json')

    make_figure(data, cfg, stem)
    plt.show()


if __name__ == '__main__':
    main()
