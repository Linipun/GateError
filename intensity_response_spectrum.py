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

Not every bump is physical: the blockade-aliasing trap
------------------------------------------------------
The toggling-frame operator carries the |rr> phase rotating at the blockade B, and
the kernel samples it at dt. Once B*dt > pi that rotation is undersampled and folds
back to |B*dt mod 2pi| / dt -- a sharp, tall, entirely spurious peak that can land
in the middle of the plotted band. It is easy to mistake for a resonance, so the
tells are worth knowing: it sits at a FIXED ABSOLUTE frequency (independent of both
Omega and Delta, unlike any real feature of this system), it MOVES with the time
grid, and it has no counterpart in the H_eff level spacings, which here are only
~Omega (dressed states) and ~B (blockade).

Concretely, at Omega/2pi = 1 MHz with B/Omega = 841 and resolution 1000 it produced
a fake peak at 20 MHz, two orders of magnitude above the true response, drifting
22.4 -> 20.8 -> 20.0 MHz as the resolution went 250 -> 500 -> 1000 and disappearing
once B*dt < pi. `spectrum` now raises the resolution per run until B*dt < pi/2 and
says so when it does. This bites at LOW Rabi frequency, where B/Omega is largest.

Only the SPECTRUM is affected. At omega = 0 the fast blockade oscillation
contributes almost nothing to the double time integral, so the quasi-static numbers
in budget_noise_response_scan.py are fine at its resolution of 200 -- checked
against resolution 6000 at 2, 10 and 25 MHz, where they agree to better than 1.6%
even with B*dt as large as 16 rad.

The bump at 2 pi f / Omega ~ 1.2
--------------------------------
Arm 2's curve does not roll off monotonically: it comes back up to ~96% of its DC
value near 2 pi f / Omega = 1.18 (f ~ 24 MHz for a 20 MHz gate). This is a
dressed-state resonance -- noise at the Rabi frequency resonantly drives transitions
between the dressed states of the driven atom -- and it appears only for
DETUNING-like noise. An amplitude operator points along the drive axis and is nearly
static in the dressed frame, so it has no such resonance; a sigma_z-like operator
rotates at the Rabi frequency and does. That is exactly the arm-1/arm-2 split: arm 2
acts through the light shift, i.e. as detuning, so it resonates, while arm 1 and the
1-photon gate are genuine amplitude noise and do not. Numerically the DC-normalised
arm-2 curve tracks the detuning response to 7%, against 70% for arm 1.

There are really TWO resonances, at Omega for the singly excited logical states and
at sqrt(2) Omega for the blockaded |11> (collective enhancement). The gate is short,
so each is Fourier-broadened to ~2 pi / (Omega T) = 0.82 in 2 pi f / Omega, wider
than their 0.41 separation, and they merge into the one broad bump seen here.
Driving at constant phase and stretching the pulse splits them back apart, which is
how the above was checked: peaks at x = 1.127 (merged) for Omega T = 7.65, 1.002 and
1.403 at Omega T = 30, and 1.000 and 1.414 at Omega T = 120.

Practically: do not assume the gate low-passes RIN away. This resonance fills in the
roll-off, which is why arm 2's -3 dB point sits out at 1.63 -- intensity noise on the
1038 nm arm counts at close to full weight all the way out to ~1.6x the Rabi
frequency, i.e. to ~33 MHz for a 20 MHz gate.

One curve per independently-noisy beam
--------------------------------------
The two 2-photon arms are NOT plotted as a single total, because they are not the
same filter. In units of 2 pi f / Omega the -3 dB points, at the default
Delta/2pi = 10 GHz, are

    2-photon arm 1 (459 nm)  : 0.55    <- narrowest
    1-photon       (319 nm)  : 0.82
    2-photon arm 2 (1038 nm) : 1.63    <- widest

so arm 1 is actually a tighter filter than the 1-photon gate, while arm 2 is twice
as wide. Arm 2 also carries a few hundred times arm 1's DC response, because its
polarizability shifts |1> and so its RIN is a detuning error rather than an
amplitude one. It therefore both dominates the total and integrates over the widest
band -- which is why a total would have looked like "the 2-photon curve" and hidden
the fact that the 459 nm arm is the best-behaved beam of the three. Summing them is
only correct when both arms carry the same RIN; with per-arm noise levels, weight
each curve separately.

Arm 2's DC response scales as Delta^2: the arms run at Omega_j = sqrt(2 Omega Delta),
so its light shift ~ ktilde_r,2 * Omega_2^2 grows linearly in Delta and the response
goes as its square. Raising Delta to suppress intermediate-state scattering costs
RIN sensitivity at exactly that rate (I_I(0) ~ 133 at 5 GHz, ~547 at 10 GHz).

Panels
------
  (a) vs normalized noise frequency 2 pi f / Omega -- the universal filter shape.
      The 2 and 20 MHz curves nearly collapse here, which is the point: the filter
      is set by the pulse, not by the absolute frequency.
  (b) vs absolute noise frequency f [MHz] -- where the roll-off actually lands in
      the lab, i.e. which part of a measured RIN spectrum the gate is sensitive to.
      Every curve spans the same 0 .. x_max * max(f_Rabi) here, so the gates are
      compared over one common band rather than each stopping at its own x_max.
      That costs the slower gates extra high-frequency points (see `spectrum`), and
      it is why they show many interference lobes across the panel: the lobes are
      spaced 2 pi / (Omega T) ~ 0.82 in 2 pi f / Omega, which is ~1.6 MHz for a
      2 MHz gate but ~16 MHz for a 20 MHz one.

Curves are one per independently-noisy beam -- the 1-photon gate's 319 nm beam and
each of the two 2-photon arms separately -- so nothing is pre-summed and the three
filters can be compared directly. The JSON also stores their sum as I_2p.

Units, following the two linear_response modules
------------------------------------------------
* 1-photon: linear_response works in NORMALIZED time (dt in 1/Omega), so the noise
  frequency passed to response_G13 is the dimensionless 2 pi f / Omega.
* 2-photon: linear_response_2photon works in REAL time (us), so response_2photon
  takes omega in rad/us, i.e. 2 pi f with f in MHz.
Both are driven from the same per-run ABSOLUTE frequency grid f_abs, converted into
each module's own convention, so the gates are always compared at the same physical
noise frequency. The sweeps use response_G13_spectrum / response_2photon_spectrum,
which build the Eq.-(G13) kernel once and contract it against every frequency.

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
from linear_response import build_Oseq, response_G13_spectrum, isometry_haar_full
from linear_response_2photon import (build_Oseq_2photon, response_2photon_spectrum,
                                     compute_ktildes, O2photon_I1, O2photon_I2)
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

    # Every run is computed on its OWN absolute-frequency grid, the union of two:
    #   * a fine grid in normalized units, 0 .. x_max in x_num steps, which is what
    #     panel (a) needs -- a shared normalized grid would leave a slow gate with only
    #     a handful of points there once the grid is stretched to cover panel (b);
    #   * the common absolute grid 0 .. f_max_abs, so that in panel (b) every curve
    #     spans the same x range instead of stopping at x_max * its own f_Rabi.
    # f_max_abs is set by the fastest gate, so that run's two grids coincide and it
    # costs nothing extra; only the slower runs get the extra high-frequency points.
    f_max_abs = cfg['x_max'] * max(cfg['f_rabis'])
    # The common grid is densified by the same factor it is stretched, so the SLOWEST
    # gate keeps its normalized resolution all the way out. Without this the response's
    # interference lobes -- spaced 2 pi / (Omega T) ~ 0.82 in 2 pi f / Omega, i.e. only
    # ~1.6 MHz apart for a 2 MHz gate -- are aliased into a fuzzy band across the
    # extended tail. Evaluation is a single matrix product, so the extra points are free.
    stretch = f_max_abs / (cfg['x_max'] * min(cfg['f_rabis']))
    n_common = min(int(cfg['x_num'] * stretch), 6001)
    f_common = np.linspace(0.0, f_max_abs, n_common)
    runs = []

    for f_Rabi in cfg['f_rabis']:
        Omega = 2 * np.pi * f_Rabi
        print(f"  Omega/2pi = {f_Rabi:g} MHz ...", end="", flush=True)

        f_abs = np.unique(np.concatenate([np.linspace(0.0, cfg['x_max'] * f_Rabi,
                                                      cfg['x_num']), f_common]))
        x = f_abs / f_Rabi                  # = 2 pi f / Omega, this run's normalized axis

        # The time step has to resolve the BLOCKADE, not just the Rabi frequency. The
        # toggling-frame operator carries the |rr> phase rotating at B, and sampling it
        # at dt folds everything above pi/dt back down: once B*dt > pi the fold lands at
        # |B*dt mod 2pi|/dt, which is a sharp spurious peak sitting in the middle of the
        # plotted band. At Omega/2pi = 1 MHz, B/Omega = 841 and resolution 1000 it put a
        # fake resonance at 20 MHz, two orders of magnitude above the true response, and
        # it moved with the grid (22.4 / 20.8 / 20.0 MHz at resolution 250 / 500 / 1000)
        # before vanishing once B*dt < pi. Requiring B*dt < pi/2 keeps it out.
        # It bites at LOW Rabi frequency, where B/Omega is largest.
        res = max(cfg['resolution'],
                  int(np.ceil(2 * cfg['pulse_time'] * max(B_1p, B_2p) / (np.pi * Omega))) + 1)
        if res > cfg['resolution']:
            print(f" [resolution {cfg['resolution']} -> {res} to resolve B/Omega ="
                  f" {max(B_1p, B_2p) / Omega:.0f}]", end="", flush=True)

        # ---- 1-photon: normalized units, so x is passed straight through ----
        phase, dt, TO_1p, _ = optimal_phase(Omega, B_1p, cfg['pulse_time'],
                                            res, tau_1p)
        o_I = build_Oseq(phases=phase, dt=dt, B=B_1p / Omega, is_intensity=True)
        I_1p = response_G13_spectrum(o_I, S_haar, x, dt=dt)

        # ---- 2-photon: real units, so x must be converted to rad/us ----
        phase2, dt2, TO_2p, _ = optimal_phase(Omega, B_2p, cfg['pulse_time'],
                                              res, tau_2p)
        Omega1 = Omega2 = np.sqrt(2 * Omega * Delta)        # balanced arms
        delta1 = ktr_1 * Omega1 ** 2 + ktr_2 * Omega2 ** 2  # cancel the background light shift
        dt_real = dt2 / Omega                               # us
        kw = dict(phases=phase2, dt=dt_real, B=B_2p, Omega1=Omega1, Omega2=Omega2,
                  delta1=delta1, delta2=0.0, Delta=Delta, inter_detuning=Delta, n=n)
        oseq_I1 = build_Oseq_2photon(Oinst_func=O2photon_I1, **kw)
        oseq_I2 = build_Oseq_2photon(Oinst_func=O2photon_I2, **kw)
        w_real = 2 * np.pi * f_abs                          # rad/us (== x * Omega)
        I1_2p = response_2photon_spectrum(oseq_I1, S_haar, w_real, dt_real)
        I2_2p = response_2photon_spectrum(oseq_I2, S_haar, w_real, dt_real)

        runs.append(dict(f_Rabi=float(f_Rabi), TO_1p=float(TO_1p), TO_2p=float(TO_2p),
                         f_abs=f_abs.tolist(),
                         I_1p=I_1p.tolist(), I1_2p=I1_2p.tolist(), I2_2p=I2_2p.tolist(),
                         I_2p=(I1_2p + I2_2p).tolist()))
        print(f"\n      1-photon      : I_I(0) = {I_1p[0]:9.3f}   -3dB at 2pi f/Omega = "
              f"{_half_power(x, I_1p):.2f}")
        print(f"      2p arm1 (459) : I_I(0) = {I1_2p[0]:9.3f}   -3dB at 2pi f/Omega = "
              f"{_half_power(x, I1_2p):.2f}")
        print(f"      2p arm2 (1038): I_I(0) = {I2_2p[0]:9.3f}   -3dB at 2pi f/Omega = "
              f"{_half_power(x, I2_2p):.2f}")

    return dict(f_max_abs=float(f_max_abs), x_max=float(cfg['x_max']), runs=runs,
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
# All one weight: where the curves collapse in panel (a) the dashes let the solid curve
# underneath show through, so the overlap still reads without a heavier line.
RABI_STYLES = [('-', 2.0), ('--', 2.0), (':', 2.0), ('-.', 2.0)]

# One place to set every text size in the figure.
FS_SUPTITLE = 17
FS_PANEL_TITLE = 15
FS_AXIS_LABEL = 14
FS_TICK = 12
FS_LEGEND = 12
FS_ANNOT = 11


def make_figure(data, cfg, path):
    runs = data['runs']
    x_max = data.get('x_max', cfg['x_max'])
    f_max_abs = data.get('f_max_abs', x_max * max(r['f_Rabi'] for r in runs))

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 6.0))
    ax_n, ax_a = axes

    for ax in axes:
        ax.set_yscale('log')
        ax.set_ylabel(r'Intensity-noise response  $I_I(\omega)$', fontsize=FS_AXIS_LABEL)
        ax.grid(True, which='major', color=GRID, linewidth=0.7, zorder=0)
        ax.grid(True, which='minor', color=GRID, linewidth=0.35, alpha=0.6, zorder=0)
        ax.set_axisbelow(True)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        for side in ('left', 'bottom'):
            ax.spines[side].set_color('#c3c2b7')
        ax.tick_params(colors=MUTED, labelsize=FS_TICK)

    # Beams outer, Rabi inner: with a 3-column legend (filled column-major) each beam
    # then gets its own column, with its Rabi frequencies stacked underneath it.
    for key, color, name, _short in BEAMS:
        for i, run in enumerate(runs):
            fR = run['f_Rabi']
            ls, lw = RABI_STYLES[i % len(RABI_STYLES)]
            f_abs = np.asarray(run['f_abs'])
            y = np.asarray(run[key])
            lab = rf'{name}, $\Omega/2\pi={fR:g}$ MHz'
            # (a) this run's own normalized axis; (b) the shared absolute one. Both are
            # the same data -- panel (a) is just clipped by set_xlim to x_max, while the
            # slower runs carry points beyond it so that panel (b) reaches f_max_abs.
            ax_n.plot(f_abs / fR, y, color=color, lw=lw, ls=ls, zorder=3, label=lab)
            ax_a.plot(f_abs, y, color=color, lw=lw, ls=ls, zorder=3, label=lab)

    ax_n.set_xlim(0, x_max)
    ax_n.set_xlabel(r'Normalized noise frequency  $2\pi f/\Omega$', fontsize=FS_AXIS_LABEL)
    ax_n.set_title('(a)  Universal filter shape', loc='left', color=INK,
                   fontsize=FS_PANEL_TITLE, fontweight='bold')
    # -3 dB point of each beam, from the lowest-Rabi run (they agree across runs). The
    # dotted verticals sit in the empty band between the arm-2 curve and the other two;
    # the values are quoted in the annotation rather than labelled on the axis, which
    # would collide with the panel title.
    cuts = []
    x0 = np.asarray(runs[0]['f_abs']) / runs[0]['f_Rabi']
    for key, color, _name, short in BEAMS:
        xh = _half_power(x0, np.asarray(runs[0][key]))
        if np.isfinite(xh):
            ax_n.axvline(xh, color=color, lw=1.0, ls=(0, (1, 2)), zorder=2)
            cuts.append((xh, short))
    cuts.sort()

    # ax_n.annotate('the two Rabi frequencies collapse — the filter is\n'
    #               'set by the pulse, not the absolute frequency.\n'
    #               'The three beams do not: each has its own width.\n'
    #               + r'$-3$ dB at $2\pi f/\Omega$ = '
    #               + ', '.join(f'{v:.2f} ({nm})' for v, nm in cuts),
    #               xy=(0.03, 0.74), xycoords='axes fraction', ha='left', va='top',
    #               fontsize=FS_ANNOT, color=MUTED, style='italic')

    # ax_a.set_xscale('log')
    ax_a.set_xlim(0, f_max_abs)         # every curve now spans this full range
    ax_a.set_xlabel(r'Noise frequency  $f$  [MHz]', fontsize=FS_AXIS_LABEL)
    ax_a.set_title('(b)  Same curves vs absolute noise frequency', loc='left', color=INK,
                   fontsize=FS_PANEL_TITLE, fontweight='bold')
    # ax_a.annotate('a faster gate pushes the roll-off out,\nso it integrates RIN over '
    #               'a wider band',
    #               xy=(0.03, 0.05), xycoords='axes fraction', ha='left', va='bottom',
    #               fontsize=FS_ANNOT, color=MUTED, style='italic')

    handles, labels = ax_n.get_legend_handles_labels()
    fig.legend(handles, labels, ncol=len(BEAMS), fontsize=FS_LEGEND, frameon=False,
               labelcolor=MUTED, loc='lower center', bbox_to_anchor=(0.5, -0.02),
               handlelength=2.6, columnspacing=2.2)

    # sub = (rf"Cs, $n={cfg['n']}$, $d={cfg['atom_d']}\,\mu$m  |  "
    #        rf"1-photon $n$P$_{{3/2}}$ ($B/2\pi={data['blockade_1p_MHz']:.0f}$ MHz), "
    #        rf"2-photon $n$S$_{{1/2}}$ ($B/2\pi={data['blockade_2p_MHz']:.0f}$ MHz, "
    #        rf"$\Delta/2\pi={cfg['inter_detuning'] / 2 / np.pi / 1e3:.0f}$ GHz, per beam)  |  "
    #        rf"time-optimal CZ, $\Omega T={cfg['pulse_time']:g}$")
    fig.suptitle('Intensity-noise response vs noise frequency', x=0.008, y=1.0,
                 ha='left', va='top', fontsize=FS_SUPTITLE, fontweight='bold', color=INK)
    # fig.text(0.008, 0.95, sub, ha='left', va='top', fontsize=FS_ANNOT, color=MUTED)
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
    p.add_argument('--inter-detuning', type=float, default=10000.0,
                   help='2-photon intermediate-state detuning Delta/2pi [MHz]')
    p.add_argument('--pulse-time', type=float, default=7.65, help='gate duration in 1/Omega')
    # 1000, not the 200 the other budget scripts use. The extended panel (b) reaches
    # 2 pi f / Omega ~ 30 for the slowest gate, and the kernel's rectangle rule needs
    # omega*dt << 1 there. Measured against a resolution-4000 reference: through the
    # knee (x <~ 3) both 200 and 1000 are fine (<0.1%), but in the deep tail 200 is off
    # by ~60% at x = 30 while 1000 holds to a few % -- where the response is already
    # four decades below DC, so a few % is immaterial. Affordable only because
    # response_*_spectrum builds the O(Nt^2) kernel once per curve instead of per point.
    p.add_argument('--resolution', type=int, default=1000, help='phase-profile time steps')
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
