# Optimizer runs

Raw output of `budget_1photon_optimize.py` and `budget_2photon_optimize.py`. Each JSON holds the
best point, its channel breakdown, the bounds, the full config and every objective evaluation
(`history`), so a run can be audited or restarted without recomputing.

Common to every run: Cs, Bz = 10 G, T = 1 uK, 200 phase steps, pulse_time = 7.65/Omega,
Monte-Carlo seed 1234. Numbers are infidelities; "scattering" is not modelled in the 1-photon
budget and "leakage" is not modelled in the 2-photon one.

## 1-photon (`1photon/`)

`rin_strength = 1e-5` throughout (1e-4, the scan-script default, puts a hard ~1e-4 floor in the
RIN channel on its own and makes the target unreachable — see `scan_n95_rin1e-5.json` for the
Omega sweep that shows it).

| file | E-field noise | best total | n | d [um] | Omega/2pi |
|---|---|---|---|---|---|
| `opt_edc1mVcm_rin1e-5.json`   | 1 mV/cm   | 1.055e-4 | 89 | 2.54 | 27.7 MHz |
| `opt_edc0.3mVcm_rin1e-5.json` | 0.3 mV/cm | 8.66e-5  | 99 | 1.73 | 23.5 MHz |

`scan_n99_omega_sweep.json` is the Omega sweep at the second point (10 000 MC samples).

Limits at the optimum: Rydberg decay ~4.3e-5, mj=1/2 leakage ~2.6e-5, RIN ~1.1e-5. Getting
below 1e-4 needs BOTH ~0.3% RMS intensity stabilisation and ~0.3 mV/cm field control.

## 2-photon (`2photon/`)

Cs 6S1/2 -> 7P1/2 (459 nm) -> nS1/2 (1038 nm), `rin_strength = 1e-4` unless noted. All runs
seeded from a common feasible point so the power comparison is apples to apples (Powell is
start-point sensitive; see the `--restarts` flag).

| file | P(459) / P(1038) | best total | n | Omega_eff/2pi | Delta/2pi | P(1038) used |
|---|---|---|---|---|---|---|
| `opt_P1038_0.25W.json`          | 200 mW / 0.25 W | 2.829e-3 | 60 |  6.6 MHz | 1.31 GHz | 100% |
| `opt_P1038_1W.json`             | 200 mW / 1 W    | 1.961e-3 | 49 | 17.1 MHz | 2.25 GHz |  73% |
| `opt_P1038_5W.json`             | 200 mW / 5 W    | 1.160e-3 | 50 | 23.6 MHz | 1.21 GHz |  96% |
| `opt_P459_50mW.json`            | 50 mW / 5 W     | 2.204e-3 | 54 | 23.6 MHz | 1.21 GHz |  84% |
| `opt_P1038_5W_rin1e-5.json`     | 200 mW / 5 W    | 1.120e-3 | 50 | 23.6 MHz | 1.21 GHz |  95% |
| `opt_P1038_5W_widebounds.json`  | 200 mW / 5 W    | 1.144e-3 | 51 | 23.6 MHz | 1.21 GHz | 100% |

The power ceilings are PLACEHOLDERS (200 mW at 459 nm, 5 W at 1038 nm); rerun with the real
numbers via `--p459-mW` / `--p1038-W`.

Limits at the 5 W optimum: 7P scattering 7.5e-4, Rydberg decay 3.4e-4, RIN 4.9e-5. Dropping RIN
tenfold buys only 3% (1.160e-3 -> 1.120e-3) because scattering and decay dominate. Widening
Omega_eff to 60 MHz, the waists to 100 um and Delta down to 0.3 GHz buys 1.4%, so the optimum is
set by the power ceiling and by the S-state blockade table's 2 um lower edge, not by the other
bounds.
