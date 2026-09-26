"""
Rydberg blockade for an ASYMMETRIC pair |nP3/2 mj=m1 ; nP3/2 mj=m2>
(default m1=1/2, m2=3/2), in the same format as blockades_symmetric_p.pkl
so the budget's find_blockade_Mrad() can read it unchanged:

    blockade_dict[atom_name][str(n)] = [(r_um, B_GHz), ...]

Blockade definition (matches blockades_symmetric_p.pkl to <0.5% for r > 3.5 um,
checked for Cs n=70, mj=3/2+3/2, theta=0):

    B_eff^-2 = sum_k |<k|r1 r2>|^2 / E_k^2 = || H^-1 |r1 r2> ||^2

where H is ARC's pair Hamiltonian relative to the unperturbed pair energy.
It is evaluated with one sparse linear solve per r, which equals the sum over the FULL
spectrum. ARC's diagonalise() keeps only a few ARPACK eigenvectors, which misses most
of the pair-state weight near Förster resonances (r < ~3 um).

For m1 != m2 the pair |m1,m2> is degenerate with |m2,m1>. The two atoms are
distinguishable (separate tweezers), so ARC keeps both orderings in the basis, and
the vdW coupling between them is included in H. B_eff is therefore the
residual-double-excitation blockade seen from the prepared state |m1,m2>.

Usage
-----
  # one n per job (what the cluster runs)
  python blockade_asymmetric.py --atom Cs --n 70 --outdir blockade_asym_out

  # merge all per-n files into the budget-format pickle
  python blockade_asymmetric.py --merge blockade_asym_out --output blockades_asymmetric_p.pkl

Use --m1 1.5 --m2 1.5 to regenerate the symmetric file with the same code.
--conv also computes B with a larger basis (nrange+1, lmax+1, 1.5*dE) so you can
see where the result is basis-converged (short range usually is NOT).
"""
import argparse
import glob
import os
import pickle
import sys
import time

import numpy as np

# SciPy >= 1.17 removed scipy.special.sph_harm, which ARC 3.8.1 imports.
# (Same shim as sitecustomize.py, inlined so the cluster job needs no extra file.)
try:
    import scipy.special as _sp
    if not hasattr(_sp, "sph_harm") and hasattr(_sp, "sph_harm_y"):
        def _sph_harm(m, n, theta, phi):
            return _sp.sph_harm_y(n, m, phi, theta)
        _sp.sph_harm = _sph_harm
except Exception:
    pass

from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve


def get_atom(name):
    from arc import Caesium, Rubidium
    return {"Cs": Caesium, "Rb": Rubidium}[name]()


def build_calc(atom, n, m1, m2, theta, Bz_T, nrange, lmax, dE_GHz):
    from arc import PairStateInteractions
    calc = PairStateInteractions(atom, n, 1, 1.5, n, 1, 1.5, m1, m2)
    calc.defineBasis(theta, 0.0, nrange, lmax, dE_GHz * 1e9, Bz=Bz_T)
    return calc


def blockade_eff_GHz(calc, r_um):
    """B_eff(r) = 1/||H(r)^-1 |r1 r2>||, H in GHz (ARC units)."""
    i0 = calc.originalPairStateIndex
    e = np.zeros(calc.matDiagonal.shape[0])
    e[i0] = 1.0
    B = np.empty(len(r_um))
    for k, r in enumerate(r_um):
        m = calc.matDiagonal
        rX = (r * 1e-6) ** 3
        for matRX in calc.matR:
            m = m + matRX / rX
            rX *= r * 1e-6
        x = spsolve(csc_matrix(m), e)
        B[k] = 1.0 / np.linalg.norm(x)
    return B


def run_one(args):
    atom = get_atom(args.atom)
    r_um = np.linspace(args.rmin, args.rmax, args.nr)
    Bz_T = args.Bz / 1e4
    theta = np.deg2rad(args.theta)
    out = {
        "atom": args.atom, "n": args.n, "m1": args.m1, "m2": args.m2,
        "r": r_um,
        "params": dict(theta_deg=args.theta, Bz_G=args.Bz, nrange=args.nrange,
                       lmax=args.lmax, dE_GHz=args.dE),
    }

    t0 = time.time()
    calc = build_calc(atom, args.n, args.m1, args.m2, theta, Bz_T,
                      args.nrange, args.lmax, args.dE)
    out["dim"] = len(calc.basisStates)
    out["B"] = blockade_eff_GHz(calc, r_um)
    print(f"[{args.atom} n={args.n}] basis dim {out['dim']}, "
          f"{time.time() - t0:.0f} s", flush=True)

    if args.conv:
        t0 = time.time()
        calc2 = build_calc(atom, args.n, args.m1, args.m2, theta, Bz_T,
                           args.nrange + 1, args.lmax + 1, 1.5 * args.dE)
        out["dim_conv"] = len(calc2.basisStates)
        out["B_conv"] = blockade_eff_GHz(calc2, r_um)
        out["params_conv"] = dict(nrange=args.nrange + 1, lmax=args.lmax + 1,
                                  dE_GHz=1.5 * args.dE)
        print(f"[{args.atom} n={args.n}] conv basis dim {out['dim_conv']}, "
              f"{time.time() - t0:.0f} s", flush=True)

    os.makedirs(args.outdir, exist_ok=True)
    fn = os.path.join(args.outdir, f"{args.atom}_n{args.n}_m{args.m1:g}_{args.m2:g}.pkl")
    with open(fn, "wb") as f:
        pickle.dump(out, f)
    for r, b in zip(r_um[::max(1, args.nr // 10)], out["B"][::max(1, args.nr // 10)]):
        print(f"  r={r:6.3f} um  B={b * 1e3:10.3f} MHz")
    print("saved", fn)


def merge(args):
    """Per-n files -> {atom: {str(n): [(r, B_GHz), ...]}} (budget format)."""
    files = sorted(glob.glob(os.path.join(args.merge, "*.pkl")))
    if not files:
        sys.exit(f"no .pkl files in {args.merge}")
    blockade = {}
    conv = {}
    meta = None
    for fn in files:
        with open(fn, "rb") as f:
            d = pickle.load(f)
        key = (d["m1"], d["m2"])
        if meta is None:
            meta = dict(m1=d["m1"], m2=d["m2"], definition="B_eff = ||H^-1|r1r2>||^-1, GHz", **d["params"])
        elif key != (meta["m1"], meta["m2"]):
            sys.exit(f"{fn}: mixed mj pairs {key} vs {(meta['m1'], meta['m2'])}")
        blockade.setdefault(d["atom"], {})[str(d["n"])] = list(zip(d["r"], d["B"]))
        if "B_conv" in d:
            conv.setdefault(d["atom"], {})[str(d["n"])] = list(zip(d["r"], d["B_conv"]))
    blockade["_meta"] = meta
    with open(args.output, "wb") as f:
        pickle.dump(blockade, f)
    print(f"wrote {args.output}: " + ", ".join(
        f"{a}: n={min(map(int, v))}..{max(map(int, v))} ({len(v)})"
        for a, v in blockade.items() if a != "_meta"))
    if conv:
        conv["_meta"] = meta
        fn_conv = os.path.splitext(args.output)[0] + "_conv.pkl"
        with open(fn_conv, "wb") as f:
            pickle.dump(conv, f)
        print("wrote", fn_conv, "(larger-basis convergence check, same format)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--atom", default="Cs", choices=["Cs", "Rb"])
    p.add_argument("--n", type=int, help="principal quantum number of both atoms")
    p.add_argument("--m1", type=float, default=0.5, help="mj of atom 1 (nP3/2)")
    p.add_argument("--m2", type=float, default=1.5, help="mj of atom 2 (nP3/2)")
    p.add_argument("--theta", type=float, default=0.0,
                   help="angle (deg) between quantization axis and interatomic axis")
    p.add_argument("--Bz", type=float, default=0.0, help="B field along z (Gauss)")
    p.add_argument("--nrange", type=int, default=5)
    p.add_argument("--lmax", type=int, default=4)
    p.add_argument("--dE", type=float, default=25.0, help="pair-energy window (GHz)")
    p.add_argument("--rmin", type=float, default=1.5)
    p.add_argument("--rmax", type=float, default=7.0)
    p.add_argument("--nr", type=int, default=100)
    p.add_argument("--conv", action="store_true", help="also run a larger basis")
    p.add_argument("--outdir", default="blockade_asym_out")
    p.add_argument("--merge", metavar="DIR", help="merge per-n pickles in DIR")
    p.add_argument("--output", default="blockades_asymmetric_p.pkl")
    args = p.parse_args()
    if args.merge:
        merge(args)
    elif args.n is None:
        p.error("--n is required (or use --merge)")
    else:
        run_one(args)
