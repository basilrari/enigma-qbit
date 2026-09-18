#!/usr/bin/env python3
"""Walk the absorb layer by layer and pinpoint the non-finite blow-up.

Mirrors the engine's own setup exactly (same merge/rewire/order), then applies
left layers one at a time with the same call the crashing loop makes, printing
max|entry| and the count of non-finite entries before and after each step.

Also sanity-checks the layers themselves for non-unitarity, because a gate whose
matrix is not unitary would poison everything downstream.

Run on the box:  cd {V} && {PY} {W}/probe_nan.py --qasm d2_s1_39b370e4.qasm
"""
import argparse
import sys
import time

import numpy as np

L2 = "/mnt/8tb_hdd2/basilrari/enigma-work/l2win"
sys.path.insert(0, L2)

from qiskit import QuantumCircuit, qasm2                              # noqa: E402
from qiskit.circuit.library import UnitaryGate                        # noqa: E402
from qiskit.quantum_info import Operator                              # noqa: E402
import quimb.tensor as qtn                                            # noqa: E402

import utils as UT                                                    # noqa: E402
import unswap_stallfix as US                                          # noqa: E402
from utils import quimb_circuit, iter_layers, merge_gates, get_tn_info  # noqa: E402

Circuit = qtn.Circuit


def load_qasm(path):
    """Verbatim from l2_absorb.py (u->u3 rewrite, QASM3 header tolerated)."""
    import re
    txt = open(path).read()
    if "OPENQASM 3" in txt:
        txt = txt.replace("OPENQASM 3.0;", "OPENQASM 2.0;")
        txt = txt.replace("stdgates.inc", "qelib1.inc")
        txt = re.sub(r"qubit\[(\d+)\]\s+(\w+)\s*;", r"qreg \2[\1];", txt)
    txt = re.sub(r"(?<![A-Za-z0-9_])u\s*\(", "u3(", txt)
    return qasm2.loads(txt)


def unitarize(qc):
    """Verbatim from l2_absorb.py."""
    skip = {"measure", "barrier", "reset", "delay", "id"}
    out = QuantumCircuit(qc.num_qubits)
    for inst in qc.data:
        if inst.operation.name in skip:
            continue
        qidx = [qc.find_bit(q).index for q in inst.qubits]
        out.append(UnitaryGate(Operator(inst.operation).data,
                               label=inst.operation.name), qidx)
    return out


def fin(mpo):
    """(max|entry|, non-finite count, total entries) across the MPO tensors."""
    worst, bad, tot = 0.0, 0, 0
    for t in mpo.tensors:
        a = np.abs(np.asarray(t.data))
        tot += a.size
        nf = int((~np.isfinite(a)).sum())
        bad += nf
        if a.size:
            f = a[np.isfinite(a)]
            if f.size:
                worst = max(worst, float(f.max()))
            if nf:
                worst = float("inf")
    return worst, bad, tot


def layer_unit_check(layers, k, limit=4):
    """Report any gate in the first k layers whose matrix is not unitary."""
    worst = 0.0
    bad_n = 0
    for li, lay in enumerate(layers[:k]):
        for inst in getattr(lay, "data", []):
            try:
                M = np.asarray(Operator(inst.operation).data)
            except Exception as exc:
                print(f"    layer {li}: cannot build operator: {exc}")
                continue
            d = M.shape[0]
            dev = float(np.abs(M.conj().T @ M - np.eye(d)).max())
            mx = float(np.abs(M).max())
            if dev > 1e-9 or not np.isfinite(mx):
                bad_n += 1
                if bad_n <= limit:
                    print(f"    NON-UNITARY layer {li} op={inst.operation.name} "
                          f"qubits={[q._index for q in inst.qubits]} "
                          f"||U+U-I||={dev:.3e} max|M|={mx:.3e}")
            worst = max(worst, dev)
    return worst, bad_n


def renorm(mpo, logscale):
    """Divide the MPO by its largest entry, tracking the scale exactly.

    Scaling an MPO by a global constant c scales every amplitude by c, so the
    factor is recoverable at the end from logscale. This keeps tensors O(1) and
    stops the runaway growth that overflows to inf on deep circuits.
    """
    s = 0.0
    for t in mpo.tensors:
        a = np.abs(np.asarray(t.data))
        f = a[np.isfinite(a)]
        if f.size:
            s = max(s, float(f.max()))
    if s > 0 and np.isfinite(s):
        t0 = mpo.tensors[0]
        t0.modify(data=t0.data / s)
        logscale += np.log(s)
    return mpo, logscale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qasm", required=True)
    ap.add_argument("--layers", type=int, default=6, help="left layers to apply")
    ap.add_argument("--center-ratio", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--sweep", action="store_true",
                    help="try several (max_bond, cutoff) pairs")
    ap.add_argument("--renorm", action="store_true",
                    help="rescale to O(1) after every layer (scale tracked)")
    ap.add_argument("--seed-race", default="",
                    help="comma-separated rewire seeds; report bond growth per seed")
    args = ap.parse_args()

    combos = [(1024, 0.002), (512, 0.002), (256, 0.002), (1024, 0.01), (256, 0.01)]
    if not args.sweep:
        combos = combos[:1]

    qc = load_qasm(args.qasm)
    n = qc.num_qubits
    uqc = unitarize(qc)
    C = int(len(uqc) * args.center_ratio)
    print(f"# {args.qasm}: {n} qubits, {len(uqc)} unitary gates, C={C}")
    print(f"# tn_info(empty core) sanity: n/a")

    circ_left = merge_gates(uqc[:C], uqc.num_qubits).inverse()
    if "measure" not in circ_left.count_ops():
        circ_left.measure_all()
    layers = list(iter_layers(circ_left))
    layers = US.rewire_layers(layers, np.arange(n, dtype=int), seed=args.seed)
    layers = layers[:-2]
    print(f"# left layers after rewire/drop: {len(layers)}")

    print("# --- layer unitarity scan (first {} layers) ---".format(args.layers))
    worst, bad_n = layer_unit_check(layers, args.layers)
    print(f"# worst ||U+U-I||={worst:.3e}  non-unitary gates={bad_n}")

    q2c = lambda c: quimb_circuit(c.decompose("unitary"), Circuit, to_backend=None)  # noqa: E731

    if args.seed_race:
        base = np.arange(n, dtype=int)
        seeds = [int(x) for x in args.seed_race.split(",") if x.strip()]
        marks_at = (3, 7, 11, 15)
        for sd in seeds:
            lay_s = US.rewire_layers([l.copy() for l in iter_layers(circ_left)],
                                     base.copy(), seed=sd)[:-2]
            mpo = US.mpo_from_circuit(q2c(QuantumCircuit(n)))
            marks = []
            t0 = time.time()
            blown = False
            top = min(args.layers, len(lay_s))
            for k in range(top):
                mpo = US.apply_circuit(mpo, q2c(lay_s[k].inverse()),
                                       side="right", max_bond=1024, cutoff=0.002)
                bm = max(int(mpo.bond_size(i, i + 1)) for i in range(len(mpo.sites) - 1))
                if (k + 1) in marks_at:
                    marks.append(f"L{k + 1}:{bm}")
                if bm >= 1024:
                    marks.append(f"L{k + 1}:{bm}*")
                    blown = True
                    break
            tail = "BOND CAP HIT" if blown else "ok"
            print(f"# seed {sd:>6}: {' '.join(marks)} | {time.time() - t0:6.1f}s | {tail}",
                  flush=True)
        return

    for (mb, cut) in combos:
        print(f"\n### SWEEP max_bond={mb} cutoff={cut} renorm={args.renorm}")
        t0 = time.time()
        logscale = 0.0
        try:
            mpo = US.mpo_from_circuit(q2c(QuantumCircuit(n)))
            print(f"  core {fin(mpo)[:2]} tn={get_tn_info(mpo)}")
            ok = True
            last = min(args.layers, len(layers))
            for k in range(last):
                pre = fin(mpo)
                mpo = US.apply_circuit(mpo, q2c(layers[k].inverse()),
                                       side="right", max_bond=mb, cutoff=cut)
                post = fin(mpo)
                if args.renorm:
                    mpo, logscale = renorm(mpo, logscale)
                bonds = [int(mpo.bond_size(i, i + 1)) for i in range(len(mpo.sites) - 1)]
                if k < 12 or k == last - 1 or post[1] > 0 or (k % 10 == 9):
                    print(f"  L{k:03d} gates={len(getattr(layers[k], 'data', [])):2d} "
                          f"max|.| {pre[0]:.2e}->{post[0]:.2e} nf {pre[1]}->{post[1]} "
                          f"bondmax={max(bonds) if bonds else 0} "
                          f"logscale={logscale:.1f} [{time.time() - t0:.1f}s]", flush=True)
                if post[1] > 0:
                    print(f"  !!! NON-FINITE appeared at left layer {k} "
                          f"(max_bond={mb}, cutoff={cut}, renorm={args.renorm})")
                    ok = False
                    break
            if ok:
                print(f"  --> survived {last} layers, elapsed {time.time() - t0:.1f}s, "
                      f"total logscale={logscale:.2f} (= 10^{logscale / np.log(10):.1f}) "
                      f"final max|.|={fin(mpo)[0]:.2e}")
        except Exception as exc:
            print(f"  EXCEPTION at max_bond={mb} cutoff={cut}: {type(exc).__name__}: {exc}")
            continue


if __name__ == "__main__":
    sys.exit(main())
