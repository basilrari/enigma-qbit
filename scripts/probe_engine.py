#!/usr/bin/env python3
"""Run the engine's own absorb loop with every contraction call instrumented.

The naive walk (probe_nan.py) saturates the bond because it bypasses the
unswap logic. This runs the REAL loop -- mpo_compress_unswap as the orchestrator
calls it -- and logs, for every apply_circuit/apply_swaps invocation: which
stage in the engine made the call, the resulting bond, max|entry| and the count
of non-finite entries. Aborts at the first non-finite tensor, so the failure
site is reported rather than buried in a traceback.

Run on the box: cd {V} && {PY} {W}/probe_engine.py --qasm d2_s1_39b370e4.qasm
"""
import argparse
import os
import sys
import time
import traceback

import numpy as np

L2 = "/mnt/8tb_hdd2/basilrari/enigma-work/l2win"
sys.path.insert(0, L2)

from qiskit import QuantumCircuit, qasm2                              # noqa: E402
from qiskit.circuit.library import UnitaryGate                        # noqa: E402
from qiskit.quantum_info import Operator                              # noqa: E402
import quimb.tensor as qtn                                            # noqa: E402

import utils as UT                                                    # noqa: E402
import unswap_stallfix as US                                          # noqa: E402

Circuit = qtn.Circuit
ENGINE_FILES = ("unswap_stallfix.py", "unswap.py", "circuit_mpo.py", "utils.py")


def load_qasm(path):
    import re
    txt = open(path).read()
    if "OPENQASM 3" in txt:
        txt = txt.replace("OPENQASM 3.0;", "OPENQASM 2.0;")
        txt = txt.replace("stdgates.inc", "qelib1.inc")
        txt = re.sub(r"qubit\[(\d+)\]\s+(\w+)\s*;", r"qreg \2[\1];", txt)
    txt = re.sub(r"(?<![A-Za-z0-9_])u\s*\(", "u3(", txt)
    return qasm2.loads(txt)


def unitarize(qc):
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
    worst, bad = 0.0, 0
    for t in mpo.tensors:
        a = np.abs(np.asarray(t.data))
        nf = int((~np.isfinite(a)).sum())
        bad += nf
        if a.size:
            f = a[np.isfinite(a)]
            if f.size:
                worst = max(worst, float(f.max()))
            if nf:
                worst = float("inf")
    return worst, bad


def bondmax(mpo):
    try:
        return max(int(mpo.bond_size(i, i + 1)) for i in range(len(mpo.sites) - 1))
    except Exception:
        return -1


def stage():
    """which engine function is calling us (2 frames up)"""
    for fr in reversed(traceback.extract_stack()[:-2]):
        if fr.filename.endswith(ENGINE_FILES):
            return f"{os.path.basename(fr.filename).replace('.py', '')}:{fr.name}"
    return "?"


CALLS = {"n": 0}
ORIG_APPLY = US.apply_circuit
ORIG_SWAPS = getattr(US, "apply_swaps", None)


def logged_apply(mpo, circ, *a, **kw):
    CALLS["n"] += 1
    pre = fin(mpo)
    t0 = time.time()
    out = ORIG_APPLY(mpo, circ, *a, **kw)
    post = fin(out)
    bm = bondmax(out)
    n = CALLS["n"]
    if post[1] > 0 or n % 20 == 0 or bm >= 900:
        print(f"  call {n:5d} {stage():26s} bondmax={bm:5d} "
              f"max|.| {pre[0]:.2e}->{post[0]:.2e} nf {post[1]:4d} "
              f"[{time.time() - t0:5.1f}s t={time.time() - T0[0]:6.1f}s]", flush=True)
    if post[1] > 0:
        raise SystemExit(f"### NON-FINITE after apply call {n} at stage {stage()} "
                         f"(bondmax={bm})")
    return out


def logged_swaps(mpo, *a, **kw):
    CALLS["n"] += 1
    pre = fin(mpo)
    out = ORIG_SWAPS(mpo, *a, **kw)
    post = fin(out)
    bm = bondmax(out)
    n = CALLS["n"]
    if post[1] > 0 or bm >= 900:
        print(f"  call {n:5d} {stage():26s} SWAPS bondmax={bm:5d} "
              f"max|.| {pre[0]:.2e}->{post[0]:.2e} nf {post[1]:4d}", flush=True)
    if post[1] > 0:
        raise SystemExit(f"### NON-FINITE after swap call {n} at stage {stage()} "
                         f"(bondmax={bm})")
    return out


T0 = [time.time()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qasm", required=True)
    ap.add_argument("--max-bond", type=int, default=1024)
    ap.add_argument("--cutoff", type=float, default=0.002)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--center-ratio", type=float, default=0.5)
    ap.add_argument("--max-seconds", type=float, default=420.0)
    args = ap.parse_args()

    T0[0] = time.time()
    qc = load_qasm(args.qasm)
    n = qc.num_qubits
    uqc = unitarize(qc)
    print(f"# {args.qasm}: {n} qubits, {len(uqc)} unitary gates")
    print(f"# max_bond={args.max_bond} cutoff={args.cutoff} seed={args.seed} "
          f"center_ratio={args.center_ratio} cap={args.max_seconds:.0f}s", flush=True)

    US.apply_circuit = logged_apply
    if ORIG_SWAPS is not None:
        US.apply_swaps = logged_swaps

    try:
        mpo, L_left, L_right, stats = US.mpo_compress_unswap(
            uqc, max_bond=args.max_bond, cutoff=args.cutoff,
            unswap_threshold=1e6, center_ratio=args.center_ratio,
            to_backend=None, seed=args.seed,
            deadline=time.time() + args.max_seconds)
        print(f"\n# ABSORB COMPLETED in {time.time() - T0[0]:.1f}s")
        print(f"# bondmax={bondmax(mpo)} max|.|={fin(mpo)[0]:.3e} nf={fin(mpo)[1]}")
        print(f"# leftover layers L={len(L_left)} R={len(L_right)} "
              f"calls={CALLS['n']}")
        print(f"# stats keys: {list(stats.keys()) if isinstance(stats, dict) else type(stats)}")
        bl = [int(mpo.bond_size(i, i + 1)) for i in range(len(mpo.sites) - 1)]
        print(f"# bond profile: {sorted(bl, reverse=True)[:12]}")
    except SystemExit as exc:
        print(str(exc))
        print(f"# aborted at t={time.time() - T0[0]:.1f}s after {CALLS['n']} calls")
        return 1
    except Exception as exc:
        print(f"# EXCEPTION {type(exc).__name__}: {str(exc)[:300]} "
              f"at t={time.time() - T0[0]:.1f}s after {CALLS['n']} calls")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
