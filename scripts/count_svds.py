#!/usr/bin/env python3
"""Attribute every SVD in the MPS build to its caller.

The profiler told us the build spends ~70% of its time in SVD-like calls and
makes ~11 of them per gate, which is far more than the algorithm should need.
The profiler cannot say WHO calls them.  This does: it wraps the SVD family
and the build's structural helpers and records, per call site, how many calls
happen and what shapes and ranks they use.  Run on a cheap instance (chi=64)
because call COUNTS per gate do not depend on chi.
"""
import collections
import os
import sys
import time

import numpy as np

V = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, V)
import hqp_solver as H  # noqa: E402

C = collections.Counter()
TM = collections.defaultdict(float)


def wrap(name):
    orig = getattr(H, name)

    def inner(*a, **k):
        caller = sys._getframe(1).f_code.co_name
        M = a[0] if a else k.get("M")
        kk = a[1] if len(a) > 1 else k.get("k")
        key = (name, caller, str(getattr(M, "shape", "?")), str(kk))
        C[key] += 1
        t0 = time.time()
        try:
            return orig(*a, **k)
        finally:
            TM[key] += time.time() - t0

    setattr(H, name, inner)


for nm in ("_svd", "_svd_topk", "_svd_full", "_apply_swap", "_apply_2site",
           "_move_center_left", "_move_center_right", "_bring_2q"):
    if hasattr(H, nm):
        wrap(nm)

qasm = sys.argv[1] if len(sys.argv) > 1 else "d2_s1_39b370e4.qasm"
chi = int(sys.argv[2]) if len(sys.argv) > 2 else 64
circ = H.load_circuit(os.path.join(V, qasm))
n_gates = len(circ.data)
t0 = time.time()
tensors, order, q_at = H.build_mps(circ, chi, verbose=False)
bt = time.time() - t0

print(f"build {bt:.1f}s  chi={chi}  gates={n_gates}  "
      f"gates/s={n_gates / bt:.1f}")
print("\n--- SVD family, by caller (sorted by total time) ---")
for key, n in sorted(C.items(), key=lambda kv: -TM[kv[0]]):
    if not key[0].startswith("_svd"):
        continue
    print(f"{key[0]:<12} <- {key[1]:<18} shape={key[2]:<12} k={key[3]:<5} "
          f"n={n:<6} {TM[key]:6.1f}s")
print("\n--- structural helpers ---")
for nm in ("_apply_swap", "_apply_2site", "_move_center_left",
           "_move_center_right", "_bring_2q"):
    tot = sum(n for k, n in C.items() if k[0] == nm)
    tt = sum(v for k, v in TM.items() if k[0] == nm)
    if tot:
        print(f"{nm:<20} {tot:6d} calls  {tt:6.1f}s  "
              f"({tot / n_gates:.2f} per gate)")
