#!/usr/bin/env python3
"""Decisive test: does the circuit contain gates paired with their own inverses?

Comparing rotation angles is convention-dependent (a 2pi shift or swapped
phi/lam changes the numbers but not the gate).  Comparing *matrices up to
global phase* is convention-proof: two gates are mutual inverses iff
M_i . M_j is proportional to the identity.

Build a phase-free canonical form for every u gate, then look up each gate's
inverse in that table.  If the circuit is U then U*, every gate in U must find
its partner in U* -- no matter how the disguise relabels the wires.
"""
import collections
import math
import re
import sys

import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "/home/basilsclaw/enigma-solve/samples/d2_s1_39b370e4.qasm"
lines = open(path).read().splitlines()


def _num(tok):
    tok = tok.strip()
    try:
        return float(tok)
    except ValueError:
        return float(eval(tok.replace("pi", repr(math.pi)), {"__builtins__": {}}, {}))


def u_matrix(t, p, l):
    """Qiskit u(theta, phi, lam) = Rz(phi) Ry(theta) Rz(lam)."""
    c, s = math.cos(t / 2.0), math.sin(t / 2.0)
    return np.array([
        [c, -np.exp(1j * l) * s],
        [np.exp(1j * p) * s, np.exp(1j * (p + l)) * c],
    ], dtype=complex)


def canon(M, tol=1e-9):
    """Phase-free key: divide by the largest-magnitude element, then round."""
    A = np.asarray(M, dtype=complex)
    k = np.unravel_index(np.argmax(np.abs(A)), A.shape)
    if abs(A[k]) < 1e-12:
        return ("zero",)
    B = A / A[k]
    return tuple(round(v, 6) for v in np.concatenate([B.real.ravel(), B.imag.ravel()]))


UG = re.compile(r"^u\(([^)]*)\)\s+q\[(\d+)\];")
us, czs = [], []
for l in (x.strip() for x in lines):
    m = UG.match(l)
    if m:
        q = [_num(v) for v in m.group(1).split(",")]
        us.append((q[0], q[1], q[2], int(m.group(2))))
        continue
    m = re.match(r"^cz\s+q\[(\d+)\],q\[(\d+)\];", l)
    if m:
        czs.append((int(m.group(1)), int(m.group(2))))

print(f"u gates: {len(us)}   cz gates: {len(czs)}")

fwd = {}
for i, (t, p, l, _q) in enumerate(us):
    fwd.setdefault(canon(u_matrix(t, p, l)), []).append(i)

pairs = []
for i, (t, p, l, _q) in enumerate(us):
    key = canon(u_matrix(t, p, l).conj().T)      # key of the INVERSE gate
    for j in fwd.get(key, ())[:50]:
        if j != i:
            pairs.append((i, j))

print(f"\n=== PHASE-PROOF INVERSE PAIRING ===")
print(f"  gates with an exact inverse partner: {len(pairs)}")
offs = collections.Counter(j - i for i, j in pairs)
print(f"  distinct partner offsets: {len(offs)}")
if offs:
    print("  top offsets:", offs.most_common(10))
    # a translated block shows a single dominant offset
    tot = sum(offs.values())
    top_off, top_n = offs.most_common(1)[0]
    print(f"  dominant offset {top_off:+d} covers {100.0*top_n/tot:.1f}% of pairs")

# where in the circuit do paired gates live?  a block shows as a contiguous run
if pairs:
    firsts = sorted(i for i, _ in pairs)
    print(f"  first paired gate at {firsts[0]}, last at {firsts[-1]}  (of {len(us)})")
    hist = collections.Counter((i * 20) // len(us) for i, _ in pairs)
    print("  pairs per 5% of circuit:", [hist.get(k, 0) for k in range(20)])

# cz structure: is the interaction graph mirrored?
czset = collections.Counter(tuple(sorted(c)) for c in czs)
print(f"\n=== CZ GRAPH ===")
print(f"  distinct cz pairs: {len(czset)}  most repeated: {czset.most_common(3)}")
deg = collections.Counter()
for a, b in czset:
    deg[a] += 1
    deg[b] += 1
print(f"  degree sequence (top 8): {sorted(deg.values(), reverse=True)[:8]}")
dup = sum(1 for c, n in czset.items() if n > 1)
print(f"  cz pairs appearing more than once: {dup}")
