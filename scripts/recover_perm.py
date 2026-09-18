#!/usr/bin/env python3
"""Recover the disguise permutation from the inverse pairing -- combinatorially.

If the circuit contains a block U and a disguise T[U] (U with wires relabelled
by pi) plus the true inverse U*, then gates pair with their inverses and each
pair (i, j) with qubits (a, b) votes for pi: a -> b.  A consistent permutation
that wins a large majority of the votes IS the disguise, and undoing it makes
the block cancel exactly -- no numeric absorption required.
"""
import collections
import math
import re
import sys

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None


def _num(tok):
    tok = tok.strip()
    try:
        return float(tok)
    except ValueError:
        return float(eval(tok.replace("pi", repr(math.pi)), {"__builtins__": {}}, {}))


def u_matrix(t, p, l):
    c, s = math.cos(t / 2.0), math.sin(t / 2.0)
    return np.array([[c, -np.exp(1j * l) * s],
                     [np.exp(1j * p) * s, np.exp(1j * (p + l)) * c]], dtype=complex)


def canon(M):
    A = np.asarray(M, dtype=complex)
    k = np.unravel_index(np.argmax(np.abs(A)), A.shape)
    if abs(A[k]) < 1e-12:
        return ("zero",)
    B = A / A[k]
    return tuple(round(v, 6) for v in np.concatenate([B.real.ravel(), B.imag.ravel()]))


UG = re.compile(r"^u\(([^)]*)\)\s+q\[(\d+)\];")
CZ = re.compile(r"^cz\s+q\[(\d+)\],q\[(\d+)\];")


def parse(path):
    us = []
    for l in (x.strip() for x in open(path).read().splitlines()):
        m = UG.match(l)
        if m:
            q = [_num(v) for v in m.group(1).split(",")]
            us.append((q[0], q[1], q[2], int(m.group(2))))
    return us


def analyse(path):
    us = parse(path)
    nq = max(q for *_x, q in us) + 1
    fwd = collections.defaultdict(list)
    for i, (t, p, l, _q) in enumerate(us):
        fwd[canon(u_matrix(t, p, l))].append(i)

    pairs = []
    for i, (t, p, l, q) in enumerate(us):
        for j in fwd.get(canon(u_matrix(t, p, l).conj().T), ())[:60]:
            if j != i:
                pairs.append((i, j))

    V = np.zeros((nq, nq), dtype=float)
    cross = 0
    for i, j in pairs:
        a, b = us[i][3], us[j][3]
        if a != b:
            cross += 1
            V[a, b] += 1

    print(f"=== {path.split('/')[-1]} ===")
    print(f"  u gates {len(us)}   qubits {nq}   inverse pairs {len(pairs)}"
          f"   of which CROSS-qubit: {cross}")

    Vc = np.maximum(V, V.T)          # undirected evidence
    if linear_sum_assignment is not None and Vc.sum() > 0:
        r, c = linear_sum_assignment(-Vc)
        tot = Vc[r, c].sum()
        agree = tot / max(1.0, Vc.sum())
        print(f"  best permutation scores {tot:.0f} of {Vc.sum():.0f} votes"
              f"  = {100*agree:.1f}% agreement")
        print(f"  pi = {list(zip(r.tolist(), c.tolist()))[:nq]}")
        fixed = sum(1 for a, b in zip(r, c) if a != b)
        print(f"  non-identity mappings: {fixed} of {nq}")
    # clarity: per source qubit, is there a dominant target?
    clear = 0
    for a in range(nq):
        row = V[a] + V[:, a]
        if row.max() > 0 and row.max() >= 0.6 * row.sum():
            clear += 1
    print(f"  source qubits with a >=60% dominant partner: {clear} / {nq}")
    print()


if __name__ == "__main__":
    for a in (sys.argv[1:] or [
        "/home/basilsclaw/enigma-solve/samples/d1_s1_4043cafb.qasm",
        "/home/basilsclaw/enigma-solve/samples/d2_s1_39b370e4.qasm",
        "/home/basilsclaw/enigma-solve/samples/d3_s1_2674779a.qasm",
    ]):
        analyse(a)
