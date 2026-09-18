#!/usr/bin/env python3
"""Discriminate a real inverse block from a reused angle pool.

If the generator sampled degrees from a small fixed pool, two gates match by
coincidence often -- and then inverse-pairs and identical-pairs appear at the
same rate.  If instead a genuine U/U* block exists, inverse pairs dominate.

Also reports whether the INVERSE pairs are near-mirrored (j ~= N-1-i), which is
what a block reversed by conjugation looks like.
"""
import collections
import math
import re
import sys

import numpy as np

sys.path.insert(0, "/home/basilsclaw/enigma-solve/scripts")


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


def parse(path):
    us = []
    for l in (x.strip() for x in open(path).read().splitlines()):
        m = UG.match(l)
        if m:
            q = [_num(v) for v in m.group(1).split(",")]
            us.append((q[0], q[1], q[2], int(m.group(2))))
    return us


def report(path):
    us = parse(path)
    fwd = collections.defaultdict(list)
    for i, (t, p, l, _q) in enumerate(us):
        fwd[canon(u_matrix(t, p, l))].append(i)

    same, inv = [], []
    for i, (t, p, l, _q) in enumerate(us):
        kf = canon(u_matrix(t, p, l))
        ki = canon(u_matrix(t, p, l).conj().T)
        for j in fwd.get(kf, ())[:60]:
            if j > i:
                same.append((i, j))
        for j in fwd.get(ki, ())[:60]:
            if j != i:
                inv.append((i, j))

    print(f"=== {path.split('/')[-1]}  ({len(us)} u gates) ===")
    print(f"  IDENTICAL gate pairs : {len(same)}")
    print(f"  INVERSE   gate pairs : {len(inv)}")
    if inv:
        mir = sum(1 for i, j in inv if abs((i + j) - (len(us) - 1)) <= 2)
        print(f"  inverse pairs that are near-mirrored (j ~ N-1-i): {mir}"
              f"  = {100.0 * mir / len(inv):.1f}%")
        offs = collections.Counter(j - i for i, j in inv)
        print(f"  top inverse offsets: {offs.most_common(6)}")
        hist = collections.Counter((min(i, j) * 20) // len(us) for i, j in inv)
        print("  inverse pairs per 5% of circuit:",
              [hist.get(k, 0) for k in range(20)])
    print()


if __name__ == "__main__":
    args = sys.argv[1:] or [
        "/home/basilsclaw/enigma-solve/samples/d1_s1_4043cafb.qasm",
        "/home/basilsclaw/enigma-solve/samples/d2_s1_39b370e4.qasm",
        "/home/basilsclaw/enigma-solve/samples/d3_s1_2674779a.qasm",
    ]
    for a in args:
        report(a)
