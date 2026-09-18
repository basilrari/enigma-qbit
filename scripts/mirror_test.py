#!/usr/bin/env python3
"""Is the matched gate pairing a MIRROR?  That decides provable cancellation.

Matching multisets is not enough: products cancel only if the second block's
gates appear in the reverse order of the first.  For each exactly matched pair
we record (position in block A, position in block B) and ask whether the
sequence of B-positions is monotonically reversing as A advances.  A clean
anti-diagonal means the blocks are U and U* verbatim and can be deleted.
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


def analyse(path):
    us = []
    for l in (x.strip() for x in open(path).read().splitlines()):
        m = UG.match(l)
        if m:
            q = [_num(v) for v in m.group(1).split(",")]
            us.append((q[0], q[1], q[2], int(m.group(2))))
    nq = max(q for *_x, q in us) + 1
    fwd = collections.defaultdict(list)
    for k, (t, p, l, _q) in enumerate(us):
        fwd[canon(u_matrix(t, p, l))].append(k)

    pairs = []
    for k, (t, p, l, _q) in enumerate(us):
        ik = canon(u_matrix(t, p, l).conj().T)
        for m in fwd.get(ik, ())[:60]:
            if m != k:
                pairs.append((k, m))
    uniq = sorted({(min(k, m), max(k, m)) for k, m in pairs})

    V = np.zeros((nq, nq))
    for k, m in uniq:
        a, b = us[k][3], us[m][3]
        V[b, a] += 1
        V[a, b] += 1
    pi = {}
    if linear_sum_assignment is not None:
        r, c = linear_sum_assignment(-V)
        pi = {int(a): int(b) for a, b in zip(r, c)}

    # under pi, pair each A gate with the B gate that is its exact inverse
    Bkey = collections.defaultdict(list)
    for m in {x for _, x in uniq}:
        Bkey[canon(u_matrix(*us[m][:3]))].append(m)
    match = []
    for k, _m in uniq:
        t, p, l, q = us[k]
        inv_here = canon(u_matrix(t, p, l).conj().T)
        for cand in Bkey.get(inv_here, ()):
            if us[cand][3] == pi.get(q, q):
                match.append((k, cand))
                break

    print(f"=== {path.split('/')[-1]} ===")
    print(f"  u gates {len(us)}  qubits {nq}  deduped inverse pairs {len(uniq)}")
    print(f"  pairs matching under pi AND inverse-exact: {len(match)}"
          f"  = {100.0*len(match)/max(1,len(uniq)):.1f}%")

    if match:
        match.sort()
        bs = [b for _a, b in match]
        inc = sum(1 for i in range(1, len(bs)) if bs[i] > bs[i - 1])
        dec = sum(1 for i in range(1, len(bs)) if bs[i] < bs[i - 1])
        print(f"  ordering: {dec} decreasing steps vs {inc} increasing"
              f"  -> {'MIRROR (products cancel)' if dec > 0.9*(dec+inc) else 'NOT a mirror'}")
        # how contiguous is each side?
        as_ = sorted(a for a, _ in match)
        gaps_a = [as_[i + 1] - as_[i] for i in range(len(as_) - 1)]
        bs_sorted = sorted(bs)
        gaps_b = [bs_sorted[i + 1] - bs_sorted[i] for i in range(len(bs_sorted) - 1)]
        print(f"  block A span {as_[0]}..{as_[-1]} (max gap {max(gaps_a) if gaps_a else 0}),"
              f" block B span {bs_sorted[0]}..{bs_sorted[-1]}"
              f" (max gap {max(gaps_b) if gaps_b else 0})")
    print()


if __name__ == "__main__":
    for a in (sys.argv[1:] or [
        "/home/basilsclaw/enigma-solve/samples/d2_s1_39b370e4.qasm",
        "/home/basilsclaw/enigma-solve/samples/d1_s1_4043cafb.qasm",
        "/home/basilsclaw/enigma-solve/samples/d3_s1_2674779a.qasm",
    ]):
        analyse(a)
