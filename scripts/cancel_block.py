#!/usr/bin/env python3
"""Undo the disguise and check whether the two blocks cancel EXACTLY.

No simulation needed: after relabelling the disguised block's wires by pi, if
its gate multiset equals the inverse block's gate multiset (u gates by value,
cz gates by wire pair), then the two blocks multiply to the identity and can be
deleted outright.  That turns a ~10 hour absorption into a text edit.
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
    """Return the circuit as an ordered list of (kind, payload, qubits)."""
    out = []
    for idx, l in enumerate(x.strip() for x in open(path).read().splitlines()):
        m = UG.match(l)
        if m:
            q = [_num(v) for v in m.group(1).split(",")]
            out.append(("u", (q[0], q[1], q[2]), (int(m.group(2)),), idx))
            continue
        m = CZ.match(l)
        if m:
            out.append(("cz", None, (int(m.group(1)), int(m.group(2))), idx))
    return out


def analyse(path):
    circ = parse(path)
    ug = [c for c in circ if c[0] == "u"]
    cz = [c for c in circ if c[0] == "cz"]
    nq = max(max(c[2]) for c in circ) + 1
    print(f"=== {path.split('/')[-1]} ===")
    print(f"  total gates {len(circ)}  (u {len(ug)}, cz {len(cz)})  qubits {nq}")

    # index u gates globally, and build value -> positions
    value = {k: canon(u_matrix(*ug[k][1])) for k in range(len(ug))}
    fwd = collections.defaultdict(list)
    for k in range(len(ug)):
        fwd[value[k]].append(k)

    pairs = []
    for k in range(len(ug)):
        t, p, l = ug[k][1]
        invkey = canon(u_matrix(t, p, l).conj().T)
        for m in fwd.get(invkey, ())[:60]:
            if m != k:
                pairs.append((k, m))

    uniq = sorted({(min(k, m), max(k, m)) for k, m in pairs})
    A = sorted({k for k, _ in uniq})
    B = sorted({m for _, m in uniq})
    print(f"  inverse pairs {len(uniq)} (deduped);"
          f"  earlier members {len(A)} (circuit {ug[A[0]][3]}..{ug[A[-1]][3]}),"
          f"  later members {len(B)} (circuit {ug[B[0]][3]}..{ug[B[-1]][3]})")

    # votes -> pi
    V = np.zeros((nq, nq), dtype=float)
    for k, m in pairs:
        a, b = ug[k][2][0], ug[m][2][0]
        if a != b:
            V[b, a] += 1        # pi maps the disguised label a -> true label b
            V[a, b] += 1
    if linear_sum_assignment is not None and V.sum() > 0:
        r, c = linear_sum_assignment(-V)
        pi = {int(a): int(b) for a, b in zip(r, c)}
        got = V[r, c].sum() / V.sum()
        print(f"  pi recovered with {100*got:.1f}% of votes")

    # multiset test: relabel A's wires, invert A's gates, compare to B.
    # (block A disguises U; block B is U*.  A gate g in U pairs with g* in U*.)
    def key_for(k, qmap):
        t, p, l = ug[k][1]
        q = qmap.get(ug[k][2][0], ug[k][2][0])
        return (canon(u_matrix(t, p, l).conj().T), q)

    for label, qmap in (("pi", pi), ("pi^-1", {v: k for k, v in pi.items()})):
        Ablock = collections.Counter(key_for(k, qmap) for k in A)
        Bblock = collections.Counter(ug[m][2][0] and (canon(u_matrix(*ug[m][1])), ug[m][2][0])
                                     for m in B)
        inter = sum(min(Ablock[x], Bblock[x]) for x in Ablock)
        cov = 100.0 * inter / max(1, min(sum(Ablock.values()), sum(Bblock.values())))
        print(f"\n  CANCELLATION CHECK under {label}: matched {inter}"
              f" / {min(sum(Ablock.values()), sum(Bblock.values()))} = {cov:.1f}%")

    # cz structure: do the two blocks' cz pairs match after relabelling?
    def czs_in(lo, hi):
        return [c for c in cz if lo <= c[3] <= hi]
    spanA = (ug[A[0]][3], ug[A[-1]][3])
    spanB = (ug[B[0]][3], ug[B[-1]][3])
    czA = czs_in(*spanA)
    czB = czs_in(*spanB)
    setA = collections.Counter(tuple(sorted((pi.get(a, a), pi.get(b, b)))) for _k, _p, (a, b), _i in czA)
    setB = collections.Counter(tuple(sorted((a, b))) for _k, _p, (a, b), _i in czB)
    czmatch = sum(min(setA[x], setB[x]) for x in setA)
    print(f"    cz gates: first span {len(czA)} (relabelled) vs second {len(czB)}")
    print(f"    cz matched: {czmatch}"
          f"  = {100.0*czmatch/max(1,min(len(czA), len(czB))):.1f}% of the smaller span")
    print()


if __name__ == "__main__":
    for a in (sys.argv[1:] or [
        "/home/basilsclaw/enigma-solve/samples/d2_s1_39b370e4.qasm",
        "/home/basilsclaw/enigma-solve/samples/d1_s1_4043cafb.qasm",
    ]):
        analyse(a)
