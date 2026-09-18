#!/usr/bin/env python3
"""Exact algebraic reduction of a circuit, and a search for the wire relabelling
that maximises it.

Fundamentals.  A circuit is a word in gates, one letter per wire.  Two gates on
the same wire cancel exactly when nothing else touches that wire between them --
gates on *other* wires commute past and are irrelevant.  A tensor network is
only needed for what survives; anything the reduction removes is gone for good,
with no truncation and no approximation.

If the challenge's "disguise" is a relabelling of the wires (as the inverse-pair
structure suggests), then in the *right* labelling the disguise and its inverse
become adjacent and annihilate.  So rather than guessing a permutation from
sparse vote counts, search directly for the labelling that cancels the most --
scoring candidates by the only thing that matters, how much circuit actually
disappears.

The reduction is exact, so any residual is still the same unitary: a much
smaller circuit that can be solved by the existing exact path.
"""
import copy
import math
import re
import sys

import numpy as np

sys.path.insert(0, "/home/basilsclaw/enigma-solve/scripts")
from recover_perm import _num, canon, u_matrix  # noqa: E402

UG = re.compile(r"^u\(([^)]*)\)\s+q\[(\d+)\];")
CZ = re.compile(r"^cz\s+q\[(\d+)\],q\[(\d+)\];")


class One:
    """A single-qubit gate, remembering its inverse's canonical form."""
    __slots__ = ("q", "key", "inv")

    def __init__(self, q, key, inv):
        self.q, self.key, self.inv = q, key, inv


class Two:
    """A controlled-Z.  Self-inverse; cancels only as a matched pair."""
    __slots__ = ("a", "b")

    def __init__(self, a, b):
        self.a, self.b = a, b


def parse_qasm(path):
    out = []
    unmatched = []
    for line in (x.strip() for x in open(path).read().splitlines()):
        if not line or line.startswith(("//", "#", "OPENQASM", "include",
                                        "qreg", "creg", "barrier", "measure")):
            continue
        m = UG.match(line)
        if m:
            q = [_num(v) for v in m.group(1).split(",")]
            out.append(("u", int(m.group(2)), q[0], q[1], q[2]))
            continue
        m = CZ.match(line)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            out.append(("cz", a, b))
            continue
        unmatched.append(line)
    return out, unmatched


def reduce_word(word, perm=None, trace=False):
    """Exact per-wire stack reduction.  Returns (residual, n_cancelled).

    perm, if given, maps *logical* wire -> *positional* wire before reducing:
    i.e. gate on wire w is read as acting on wire perm[w].
    """
    stacks = {}
    cancelled = 0
    residual = []
    for g in word:
        if g[0] == "u":
            _, q, t, p, l = g
            q = perm[q] if perm else q
            M = u_matrix(t, p, l)
            st = stacks.setdefault(q, [])
            if st and isinstance(st[-1], One) and st[-1].inv == canon(M.conj().T):
                st.pop()
                cancelled += 1
                if trace:
                    residual.append(("cancel_u", q))
                continue
            e = One(q, canon(M), canon(M.conj().T))
            st.append(e)
            residual.append(("u", q, t, p, l))
        else:
            _, a, b = g
            a = perm[a] if perm else a
            b = perm[b] if perm else b
            sa, sb = stacks.setdefault(a, []), stacks.setdefault(b, [])
            if sa and sb and sa[-1] is sb[-1] and isinstance(sa[-1], Two):
                two = sa[-1]
                sa.pop()
                sb.pop()
                cancelled += 1
                if trace:
                    residual.append(("cancel_cz", two.a, two.b))
                continue
            e = Two(a, b)
            sa.append(e)
            sb.append(e)
            residual.append(("cz", a, b))
    return residual, cancelled


def residual_word(residual, perm=None):
    """Turn a traced residual back into a clean gate list on original wires."""
    inv = None
    if perm:
        inv = [0] * len(perm)
        for k, v in enumerate(perm):
            inv[v] = k
    out = []
    for item in residual:
        if item[0] == "u":
            _, q, t, p, l = item
            out.append(("u", inv[q] if inv else q, t, p, l))
        elif item[0] == "cz":
            _, a, b = item
            out.append(("cz", inv[a] if inv else a, inv[b] if inv else b))
    return out


def vote_perm(word):
    """The inverse-pair vote permutation (sparse heuristic), as a starting point."""
    nq = max((g[4] if g[0] == "u" else max(g[1], g[2])) for g in word) + 1
    us = [g for g in word if g[0] == "u"]
    fwd = {}
    for i, (_, q, t, p, l) in enumerate(us):
        fwd.setdefault(canon(u_matrix(t, p, l)), []).append(i)
    V = np.zeros((nq, nq), dtype=float)
    for i, (_, q, t, p, l) in enumerate(us):
        for j in fwd.get(canon(u_matrix(t, p, l).conj().T), ())[:60]:
            if j != i:
                b = us[j][1]
                if b != q:
                    V[q, b] += 1
    Vc = np.maximum(V, V.T)
    try:
        from scipy.optimize import linear_sum_assignment
        r, c = linear_sum_assignment(-Vc)
        return [int(x) for x in c], nq
    except ImportError:
        return list(range(nq)), nq


def hill_climb(word, perm, nq, steps=6, max_swaps=400, log=print):
    """Greedy search over transpositions in the labelling, scored by exact
    cancellation.  Cheap: the reduction is O(gates)."""
    best = list(perm)
    _, best_score = reduce_word(word, best)
    for s in range(steps):
        improved = False
        swaps = [(i, j) for i in range(nq) for j in range(i + 1, nq)]
        if len(swaps) > max_swaps:
            stride = len(swaps) // max_swaps + 1
            swaps = swaps[::stride]
        for (i, j) in swaps:
            cand = list(best)
            cand[i], cand[j] = cand[j], cand[i]
            _, sc = reduce_word(word, cand)
            if sc > best_score:
                best, best_score, improved = cand, sc, True
        log(f"    climb step {s+1}: cancelled={best_score}")
        if not improved:
            break
    return best, best_score


def main():
    paths = sys.argv[1:] or [
        "/home/basilsclaw/enigma-solve/samples/d1_s1_4043cafb.qasm",
        "/home/basilsclaw/enigma-solve/samples/d2_s1_39b370e4.qasm",
    ]
    for path in paths:
        word, unmatched = parse_qasm(path)
        name = path.split("/")[-1]
        if not word:
            print(f"\n=== {name}: no u/cz gates (different gate set; skipped) ===")
            continue
        nq = max((g[4] if g[0] == "u" else max(g[1], g[2])) for g in word) + 1
        n_u = sum(1 for g in word if g[0] == "u")
        n_cz = len(word) - n_u
        print(f"\n=== {name}: {nq} qubits, {len(word)} gates "
              f"({n_u} u, {n_cz} cz) ===")
        if unmatched:
            print(f"  !! {len(unmatched)} UNPARSED lines - reduction would be "
                  f"UNSOUND for this file. First: {unmatched[:2]}")
        else:
            print("  coverage: every instruction parsed (reduction is sound)")

        res, canc = reduce_word(word)
        print(f"  identity labelling      : cancelled {canc:5d}  "
              f"residual {len(res):5d}")

        vp, _ = vote_perm(word)
        res_v, canc_v = reduce_word(word, vp)
        print(f"  vote permutation        : cancelled {canc_v:5d}  "
              f"residual {len(res_v):5d}   ({canc_v/len(word):.1%} gone)")

        if len(sys.argv) > 3:      # opt-in: the expensive search
            print("  hill-climbing the labelling...")
            bp, bs = hill_climb(word, vp, nq)
            res_b, _ = reduce_word(word, bp)
            print(f"  climbed permutation     : cancelled {bs:5d}  "
                  f"residual {len(res_b):5d}   ({bs/len(word):.1%} gone)")
            print(f"  perm = {bp}")


if __name__ == "__main__":
    main()
