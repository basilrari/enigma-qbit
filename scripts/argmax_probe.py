#!/usr/bin/env python3
"""EXACT global argmax (top-M) of an MPS by branch-and-bound.

FUNDAMENTAL POINT
-----------------
Everything we have run so far -- ascend(), double_polish(), P_enum(),
sampling -- finds a LOCAL optimum of the MPS probability, or a random draw
from it.  None of them can answer the question that actually matters:

    Is the answer outside the model, or is our search failing to reach the
    model's optimum?

A local search cannot distinguish those two cases.  An exact optimiser can.

THE BOUND
---------
For an MPS, the suffix norm environments satisfy

    sum over ALL completions of a prefix  |amp(completion)|^2 = v_k R_k v_k^H

Since every term is non-negative, that total mass is an exact UPPER BOUND on
the squared amplitude of ANY single completion:

    |amp(x)|  <=  sqrt( v_k R_k v_k^H )

v_k is a 1 x l_k row vector and R_k an l_k x l_k matrix, so the bound costs
O(chi^2) per node.  It is gauge independent (v and R transform consistently),
needs no canonical form, and depends only on which *prefix* we are in.

ALGORITHM
---------
Best-first search over prefixes, ordered by this bound.  Pop the most
promising prefix; if its bound cannot beat the M-th best complete string
already found, STOP -- everything remaining in the frontier is bounded by it
too, so the result is the certified global top-M of the model.  No restarts,
no local minima, no heuristic.

Cost is governed by the model's own concentration: pruning begins around
depth ~2*log2(1/|amp_max|), so it is cheap when the model has a genuine heavy
hitter and degenerates to enumeration only for a flat model.  That behaviour
is itself the diagnostic.
"""

import heapq
import itertools
import math
import os
import sys
import time
import json

import numpy as np

VER = "/mnt/8tb_hdd2/basilrari/enigma-work/verify"
sys.path.insert(0, VER)
import hqp_solver as HS


def suffix_envs(T):
    """R[k] = sum_b T_k^b R[k+1] (T_k^b)^H,  R[n] = 1."""
    n = len(T)
    R = [None] * (n + 1)
    R[n] = np.ones((1, 1), dtype=complex)
    for k in range(n - 1, -1, -1):
        acc = None
        for b in (0, 1):
            A = T[k][:, b, :]
            t = A @ R[k + 1] @ A.conj().T
            acc = t if acc is None else acc + t
        R[k] = acc
    return R


def bits_to_str(bits, n, qo):
    s = ["0"] * n
    for k in range(n):
        if (bits >> (n - 1 - k)) & 1:
            s[qo[k]] = "1"
    return "".join(s)


def exact_top(mps, top=1, node_cap=6_000_000, verbose=True, tag=""):
    """Certified global top-`top` strings of the MPS by branch-and-bound."""
    T, n, qo = mps.T, mps.n, mps.qo
    t0 = time.time()
    R = suffix_envs(T)

    tick = itertools.count()
    # heap entries: (-bound, depth, prefix_bits, unique_tick, left_vector)
    heap = [(-1.0, 0, 0, next(tick), np.ones((1, 1), dtype=complex))]
    found = []          # min-heap of (amp, tick, bits)
    nodes = 0
    expanded = 0
    max_heap = 1
    certified = False

    while heap:
        nb, k, bits, _, v = heapq.heappop(heap)
        nodes += 1
        bound = -nb
        thresh = found[0][0] if len(found) >= top else 0.0
        if bound <= thresh:
            certified = True
            break
        if nodes > node_cap:
            certified = False
            break
        if k == n:
            a = float(abs(v[0, 0]))
            expanded += 1
            if len(found) < top:
                heapq.heappush(found, (a, next(tick), bits))
            elif a > found[0][0]:
                heapq.heapreplace(found, (a, next(tick), bits))
            continue
        Tk = T[k]
        Rk1 = R[k + 1]
        for b in (0, 1):
            w = v @ Tk[:, b, :]                      # 1 x l_{k+1}
            m = float(np.real(w @ Rk1 @ w.conj().T)[0, 0])
            bnd = math.sqrt(m) if m > 0.0 else 0.0
            if len(found) < top or bnd > found[0][0]:
                heapq.heappush(heap, (-bnd, k + 1, (bits << 1) | b,
                                      next(tick), w))
        if len(heap) > max_heap:
            max_heap = len(heap)

    res = []
    for a, _, bits in sorted(found, reverse=True):
        res.append((bits_to_str(bits, n, qo), a * a))
    if verbose:
        print(f"[{tag}] nodes={nodes} complete={expanded} maxheap={max_heap} "
              f"certified={certified} time={time.time()-t0:.1f}s", flush=True)
    return res, {"nodes": nodes, "complete": expanded, "max_heap": max_heap,
                 "certified": certified, "seconds": time.time() - t0}


def validate(ntrials=3, nq=14, depth=14, chi_full=None):
    """A* must reproduce brute-force enumeration of the same MPS exactly."""
    print(f"=== VALIDATION: {ntrials} random {nq}-qubit circuits ===",
          flush=True)
    rng = np.random.default_rng(7)
    ok = True
    for t in range(ntrials):
        lines = ["OPENQASM 2.0;", 'include "qelib1.inc";',
                 f"qreg q[{nq}];", f"creg c[{nq}];"]
        prev = list(range(nq))
        for d in range(depth):
            for q in range(nq):
                th = rng.uniform(0, 2 * math.pi)
                ph = rng.uniform(0, 2 * math.pi)
                lm = rng.uniform(0, 2 * math.pi)
                lines.append(f"u({th},{ph},{lm}) q[{q}];")
            d2 = list(range(nq))
            rng.shuffle(d2)
            for i in range(0, nq - 1, 2):
                lines.append(f"cz q[{d2[i]}],q[{d2[i+1]}];")
        path = f"{VER}/_val_{t}.qasm"
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")
        circ = HS.load_circuit(path)
        tensors, order, q_at = HS.build_mps(circ, chi_full or (2 ** (nq // 2)),
                                            verbose=False)
        mps = HS.MPS(tensors, q_at, peak=None, tag=f"val{t}")

        # brute force over the whole space using the SAME model
        allp = []
        for x in range(1 << nq):
            s = format(x, f"0{nq}b")
            allp.append((mps.P(s), s))
        allp.sort(reverse=True)
        bf_top = [(s, p) for p, s in allp[:5]]

        ast, info = exact_top(mps, top=5, verbose=False)
        match = all(abs(a[1] - b[1]) < 1e-12 for a, b in zip(ast, bf_top)) and \
            all(a[0] == b[0] for a, b in zip(ast, bf_top))
        print(f"  trial {t}: maxbond={max(x.shape[2] for x in tensors)} "
              f"A*top={ast[0][0]} P={ast[0][1]:.6e} | "
              f"BFtop={bf_top[0][0]} P={bf_top[0][1]:.6e} | "
              f"nodes={info['nodes']} certified={info['certified']} "
              f"-> {'MATCH' if match else 'MISMATCH'}", flush=True)
        if not match:
            ok = False
            print(f"    A* : {ast}", flush=True)
            print(f"    BF : {bf_top}", flush=True)
        os.remove(path)
    print(f"=== VALIDATION {'PASSED' if ok else 'FAILED'} ===", flush=True)
    return ok


def main():
    if "--validate" in sys.argv:
        validate()
        return
    cid = sys.argv[1]
    chi = int(sys.argv[2])
    top = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    peaks = json.load(open(f"{VER}/_peaks.json"))
    known = peaks.get(cid)
    path = f"{VER}/{cid}.qasm"
    circ = HS.load_circuit(path)
    n = circ.num_qubits
    print(f"=== EXACT ARGMAX  {cid}  chi={chi}  qubits={n} ===", flush=True)
    tb = time.time()
    tensors, order, q_at = HS.build_mps(circ, chi, verbose=False)
    print(f"build {time.time()-tb:.0f}s  maxbond="
          f"{max(t.shape[2] for t in tensors)}", flush=True)
    mps = HS.MPS(tensors, q_at, peak=known, tag=cid)
    if known:
        print(f"MPS P(known) = {mps.P(known):.6e}  "
              f"(uniform {2.0**-n:.3e}, ratio {mps.P(known)*2.0**n:.3e})",
              flush=True)
    res, info = exact_top(mps, top=top, verbose=True, tag=cid)
    print(f"\n--- certified top-{top} of the model (chi={chi}) ---", flush=True)
    for i, (s, p) in enumerate(res):
        h = sum(a != b for a, b in zip(s, known)) if known else "?"
        print(f"  #{i+1}  H={h}  P={p:.6e}  {s}", flush=True)
    out = {"cid": cid, "chi": chi, "qubits": n, "info": info,
           "top": [{"str": s, "P": p,
                    "H": (sum(a != b for a, b in zip(s, known))
                          if known else None)} for s, p in res],
           "P_known": (mps.P(known) if known else None)}
    with open(f"{VER}/argmax_{cid}_c{chi}.json", "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {VER}/argmax_{cid}_c{chi}.json", flush=True)


if __name__ == "__main__":
    main()
