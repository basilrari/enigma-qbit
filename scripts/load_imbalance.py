#!/usr/bin/env python3
"""Is the answer encoded in HOW MUCH each qubit is touched?

Fundamental reasoning: a uniformly-scrambled circuit gives a Porter-Thomas
output whose maximum is only ~ln(2^n) times uniform -- for d1 that is ~32x.
Our measured d1 peak is 1.8e11 x uniform.  A peak that strong CANNOT come from
scrambling; it must be planted.  The cheapest way to plant one is LOAD
IMBALANCE: leave a subset of qubits nearly untouched, so they stay close to
|0> and their bits are almost deterministic, while the rest scramble.  Then the
answer is mostly the "idle" qubits' values.

So: for every structural feature we can compute per qubit (how many gates touch
it, when the first and last gate hit it, parity of the gate count, mean angle),
test whether it predicts that qubit's answer bit.  If any feature separates the
bits, the answer is readable for free and d2/d3 fall out in seconds.
"""

import glob
import itertools
import json
import math
import os
import re
import sys

SAMPLES = "/home/basilsclaw/enigma-solve/samples"
sys.path.insert(0, "/home/basilsclaw/enigma-solve/scripts")
from peak_read import parse  # noqa: E402


def features(cid):
    known = json.load(open(f"{SAMPLES}/{cid}_meta.json"))["peaked_state"]
    n, gates = parse(f"{SAMPLES}/{cid}.qasm")
    if n == 0:
        return None
    n_u = [0] * n
    n_cz = [0] * n
    first = [10 ** 9] * n
    last = [-1] * n
    sumth = [0.0] * n
    cntth = [0] * n
    firstth = [None] * n
    lastth = [None] * n
    czdeg = [[] for _ in range(n)]
    for gi, (kind, qs, par) in enumerate(gates):
        for q in qs:
            if kind == "u":
                n_u[q] += 1
                sumth[q] += par[0]
                cntth[q] += 1
                if firstth[q] is None:
                    firstth[q] = par[0]
                lastth[q] = par[0]
            else:
                n_cz[q] += 1
            if gi < first[q]:
                first[q] = gi
            if gi > last[q]:
                last[q] = gi
    return {"known": known, "n": n, "gates": len(gates), "n_u": n_u,
            "n_cz": n_cz, "first": first, "last": last, "sumth": sumth,
            "cntth": cntth, "firstth": firstth, "lastth": lastth,
            "tot": [n_u[q] + n_cz[q] for q in range(n)]}


def best_threshold(vals, bits):
    """best single-threshold accuracy over the observed values."""
    uniq = sorted(set(vals))
    best = (0.5, None, None)
    for t in uniq:
        for op in ("<", ">="):
            pred = [1 if ((v < t) if op == "<" else (v >= t)) else 0
                    for v in vals]
            acc = sum(p == b for p, b in zip(pred, bits)) / len(bits)
            if acc > best[0]:
                best = (acc, t, op)
    return best


def analyse(cid):
    F = features(cid)
    if F is None:
        return
    n, known = F["n"], F["known"]
    bits = [int(known[q]) for q in range(min(n, len(known)))]
    print(f"\n{'='*74}\n{cid}  n={n}  gates={F['gates']}  ones={sum(bits)}/{len(bits)}")

    def show(name, vals):
        vals = vals[:len(bits)]
        v0 = [v for v, b in zip(vals, bits) if b == 0]
        v1 = [v for v, b in zip(vals, bits) if b == 1]
        acc, t, op = best_threshold(vals, bits)
        m0 = sum(v0) / len(v0) if v0 else float("nan")
        m1 = sum(v1) / len(v1) if v1 else float("nan")
        print(f"  {name:12s} mean|bit=0 {m0:9.2f}   mean|bit=1 {m1:9.2f}   "
              f"best rule acc {100*acc:5.1f}%  (v{op}{t:g} if t is not None else "-")   "
              f"range {min(vals):g}..{max(vals):g}")
        return acc

    print("  -- single-feature tests (baseline accuracy = "
          f"{100*max(sum(bits), len(bits)-sum(bits))/len(bits):.1f}% from "
          "guessing the majority bit) --")
    show("n_u", F["n_u"])
    show("n_cz", F["n_cz"])
    show("total gates", F["tot"])
    show("first gate", F["first"])
    show("last gate", F["last"])
    show("mean theta", [F["sumth"][q] / max(1, F["cntth"][q]) for q in range(n)])
    show("n_u parity", [F["n_u"][q] % 2 for q in range(n)])
    show("n_cz parity", [F["n_cz"][q] % 2 for q in range(n)])
    show("total parity", [F["tot"][q] % 2 for q in range(n)])
    fm = F["first"]
    show("first-half?", [1 if fm[q] < F["gates"] / 2 else 0 for q in range(n)])
    show("last-10%?", [1 if F["last"][q] > 0.9 * F["gates"] else 0
                       for q in range(n)])

    # load imbalance: is the toucher-count distribution actually uneven?
    nu = sorted(F["n_u"])
    tot = sorted(F["tot"])
    print(f"\n  n_u per qubit: min={nu[0]} q1={nu[len(nu)//4]} med={nu[len(nu)//2]}"
          f" q3={nu[3*len(nu)//4]} max={nu[-1]}   "
          f"spread={nu[-1]/max(1,nu[0]):.2f}x")
    print(f"  total per qubit: min={tot[0]} med={tot[len(tot)//2]} max={tot[-1]}"
          f"   spread={tot[-1]/max(1,tot[0]):.2f}x")
    idle = [q for q in range(n) if F["tot"][q] < 0.5 * tot[len(tot) // 2]]
    print(f"  strongly under-touched qubits (<half median): {idle}")
    if idle:
        print(f"    their answer bits: {[bits[q] for q in idle if q < len(bits)]}"
              f"  (vs global mean {sum(bits)/len(bits):.2f})")


for f in sorted(glob.glob(f"{SAMPLES}/*.qasm")):
    analyse(os.path.basename(f)[:-5])
