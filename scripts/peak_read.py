#!/usr/bin/env python3
"""Can the answer be READ OUT of the circuit's gate structure directly?

Rationale (fundamentals, not search): a circuit of random gates produces a
Porter-Thomas-ish distribution whose maximum is only ~O(n) times uniform.  Our
samples have peaks 10^11 x uniform (d1) -- that cannot arise from scrambling
alone, it has to be PLANTED.  If the generator plants it, the plant is a
structural feature of the QASM, and structural features are readable.

Prior forensics found near-special gates (theta near 0 / pi/2 / pi, with phi and
lambda pinned to exact +/-pi or pi/2) CLUSTERED IN A NARROW WINDOW IN THE MIDDLE
of each circuit -- something floating-point noise would never do.  That thread
was closed as an "artifact".  This script reopens it and asks the decisive
question: does the plant ENCODE the answer bits, per qubit?

If it does, d2 and d3 are readable in seconds with no simulation at all.
"""

import glob
import json
import math
import os
import re
import sys

SAMPLES = "/home/basilsclaw/enigma-solve/samples"
SPECIAL = {0.0: "0", math.pi / 2: "pi/2", math.pi: "pi",
           3 * math.pi / 2: "3pi/2"}


def val(tok):
    tok = tok.strip()
    if tok in ("pi", "-pi"):
        return math.pi if tok == "pi" else -math.pi
    if "/" in tok:
        num, den = tok.split("/")
        return val(num) / val(den)
    try:
        return float(tok)
    except ValueError:
        return float("nan")


def parse(path):
    """-> (n, gates) with gates = [(kind, qubits_tuple, (th,ph,lm) or None)]."""
    n = 0
    gates = []
    rx_u = re.compile(r"^u\(([^)]*)\)\s+q\[(\d+)\]\s*;")
    rx_cz = re.compile(r"^cz\s+q\[(\d+)\]\s*,\s*q\[(\d+)\]\s*;")
    for line in open(path):
        line = line.strip()
        m = re.match(r"qreg\s+\w+\[(\d+)\]", line)
        if m:
            n = max(n, int(m.group(1)))
        m = rx_u.match(line)
        if m:
            p = [val(x) for x in m.group(1).split(",")]
            gates.append(("u", (int(m.group(2)),), tuple(p)))
            continue
        m = rx_cz.match(line)
        if m:
            gates.append(("cz", (int(m.group(1)), int(m.group(2))), None))
    return n, gates


def wrap(a):
    """fold to (-pi, pi]"""
    while a > math.pi:
        a -= 2 * math.pi
    while a <= -math.pi:
        a += 2 * math.pi
    return a


def nearest(t, cands=(0.0, math.pi / 2, math.pi, -math.pi / 2)):
    return min(cands, key=lambda c: abs(wrap(t - c))), \
        min(abs(wrap(t - c)) for c in cands)


def analyse(cid):
    qasm = f"{SAMPLES}/{cid}.qasm"
    meta = f"{SAMPLES}/{cid}_meta.json"
    if not os.path.exists(qasm):
        return
    known = json.load(open(meta))["peaked_state"]
    n, gates = parse(qasm)
    print(f"\n{'='*72}\n{cid}   n={n}  gates={len(gates)}  "
          f"difficulty={json.load(open(meta))['difficulty']}")
    print(f"answer = {known}")

    # per-qubit gate lists, in time order
    per = {q: [] for q in range(n)}
    for gi, (kind, qs, par) in enumerate(gates):
        for q in qs:
            per[q].append((gi, kind, par))

    # ---- detector 1: is there a layer near the END of the circuit where
    #      every qubit's gate is near-special in theta?
    print("\n-- fraction of qubits whose k-th-from-last gate has theta near "
          "{0, pi/2, pi} (tol 0.02) --")
    for k in range(1, 5):
        frac, kinds = [], []
        thetas = []
        for q in range(n):
            us = [g for g in per[q] if g[1] == "u"]
            if len(us) < k:
                continue
            gi, _, par = us[-k]
            c, d = nearest(par[0])
            frac.append(d < 0.02)
            kinds.append(SPECIAL.get(round(c, 6), "?"))
            thetas.append(par[0])
        if not frac:
            continue
        print(f"  k={k}: {sum(frac)}/{len(frac)} near-special   "
              f"kinds={ {x: kinds.count(x) for x in set(kinds)} }")

    # ---- detector 2: does the k-th-from-last gate's theta PREDICT the bit?
    print("\n-- per-qubit bit prediction from the k-th-from-last u-gate --")
    for k in (1, 2, 3):
        for mode in ("th_pi", "ph_pi", "lm_pi"):
            agree = 0
            tot = 0
            detail = []
            for q in range(n):
                us = [g for g in per[q] if g[1] == "u"]
                if len(us) < k:
                    continue
                _, _, par = us[-k]
                th, ph, lm = par
                if mode == "th_pi":
                    pred = 1 if abs(wrap(th - math.pi)) < abs(wrap(th)) else 0
                elif mode == "ph_pi":
                    pred = 1 if abs(wrap(abs(ph) - math.pi)) < 0.35 else 0
                else:
                    pred = 1 if abs(wrap(abs(lm) - math.pi)) < 0.35 else 0
                bit = int(known[q]) if q < len(known) else -1
                if bit < 0:
                    continue
                tot += 1
                agree += (pred == bit)
                detail.append((q, bit, pred))
            if tot:
                print(f"  k={k} {mode:7s}: agreement {agree}/{tot} "
                      f"({100.0*agree/tot:.1f}%)")

    # ---- detector 3: where in the circuit are the near-special gates?
    print("\n-- temporal distribution of near-special u-gates "
          "(theta AND phi/lambda) --")
    tot_g = len(gates)
    marks = []
    for gi, (kind, qs, par) in enumerate(gates):
        if kind != "u":
            continue
        th, ph, lm = par
        _, dth = nearest(th)
        dph = min(abs(wrap(abs(ph) - v)) for v in (0.0, math.pi / 2, math.pi))
        dlm = min(abs(wrap(abs(lm) - v)) for v in (0.0, math.pi / 2, math.pi))
        if dth < 0.02 or dph < 0.02 or dlm < 0.02:
            marks.append((gi, gi / tot_g, dth, dph, dlm, qs[0]))
    print(f"  near-special u-gates: {len(marks)} / {len(gates)} "
          f"({100.0*len(marks)/max(1,len(gates)):.2f}%)")
    if marks:
        # histogram over 10 temporal deciles
        hist = [0] * 10
        for _, frac, *_ in marks:
            hist[min(9, int(frac * 10))] += 1
        print(f"  decile histogram: {hist}")
        print(f"  first at gate {marks[0][0]} ({100*marks[0][1]:.1f}%), "
              f"last at gate {marks[-1][0]} ({100*marks[-1][1]:.1f}%)")
        print("  sample (gate, pos%, d_theta, d_phi, d_lam, qubit):")
        for m in marks[:12]:
            print(f"    {m[0]:5d} {100*m[1]:5.1f}%  {m[2]:.4f} {m[3]:.4f} "
                  f"{m[4]:.4f}  q{m[5]}")

    # ---- detector 4: do the flagged qubits' positions match answer 1-bits?
    if marks:
        fq = sorted({m[5] for m in marks})
        ones = [q for q in range(n) if q < len(known) and known[q] == "1"]
        print(f"\n  flagged qubits ({len(fq)}): {fq}")
        print(f"  answer 1-bits ({len(ones)}): {ones}")
        print(f"  overlap: {len(set(fq) & set(ones))}")


if __name__ == "__main__":
    for f in sorted(glob.glob(f"{SAMPLES}/*.qasm")):
        analyse(os.path.basename(f)[:-5])
