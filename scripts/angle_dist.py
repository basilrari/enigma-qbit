#!/usr/bin/env python3
"""DECISIVE TEST: is the near-special-gate excess a planted gadget, or just
an artifact of a skewed angle-sampling distribution?

Measures the true marginal distribution of theta, phi, lambda over every
u-gate in every sample, plus the N(TOL) scaling law for "distance to nearest
multiple of pi/2".  A planted Clifford/identity gadget shows up as a spike at
d~0 that does NOT follow the distribution's own density.
"""
import re, os, math, json
from collections import Counter

S = "/home/basilsclaw/enigma-solve/samples"
SAMPLES = ["d1_s1_4043cafb", "d1_s2_adeddcf3", "d2_s1_39b370e4",
           "d2_s2_1efabaf4", "d3_s1_2674779a", "d3_s2_c09ba537"]


def parse(path):
    txt = open(path).read()
    n = int(re.search(r"qreg\s+\w+\[(\d+)\]", txt).group(1))
    out = []
    for raw in txt.splitlines():
        line = raw.strip()
        if not line.startswith("u("):
            continue
        m = re.match(r"u\(([^)]*)\)\s+q\[(\d+)\]", line)
        if not m:
            continue
        prm = [p.strip() for p in m.group(1).split(",")]
        v = []
        for p in prm:
            try:
                v.append(float(eval(p, {"__builtins__": {}}, {"pi": math.pi})))
            except Exception:
                v.append(float("nan"))
        if len(v) == 3:
            out.append((int(m.group(2)), prm, v))
    return n, out


allg = {}
for cid in SAMPLES:
    n, g = parse(f"{S}/{cid}.qasm")
    allg[cid] = g
    print(f"{cid:22s} n={n:3d} u-gates={len(g)}")

print("\n" + "=" * 96)
print("1. RANGE CHECK — do angles stay inside [0, 2pi)  (excl. exact-pi literals)?")
print("=" * 96)
for j, nm in enumerate(["theta", "phi", "lambda"]):
    vals = [v[j] for cid in SAMPLES for _, _, v in allg[cid]
            if not math.isnan(v[j]) and "pi" not in allg[cid][0][1][0]]
    vals = [v for v in vals if not math.isnan(v)]
    if vals:
        print(f"  {nm:7s} n={len(vals):6d}  min={min(vals):9.4f}  max={max(vals):9.4f}  "
              f"mean={sum(vals)/len(vals):8.4f}")
print(f"  (2pi = {2*math.pi:.4f},  pi = {math.pi:.4f}; negatives/extras imply non-[0,2pi) sampling)")

print("\n" + "=" * 96)
print("2. HISTOGRAM of theta and lambda over [-2pi, 8pi], 20 bins")
print("=" * 96)
for j, nm in enumerate(["theta", "phi", "lambda"]):
    vals = [v[j] for cid in SAMPLES for _, _, v in allg[cid] if not math.isnan(v[j])]
    if not vals:
        continue
    lo, hi = -2 * math.pi, 8 * math.pi
    nb = 20
    cnt = [0] * nb
    for v in vals:
        b = int((v - lo) / (hi - lo) * nb)
        b = max(0, min(nb - 1, b))
        cnt[b] += 1
    print(f"  {nm}: total={len(vals)}")
    mx = max(cnt)
    for b in range(nb):
        a = lo + (hi - lo) * b / nb
        bb = lo + (hi - lo) * (b + 1) / nb
        bar = "#" * int(60 * cnt[b] / mx)
        print(f"    [{a:7.2f},{bb:7.2f}) {cnt[b]:6d} {bar}")

print("\n" + "=" * 96)
print("3. N(TOL): gates with ALL THREE angles within TOL of a multiple of pi/2")
print("   (uniform-angle prediction ~ n * (8*TOL/2pi)^3 for TOL small)")
print("=" * 96)
for cid in SAMPLES:
    g = allg[cid]
    vals = [v for _, _, v in g if not all(math.isnan(x) for x in v)]
    N = len(vals)
    row = []
    for TOL in [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]:
        c = 0
        for v in vals:
            ok = True
            for x in v:
                if math.isnan(x):
                    ok = False
                    break
                d = abs(x - round(x / (math.pi / 2)) * (math.pi / 2))
                if d > TOL:
                    ok = False
                    break
            if ok:
                c += 1
        pred = N * (8 * TOL / (2 * math.pi)) ** 3
        row.append(f"TOL={TOL}:{c:5d}(pred {pred:8.2f})")
    print(f"  {cid:22s} N={N:5d}  " + "  ".join(row))

print("\n" + "=" * 96)
print("4. SINGLE-ANGLE near-zero excess: P(|v| < 0.05) vs uniform 0.0159")
print("=" * 96)
for j, nm in enumerate(["theta", "phi", "lambda"]):
    vals = [v[j] for cid in SAMPLES for _, _, v in allg[cid] if not math.isnan(v[j])]
    for TOL in [0.001, 0.01, 0.05, 0.3]:
        c = sum(1 for v in vals if abs(v) <= TOL or abs(abs(v) - 2 * math.pi) <= TOL)
        print(f"  {nm:7s} TOL={TOL:<6} count={c:6d} frac={c/len(vals):.4f}  "
              f"uniform_frac={2*TOL/math.pi:.4f}  ratio={c/len(vals)/(2*TOL/math.pi):6.2f}x")
