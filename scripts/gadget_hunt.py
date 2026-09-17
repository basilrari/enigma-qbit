#!/usr/bin/env python3
"""GADGET HUNT.

Every circuit contains u-gates whose parameters are exact pi-literals or
tiny-perturbation versions of exact Clifford values.  Clifford gates are
where a generator would hide a plant.  Extract them all, with the KNOWN
answers as ground truth, and test: is the answer encoded in them?

Decode logic: U(t,p,l) = [[c, -e^{il} s], [e^{ip} s, e^{i(p+l)} c]], c=cos(t/2), s=sin(t/2)
  t=0      -> identity (up to phase)
  t=pi     -> X (up to phase)
  t=pi/2   -> H-like: splits |0> into (|0>+e^{ip}|1>)/sqrt2
  l=pi     -> makes the off-diagonal -e^{il}s = +s  (no minus sign)
"""
import re, os, json, math
from collections import Counter

S = "/home/basilsclaw/enigma-solve/samples"
SAMPLES = ["d1_s1_4043cafb", "d1_s2_adeddcf3", "d2_s1_39b370e4",
           "d2_s2_1efabaf4", "d3_s1_2674779a", "d3_s2_c09ba537"]
TOL = 0.05
HALFPI = math.pi / 2


def nearest_multiple(v, unit=HALFPI):
    k = round(v / unit)
    return k, abs(v - k * unit)


def parse(path):
    txt = open(path).read()
    n = int(re.search(r"qreg\s+\w+\[(\d+)\]", txt).group(1))
    gates = []
    for raw in txt.splitlines():
        line = raw.strip()
        if line.startswith("u("):
            m = re.match(r"u\(([^)]*)\)\s+q\[(\d+)\]", line)
            if m:
                prm = [p.strip() for p in m.group(1).split(",")]
                pf = []
                for p in prm:
                    try:
                        pf.append(float(eval(p, {"__builtins__": {}}, {"pi": math.pi})))
                    except Exception:
                        pf.append(float("nan"))
                gates.append(("u", int(m.group(2)), prm, pf))
        elif line.startswith("cz"):
            qs = [int(x) for x in re.findall(r"q\[(\d+)\]", line)]
            gates.append(("cz", qs, None, None))
    return n, gates


print("=" * 100)
print("A. ALL u-GATES WITH A LITERAL 'pi' PARAMETER  (position%, qubit, θ,φ,λ, decoded)")
print("=" * 100)

for cid in SAMPLES:
    meta = json.load(open(f"{S}/{cid}_meta.json"))
    peak = str(meta["peaked_state"])
    n, gates = parse(f"{S}/{cid}.qasm")
    print(f"\n### {cid}  n={n}  ngates={len(gates)}  peak={peak}")
    rows = []
    for idx, (k, q, raw, pf) in enumerate(gates):
        if k != "u":
            continue
        if any("pi" in r for r in raw):
            pos = 100.0 * idx / len(gates)
            rows.append((idx, pos, q, raw, pf))
    print(f"  count={len(rows)}")
    for idx, pos, q, raw, pf in rows:
        t = pf[0] if len(pf) > 0 else float("nan")
        p = pf[1] if len(pf) > 1 else float("nan")
        l = pf[2] if len(pf) > 2 else float("nan")
        bit = peak[q] if q < len(peak) else "?"
        # classify
        def cls(v):
            k, d = nearest_multiple(v)
            return f"{k}*pi/2(d={d:.2e})"
        print(f"   idx={idx:5d} pos={pos:5.1f}% q={q:2d} bit={bit}  "
              f"t={t:.6f}[{cls(t)}]  phi={p:.6f}[{cls(p)}]  lam={l:.6f}[{cls(l)}]"
              f"   raw={raw}")

print()
print("=" * 100)
print("B. NEAR-SPECIAL GATES WITHOUT pi LITERALS (all three angles within %.3f of k*pi/2)" % TOL)
print("=" * 100)
for cid in SAMPLES:
    meta = json.load(open(f"{S}/{cid}_meta.json"))
    peak = str(meta["peaked_state"])
    n, gates = parse(f"{S}/{cid}.qasm")
    hits = []
    for idx, (k, q, raw, pf) in enumerate(gates):
        if k != "u" or len(pf) != 3:
            continue
        ds = [nearest_multiple(v)[1] for v in pf]
        if all(d < TOL for d in ds):
            hits.append((idx, q, pf, ds, raw))
    print(f"\n### {cid}: {len(hits)} near-special gates")
    for idx, q, pf, ds, raw in hits:
        bit = peak[q] if q < len(peak) else "?"
        anypi = any("pi" in r for r in raw)
        print(f"   idx={idx:5d} q={q:2d} bit={bit} devs=({ds[0]:.1e},{ds[1]:.1e},{ds[2]:.1e}) anypi={anypi} raw={raw}")

print()
print("=" * 100)
print("C. DOES THE GATE QUIBIT CORRELATE WITH THE ANSWER BIT?")
print("=" * 100)
for cid in SAMPLES:
    meta = json.load(open(f"{S}/{cid}_meta.json"))
    peak = str(meta["peaked_state"])
    n, gates = parse(f"{S}/{cid}.qasm")
    pis = [g for g in gates if g[0] == "u" and any("pi" in r for r in g[2])]
    qs = [g[1] for g in pis]
    ones = sum(1 for q in qs if q < len(peak) and peak[q] == "1")
    base = peak.count("1") / len(peak)
    print(f"  {cid}: {len(qs)} gadget qubits, {ones} on bit=1 "
          f"({ones/max(len(qs),1):.3f}) vs baseline {base:.3f}")
