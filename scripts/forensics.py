#!/usr/bin/env python3
"""Forensic pass over the sample circuits, with the KNOWN answers as ground truth.

Question: is the planted peak recoverable from circuit STRUCTURE rather than
simulation? Test every locally-measurable feature for correlation with the
known answer bits, and look for disguised / no-op gadget layers.
"""
import re, os, json, math, itertools
from collections import Counter, defaultdict

S = "/home/basilsclaw/enigma-solve/samples"
OUT = "/home/basilsclaw/enigma-solve/forensics_report.txt"

SAMPLES = ["d0_s0_trivial", "d1_s1_4043cafb", "d1_s2_adeddcf3",
           "d2_s1_39b370e4", "d2_s2_1efabaf4",
           "d3_s1_2674779a", "d3_s2_c09ba537"]

lines = []
def w(s=""):
    lines.append(str(s))

def parse(path):
    txt = open(path).read()
    n = int(re.search(r"qreg\s+\w+\[(\d+)\]", txt).group(1))
    gates = []  # (kind, qubits, params_raw, params_float)
    for raw in txt.splitlines():
        line = raw.strip()
        if not line or line.startswith("//") or line.startswith("OPENQASM") \
           or line.startswith("include") or line.startswith("qreg") \
           or line.startswith("creg") or line.startswith("barrier") \
           or line.startswith("measure"):
            continue
        m = re.match(r"u\(([^)]*)\)\s+q\[(\d+)\]", line)
        if m:
            prm = [p.strip() for p in m.group(1).split(",")]
            try:
                pf = [float(p) for p in prm]
            except ValueError:
                pf = None
            gates.append(("u", [int(m.group(2))], prm, pf))
            continue
        m = re.match(r"cz\s+(.*);", line)
        if m:
            qs = [int(x) for x in re.findall(r"q\[(\d+)\]", m.group(1))]
            gates.append(("cz", qs, None, None))
            continue
        m = re.match(r"cx\s+(.*);", line)
        if m:
            qs = [int(x) for x in re.findall(r"q\[(\d+)\]", m.group(1))]
            gates.append(("cx", qs, None, None))
            continue
        gates.append(("OTHER:" + line.split()[0], [], None, None))
    return n, gates

def load_peak(cid):
    p = f"{S}/{cid}_meta.json"
    d = json.load(open(p))
    k = [k for k in d.keys() if "peak" in k.lower() or "state" in k.lower()]
    return d, k

w("=" * 78)
w("FORENSIC REPORT — structure vs known answer")
w("=" * 78)

for cid in SAMPLES:
    meta, keys = load_peak(cid)
    peak = meta.get("peaked_state")
    path = f"{S}/{cid}.qasm"
    if not os.path.exists(path):
        w(f"\n### {cid}: QASM MISSING")
        continue
    n, gates = parse(path)
    kinds = Counter(g[0] for g in gates)
    w(f"\n### {cid}  qubits={n}  gates={len(gates)}  peak={peak}")
    w(f"    meta keys: {sorted(meta.keys())}")
    w(f"    gate mix: {dict(kinds)}")
    if peak is None:
        continue
    peak = str(peak)
    w(f"    peak len={len(peak)}  ones={peak.count('1')}")

    # --- symbolic / exact-special angles in u gates ---
    special = []
    for idx, (k, qs, prm, pf) in enumerate(gates):
        if k != "u":
            continue
        for j, p in enumerate(prm):
            if re.search(r"pi", p):
                special.append((idx, qs[0], j, p))
                break
    w(f"    u-gates with SYMBOLIC 'pi' in any angle: {len(special)}")
    if special:
        pos = [100.0 * s[0] / len(gates) for s in special]
        dec = Counter(min(9, int(p // 10)) for p in pos)
        w(f"      decile histogram: {[dec.get(i,0) for i in range(10)]}")
        ctr = Counter(s[3] for s in special)
        w(f"      raw param strings: {dict(ctr.most_common(8))}")
        # does the qubit of a special gate correlate with the answer bit?
        agree = sum(1 for s in special if s[1] < len(peak) and peak[s[1]] == "1")
        w(f"      special-gate qubits with peak bit = 1: {agree}/{len(special)}")
        w(f"      first 6 special gates (idx,qubit,pos%): "
          f"{[(s[0], s[1], round(100.0*s[0]/len(gates),1)) for s in special[:6]]}")

    # --- numeric special angles (multiples of pi/4) ---
    num_special = []
    for idx, (k, qs, prm, pf) in enumerate(gates):
        if k != "u" or pf is None:
            continue
        for j, v in enumerate(pf):
            r = v / (math.pi / 4)
            if abs(r - round(r)) < 1e-9 and abs(round(r)) % 2 == 0:
                num_special.append((idx, qs[0], j, round(r)))
                break
    w(f"    u-gates with NUMERIC multiple-of-pi/2 angle: {len(num_special)}")
    if num_special:
        ctr = Counter(s[3] for s in num_special)
        w(f"      angle values (units of pi/4): {dict(ctr.most_common(8))}")

    # --- per-qubit gate counts vs answer bit ---
    ugc = Counter()
    czdeg = Counter()
    for k, qs, prm, pf in gates:
        if k == "u":
            ugc[qs[0]] += 1
        elif k == "cz":
            for q in qs:
                czdeg[q] += 1
    ones = [q for q in range(min(n, len(peak))) if peak[q] == "1"]
    zeros = [q for q in range(min(n, len(peak))) if peak[q] == "0"]
    def mean(c, qs):
        return sum(c[q] for q in qs) / max(len(qs), 1)
    w(f"    u-count  mean: bit1={mean(ugc,ones):.1f}  bit0={mean(ugc,zeros):.1f}")
    w(f"    cz-deg   mean: bit1={mean(czdeg,ones):.1f}  bit0={mean(czdeg,zeros):.1f}")

    # --- LAST gate touching each qubit: param vs bit ---
    lastu = {}
    firstu = {}
    for idx, (k, qs, prm, pf) in enumerate(gates):
        if k == "u":
            lastu[qs[0]] = (idx, prm)
            firstu.setdefault(qs[0], (idx, prm))
    w(f"    qubits with no u gate: "
      f"{[q for q in range(n) if q not in lastu]}")

    # --- block-structure test: split circuit in halves, count cz inside each ---
    half = len(gates) // 2
    cz1 = sum(1 for i, g in enumerate(gates) if i < half and g[0] == "cz")
    cz2 = sum(1 for i, g in enumerate(gates) if i >= half and g[0] == "cz")
    w(f"    cz in first half={cz1}  second half={cz2}")

print("\n".join(lines))
with open(OUT, "w") as f:
    f.write("\n".join(lines))
print(f"\n[written to {OUT}]")
