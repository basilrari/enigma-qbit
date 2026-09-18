#!/usr/bin/env python3
"""Find the disguised identity block by parameters alone -- permutation-blind.

A qubit permutation relabels wires but changes no rotation angle, so the
multiset of `u(theta,phi,lam)` triples inside the disguised block is identical
to the block it disguises.  The inverse of U(t,p,l) is U(-t,-l,-p) (note the
phi/lam swap in the Rz-Ry-Rz convention), so if the circuit contains U then
U*, the parameter multiset must pair up under that inversion.

This test never looks at which qubit a gate acts on, so the disguise cannot
hide from it.
"""
import collections
import math
import re
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "/home/basilsclaw/enigma-solve/samples/d2_s1_39b370e4.qasm"
lines = open(path).read().splitlines()

UG = re.compile(r"^u\(([^)]*)\)\s+q\[(\d+)\];")


def _num(tok):
    """QASM parameters may be expressions like -pi or pi/2."""
    tok = tok.strip()
    try:
        return float(tok)
    except ValueError:
        return float(eval(tok.replace("pi", repr(math.pi)), {"__builtins__": {}}, {}))


us = []          # (theta, phi, lam, qubit) in circuit order
czs = []         # (q1, q2) in circuit order
for l in (x.strip() for x in lines):
    m = UG.match(l)
    if m:
        p = [_num(v) for v in m.group(1).split(",")]
        us.append((p[0], p[1], p[2], int(m.group(2))))
        continue
    m = re.match(r"^cz\s+q\[(\d+)\],q\[(\d+)\];", l)
    if m:
        czs.append((int(m.group(1)), int(m.group(2))))

print(f"u gates: {len(us)}   cz gates: {len(czs)}")


def key_inv(t, p, l):
    """Canonical key of the inverse of U(t,p,l) = U(-t,-l,-p)."""
    return (round(-t, 9), round(-l, 9), round(-p, 9))


def key_fwd(t, p, l):
    return (round(t, 9), round(p, 9), round(l, 9))


# --- global pairing: how many u gates have an inverse partner somewhere? ---
fwd = collections.Counter(key_fwd(t, p, l) for t, p, l, _ in us)
inv = collections.Counter(key_inv(t, p, l) for t, p, l, _ in us)
paired = sum(min(fwd[k], inv[k]) for k in inv)
print(f"\n=== GLOBAL PAIRING ===")
print(f"  u gates whose inverse also appears: {paired} / {len(us)}"
      f"  ({100.0 * paired / len(us):.1f}%)")
exact = sum(min(fwd[k], fwd[k]) for k in [] )
selfinv = sum(1 for t, p, l, _ in us if key_fwd(t, p, l) == key_inv(t, p, l))
print(f"  self-inverse gates: {selfinv}")

# --- where do the partners sit?  a fixed offset means a translated block ---
pos = collections.defaultdict(list)
for i, (t, p, l, _) in enumerate(us):
    pos[key_fwd(t, p, l)].append(i)
offsets = collections.Counter()
for i, (t, p, l, _) in enumerate(us):
    for j in pos.get(key_inv(t, p, l), [])[:40]:
        offsets[j - i] += 1
print(f"\n=== PARTNER OFFSETS (top 12) ===")
for off, n in offsets.most_common(12):
    print(f"   offset {off:+5d}: {n}")

# --- split scan: is there a boundary where left U* == right U as multisets? ---
print(f"\n=== SPLIT SCAN (best 8) ===")
K = len(us)
best = []
for s in range(100, K - 100, 10):
    left = collections.Counter(key_inv(t, p, l) for t, p, l, _ in us[:s])
    right = collections.Counter(key_fwd(t, p, l) for t, p, l, _ in us[s:])
    match = sum(min(left[k], right[k]) for k in left)
    frac = match / max(1, min(sum(left.values()), sum(right.values())))
    best.append((frac, s, match))
best.sort(reverse=True)
for frac, s, match in best[:8]:
    print(f"   split at u-index {s:5d} (circuit gate {s}, {100*s/K:.0f}%): "
          f"{match} matched, {100*frac:.1f}% of the smaller side")
