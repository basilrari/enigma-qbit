#!/usr/bin/env python3
"""Look for the disguised identity block directly in the QASM gate list.

If the circuit is  P  ->  U disguised by swaps  ->  U*  ->  scramble, then the
disguise is a *combinatorial* pattern: the same gate sequence with qubit labels
relabelled and the layers interleaved.  That is far cheaper to spot than to
absorb numerically.
"""
import collections
import re
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "/home/basilsclaw/enigma-solve/samples/d2_s1_39b370e4.qasm"
src = open(path).read()
lines = [l.strip() for l in src.splitlines() if l.strip() and not l.strip().startswith("//")]

print("=== HEADER (first 12 lines) ===")
for l in lines[:12]:
    print("  ", l[:120])

GATE = re.compile(r"^([a-zA-Z_0-9]+)\s*(\(([^)]*)\))?\s*([a-zA-Z_0-9\[\],\s]+);")
gates = []
for l in lines:
    m = GATE.match(l)
    if not m:
        continue
    name, _, params, qs = m.groups()
    ql = [q.strip() for q in qs.split(",") if q.strip()]
    gates.append((name, params or "", ql))

print(f"\n=== GATES: {len(gates)} ===")
hist = collections.Counter(g[0] for g in gates)
print("  by name:", dict(hist.most_common(12)))
wide = collections.Counter(len(g[2]) for g in gates)
print("  by #qubits:", dict(wide))

params = [g[1] for g in gates if g[1]]
print(f"  gates with parameters: {len(params)}; first 6: {params[:6]}")

print("\n=== FIRST 40 GATES ===")
for g in gates[:40]:
    print("  ", g[0], g[1], g[2])

print("\n=== LAST 40 GATES ===")
for g in gates[-40:]:
    print("  ", g[0], g[1], g[2])

# per-qubit activity, to see the interaction pattern
qcount = collections.Counter()
for _, _, ql in gates:
    for q in ql:
        qcount[q] += 1
print(f"\n=== QUBITS: {len(qcount)} distinct ===")
print("  ", dict(sorted(qcount.items(), key=lambda kv: -kv[1])[:10]))
print("   least busy:", dict(sorted(qcount.items(), key=lambda kv: kv[1])[:6]))
