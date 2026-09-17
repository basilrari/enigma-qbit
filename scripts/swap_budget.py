#!/usr/bin/env python3
"""How many tensor truncations does the circuit actually cost, and can we
cut that by reordering?

Every 2-qubit gate in our MPS engine costs (|pos(a)-pos(b)| - 1) swaps, and
each swap is a 2-site SVD + truncation -> an error injection.  The circuit
is ~33% CZ gates with all-to-all connectivity, so this is the dominant error
source.

But ALL CZ gates commute with each other, and u gates on different qubits
commute, so we may apply the CZ gates in ANY order consistent with each
qubit's own gate sequence.  This script measures the swap count under

  (a) file order (what the engine does now), and
  (b) greedy "apply the closest ready CZ" scheduling.

Purely combinatorial - no tensor work.  If (b) is much cheaper, that is a
direct fidelity win at fixed chi.
"""
import re, os, sys

SAMPLES = "/home/basilsclaw/enigma-solve/samples"


def parse(path):
    txt = open(path).read()
    n = int(re.search(r"qreg\s+\w+\[(\d+)\]", txt).group(1))
    gates = []
    for raw in txt.splitlines():
        line = raw.strip()
        mu = re.match(r"u\(([^)]*)\)\s+q\[(\d+)\]", line)
        if mu:
            gates.append(("u", (int(mu.group(2)),)))
            continue
        mc = re.match(r"cz\s+q\[(\d+)\]\s*,\s*q\[(\d+)\]", line)
        if mc:
            gates.append(("cz", (int(mc.group(1)), int(mc.group(2)))))
    return n, gates


def file_order_swaps(n, gates):
    pos = list(range(n))
    swaps = 0
    for kind, qs in gates:
        if kind == "cz":
            a, b = qs
            swaps += abs(pos[a] - pos[b]) - 1
            # engine swaps b towards a (positions of others shift by +/-1)
            ia, ib = pos[a], pos[b]
            lo, hi = min(ia, ib), max(ia, ib)
            # after the swaps a and b are adjacent at (lo, lo+1)
            for q in range(n):
                if pos[q] > hi:
                    pos[q] -= (hi - lo - 1)
                elif lo < pos[q] < hi:
                    pos[q] -= 1
            pos[a], pos[b] = lo, lo + 1
    return swaps


def greedy_swaps(n, gates):
    per = [[] for _ in range(n)]
    for gi, (kind, qs) in enumerate(gates):
        for q in qs:
            per[q].append(gi)
    ptr = [0] * n
    applied = [False] * len(gates)
    pos = list(range(n))
    swaps = 0
    napp = 0
    total = len(gates)
    while napp < total:
        # ready CZ gates
        best = None
        for gi, (kind, qs) in enumerate(gates):
            if kind != "cz" or applied[gi]:
                continue
            a, b = qs
            if per[a][ptr[a]] == gi and per[b][ptr[b]] == gi:
                d = abs(pos[a] - pos[b])
                if best is None or d < best[0]:
                    best = (d, gi, a, b)
        if best is not None:
            d, gi, a, b = best
            swaps += d - 1
            ia, ib = pos[a], pos[b]
            lo, hi = min(ia, ib), max(ia, ib)
            for q in range(n):
                if pos[q] > hi:
                    pos[q] -= (hi - lo - 1)
                elif lo < pos[q] < hi:
                    pos[q] -= 1
            pos[a], pos[b] = lo, lo + 1
            applied[gi] = True
            ptr[a] += 1
            ptr[b] += 1
            napp += 1
        else:
            # no ready CZ -> must apply a single-qubit gate
            did = False
            for q in range(n):
                if ptr[q] >= len(per[q]):
                    continue
                gi = per[q][ptr[q]]
                if not applied[gi] and gates[gi][0] == "u":
                    applied[gi] = True
                    ptr[q] += 1
                    napp += 1
                    did = True
                    break
            if not did:
                # stuck: apply any unapplied gate
                for gi in range(total):
                    if not applied[gi]:
                        applied[gi] = True
                        for q in gates[gi][1]:
                            ptr[q] += 1
                        napp += 1
                        break
    return swaps


print(f"{'circuit':<20} {'qubits':>6} {'CZ':>5} {'file-order swaps':>18} "
      f"{'greedy swaps':>13} {'reduction':>10} {'swaps/CZ (file)':>16}")
print("-" * 100)
for f in sorted(os.listdir(SAMPLES)):
    if not f.endswith(".qasm") or f.startswith("d0"):
        continue
    n, gates = parse(os.path.join(SAMPLES, f))
    cz = sum(1 for k, _ in gates if k == "cz")
    sf = file_order_swaps(n, gates)
    sg = greedy_swaps(n, gates)
    print(f"{f[:-5]:<20} {n:>6} {cz:>5} {sf:>18} {sg:>13} "
          f"{sf/max(sg,1):>9.2f}x {sf/max(cz,1):>16.2f}")
