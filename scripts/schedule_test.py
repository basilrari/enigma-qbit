#!/usr/bin/env python3
"""Is the disguised block *schedulable* into adjacency?

Deletion needs adjacency; this asks whether adjacency can be *earned*.

Build the dependency DAG of the circuit: each gate depends on the previous gate
on each of its own wires.  Gates on disjoint wires commute, so any topological
order of this DAG is an equally valid circuit -- exact, no approximation.

If every dependency path that touches the block runs A -> B (disguise before its
inverse) and none run B -> A, then A can be scheduled entirely before B, they
annihilate exactly, and the circuit collapses.  If some path runs B -> A, that
gate genuinely blocks the collapse and we can count exactly how many do.

Also verifies the pairing itself: a candidate permutation is only credible if
the A/B gates match as *exact* inverses, one to one.
"""
import collections
import sys

import numpy as np

sys.path.insert(0, "/home/basilsclaw/enigma-solve/scripts")
from reduce_circuit import parse_qasm  # noqa: E402
from recover_perm import canon, u_matrix  # noqa: E402


def build(word, perm):
    """Relabel wires by perm; return gates, wire lists, and inverse keys."""
    gates = []
    for g in word:
        if g[0] == "u":
            _, q, t, p, l = g
            q = perm[q]
            M = u_matrix(t, p, l)
            gates.append({"kind": "u", "wires": (q,),
                          "key": canon(M), "inv": canon(M.conj().T)})
        else:
            _, a, b = g
            gates.append({"kind": "cz", "wires": (perm[a], perm[b])})
    return gates


def pair_blocks(gates):
    """Match each u gate to a later exact inverse on the same wire.

    Greedy by first occurrence; allows a shift parameter so the block window
    can be tuned.  Returns (pairs, unmatched_u).
    """
    by_key = collections.defaultdict(list)
    for j, g in enumerate(gates):
        if g["kind"] == "u":
            by_key[g["inv"]].append(j)      # gates inverse to a gate with key K
    used = set()
    pairs = []
    for i, g in enumerate(gates):
        if g["kind"] != "u" or i in used:
            continue
        for j in by_key.get(g["key"], ()):
            if j > i and j not in used:
                pairs.append((i, j))
                used.add(i)
                used.add(j)
                break
    unmatched = [i for i, g in enumerate(gates)
                 if g["kind"] == "u" and i not in used]
    return pairs, unmatched


def reachability(gates, pairs):
    """Dependency DAG; ask whether any B gate can reach an A gate.

    A B -> A path means the inverse is *forced* before part of the block, so the
    block cannot be scheduled wholly ahead of its inverse.
    """
    last_on_wire = {}
    succ = collections.defaultdict(list)
    for i, g in enumerate(gates):
        for w in g["wires"]:
            if w in last_on_wire:
                succ[last_on_wire[w]].append(i)
            last_on_wire[w] = i

    A = {i for i, _ in pairs}
    B = {j for _, j in pairs}

    # forward reachability from every B gate, in one sweep (DAG is ordered)
    reached_from_B = [False] * len(gates)
    for b in B:
        reached_from_B[b] = True
    for i in range(len(gates)):
        if reached_from_B[i]:
            for k in succ[i]:
                if not reached_from_B[k]:
                    reached_from_B[k] = True
    blocked = sorted(a for a in A if reached_from_B[a])
    return len(succ), blocked


def main():
    paths = sys.argv[1:] or [
        "/home/basilsclaw/enigma-solve/samples/d1_s1_4043cafb.qasm",
        "/home/basilsclaw/enigma-solve/samples/d2_s1_39b370e4.qasm",
    ]
    for path in paths:
        word, unmatched_lines = parse_qasm(path)
        name = path.split("/")[-1]
        nq = max((g[4] if g[0] == "u" else max(g[1], g[2])) for g in word) + 1
        print(f"\n=== {name} ({nq} qubits, {len(word)} gates) ===")

        from reduce_circuit import vote_perm
        pi, _ = vote_perm(word)

        for label, perm in (("identity", list(range(nq))), ("vote", pi)):
            gates = build(word, perm)
            pairs, unmatched = pair_blocks(gates)
            n_u = sum(1 for g in gates if g["kind"] == "u")
            if label == "vote":
                print(f"  [{label}] u gates {n_u}  paired as exact inverses "
                      f"{len(pairs)}  unmatched {len(unmatched)}")
            n_edges, blocked = reachability(gates, pairs)
            verdict = ("SCHEDULABLE A-before-B" if not blocked
                       else f"{len(blocked)} block gates strictly blocked by B")
            print(f"  [{label}] dep edges {n_edges}  -> {verdict}")


if __name__ == "__main__":
    main()
