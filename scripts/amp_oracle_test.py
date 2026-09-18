#!/usr/bin/env python3
"""Validate the amplitude oracle against brute-force dense simulation.

Builds a small random circuit in the exact same gate set, computes amplitudes
two independent ways -- dense statevector and the tensor-network oracle -- and
compares.  If they agree to machine precision then the whole pipeline is
trustworthy: TN construction, gate conventions, boundary pinning, and crucially
the qubit-index convention (string index i <-> qubit i).

Nothing downstream is worth building until this passes.

Usage: amp_oracle_test.py [--qubits 6] [--gates 40] [--seed 1]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import amp_oracle as ao  # noqa: E402


def random_circuit(n, ngates, rng, all_u=False):
    gates = []
    for _ in range(ngates):
        if all_u or rng.random() < 0.6 or n < 2:
            q = int(rng.integers(0, n))
            prm = [float(x) for x in rng.uniform(-np.pi, np.pi, 3)]
            gates.append(("u", q, prm))
        else:
            a, b = rng.choice(n, size=2, replace=False)
            gates.append(("cz", int(a), int(b)))
    return gates


def dense_state(n, gates):
    """Full statevector, axis q is qubit q (same convention as the oracle)."""
    psi = np.zeros((2,) * n, dtype=complex)
    psi[(0,) * n] = 1.0
    for g in gates:
        if g[0] == "u":
            _, q, prm = g
            m = np.asarray(ao.u_matrix(*prm), dtype=complex)
            psi = np.tensordot(m, psi, axes=([1], [q]))
            psi = np.moveaxis(psi, 0, q)
        else:
            _, a, b = g
            slicer = [slice(None)] * n
            slicer[a] = 1
            slicer[b] = 1
            psi[tuple(slicer)] *= -1.0
    return psi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qubits", type=int, default=6)
    ap.add_argument("--gates", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--sample", type=int, default=8)
    ap.add_argument("--all-u", action="store_true",
                    help="no cz gates (isolates the 1-qubit path)")
    args = ap.parse_args()

    n = args.qubits
    rng = np.random.default_rng(args.seed)
    gates = random_circuit(n, args.gates, rng, all_u=args.all_u)
    print("[gates] " + " ".join(
        (f"u{g[1]}" if g[0] == "u" else f"cz{g[1]},{g[2]}") for g in gates[:12]))

    psi = dense_state(n, gates)
    print(f"[test] {n} qubits, {len(gates)} gates, "
          f"norm={np.linalg.norm(psi):.12f}")

    # random strings plus the two extremes
    strings = ["0" * n, "1" * n]
    for _ in range(args.sample):
        strings.append("".join(str(int(x)) for x in rng.integers(0, 2, n)))

    worst = 0.0
    for b in strings:
        want = complex(psi[tuple(int(c) for c in b)])
        got, dt = ao.amplitude(n, gates, b, slice_bits=0)
        dev = abs(got - want)
        worst = max(worst, dev)
        print(f"  bits={b} dense={want:.6e} oracle={got:.6e} "
              f"dev={dev:.2e} {dt:.2f}s")

    print(f"[worst] absolute deviation = {worst:.3e}")
    print("VERDICT: ORACLE CORRECT" if worst < 1e-11 else "VERDICT: MISMATCH")


if __name__ == "__main__":
    main()
