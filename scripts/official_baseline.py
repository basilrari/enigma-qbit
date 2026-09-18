#!/usr/bin/env python3
"""Reproduce the OFFICIAL reference solver, to establish the BAR.

The published example (workbench/challenges/hardening_quantum_proof/
example_solution) does exactly two things:

  - n <= 26 qubits: exact statevector, argmax
  - n  > 26 qubits: Aer MPS at Aer's DEFAULT bond dimension (128) with
                    10000 shots, then take the most frequent outcome

That is the sanctioned baseline, and it is a *statistical* estimator, not an
exact argmax.  Chasing exact-argmax convergence was the wrong target: what
matters is whether the mode of the sampled distribution is the true peak.

We add the one thing the official example leaves unused -- chi, plus shots --
because chi is exactly what the 4-hour / 96 GB VRAM validator budget can pay
for.  Sweeping it measures the real gap.

Run on the box (needs qiskit + qiskit_aer):

  python official_baseline.py --qasm d2_s1_39b370e4.qasm \
      --truth 1110101100010111001000000011101111001000 --sweep 128,256,512
"""
import argparse
import json
import os
import sys
import time


def _load_circuit(qasm_file):
    """QASM 2 or 3; the 'u' gate is not in stock qelib1.inc, so register it."""
    from qiskit import qasm2
    from qiskit.circuit.library import UGate

    with open(qasm_file) as f:
        header = f.readline().strip()
    if "3.0" in header:
        import qiskit.qasm3 as qasm3
        return qasm3.load(qasm_file)
    custom = [qasm2.CustomInstruction('u', 3, 1,
                                      lambda t, p, lam: UGate(t, p, lam),
                                      builtin=True)]
    return qasm2.load(qasm_file, custom_instructions=custom)


def hamming(a, b):
    return sum(x != y for x, y in zip(a, b))


def run(nqubits, qasm_file, chi, shots, log=print):
    from qiskit_aer import AerSimulator

    circ = _load_circuit(qasm_file)
    opts = {"method": "matrix_product_state"}
    if chi:
        opts["matrix_product_state_max_bond_dimension"] = int(chi)
    circ.measure_all()
    backend = AerSimulator(**opts)
    t0 = time.time()
    res = backend.run(circ, shots=shots).result()
    dt = time.time() - t0
    counts = res.get_counts()
    # Qiskit reports big-endian; our convention is string index i <-> qubit i
    mode = max(counts, key=lambda k: counts[k])[::-1]
    log(f"  chi={chi if chi else 'default(128)'} shots={shots} in {dt:.1f}s -> "
        f"mode={mode}")
    return mode, counts, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qasm", required=True)
    ap.add_argument("--truth", required=True)
    ap.add_argument("--chi", default=None, help="single bond dimension")
    ap.add_argument("--sweep", default=None, help="comma list, e.g. 128,256,512")
    ap.add_argument("--shots", type=int, default=10000)
    ap.add_argument("--statevector-cutoff", type=int, default=26)
    args = ap.parse_args()

    circ = _load_circuit(args.qasm)
    n = circ.num_qubits
    print(f"[circuit] {os.path.basename(args.qasm)} qubits={n} "
          f"gates={circ.size()}", flush=True)

    # Faithful first: exactly what the official example would do.
    if n <= args.statevector_cutoff:
        import numpy as np
        from qiskit_aer import AerSimulator
        c2 = _load_circuit(args.qasm)
        c2.save_statevector()
        c2.remove_final_measurements(inplace=True)
        sv = np.asarray(AerSimulator(method="statevector").run(c2, shots=1)
                        .result().data(0)["statevector"])
        idx = int(np.argmax(np.abs(sv) ** 2))
        mode = f"{idx:0{n}b}"[::-1]
        print(f"[statevector] mode={mode}", flush=True)
    else:
        print(f"[mps] official path: default bond, {args.shots} shots", flush=True)
        mode, counts, _ = run(n, args.qasm, None, args.shots)

    h = hamming(mode, args.truth)
    print(f"[official-baseline] H={h}/{n} {'MATCH' if h == 0 else 'MISS'}",
          flush=True)

    chis = []
    if args.sweep:
        chis = [int(x) for x in args.sweep.split(",")]
    elif args.chi:
        chis = [int(args.chi)]
    for chi in chis:
        m, counts, dt = run(n, args.qasm, chi, args.shots)
        hh = hamming(m, args.truth)
        top = sorted(counts.items(), key=lambda kv: -kv[1])[:3]
        share = top[0][1] / args.shots
        print(f"[sweep] chi={chi} H={hh}/{n} "
              f"{'MATCH' if hh == 0 else 'MISS'} top_share={share:.3f} "
              f"truth_count={counts.get(args.truth[::-1], 0)}", flush=True)


if __name__ == "__main__":
    main()
