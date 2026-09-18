#!/usr/bin/env python3
"""Run the WINNER'S OWN engine (l2win/) on the box's GPU.

Our numpy reimplementation absorbed 92.7% but extracted a state scoring
1e-22, while the winner's extract.py reports a ~0.1-weight peak. Rather
than keep porting, drive their modules directly -- same stack is installed
(torch 2.13.0+cu130, quimb 1.13.0, cotengra 0.7.5).
"""
import json
import os
import sys
import time

W = os.environ.get("HQP_WIN", "/mnt/8tb_hdd2/basilrari/enigma-work/l2win")
sys.path.insert(0, W)

import utils  # noqa: E402
import circuit_mpo  # noqa: E402
import unswap  # noqa: E402
import extract  # noqa: E402
from qiskit import QuantumCircuit  # noqa: E402


def load_circuit(path):
    src = open(path).read()
    try:
        from qiskit import qasm3
        return qasm3.loads(src)
    except Exception as e:
        print(f"[load] qasm3 failed ({type(e).__name__}), trying from_qasm_str", flush=True)
        return QuantumCircuit.from_qasm_str(src)


def hamming(a, b):
    return sum(x != y for x, y in zip(a, b)) if len(a) == len(b) else None


def main():
    qasm = sys.argv[1]
    budget = float(sys.argv[2]) if len(sys.argv) > 2 else 12000.0
    max_bond = int(sys.argv[3]) if len(sys.argv) > 3 else 1024
    cutoff = float(sys.argv[4]) if len(sys.argv) > 4 else 0.002
    seed = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    truth = sys.argv[6] if len(sys.argv) > 6 else None

    t0 = time.time()
    deadline = t0 + budget
    print(f"[cfg] max_bond={max_bond} cutoff={cutoff} seed={seed} budget={budget}s", flush=True)

    qc = load_circuit(qasm)
    print(f"[circuit] {qc.num_qubits} qubits, {sum(qc.count_ops().values())} gates, "
          f"ops={dict(qc.count_ops())}", flush=True)

    res = unswap.mpo_compress_unswap(
        qc, max_bond=max_bond, cutoff=cutoff, unswap_threshold=1e6,
        center_ratio=0.5, max_its=20, to_backend=utils.to_backend_cuda,
        seed=seed, hows=("both", "left", "right"), deadline=deadline)
    mpo_core, layers_left, layers_right, stats = res
    print(f"[unswap] t={time.time()-t0:.0f}s L={len(layers_left)} R={len(layers_right)}", flush=True)

    psi = unswap.mpo_to_mps(mpo_core, layers_left, layers_right,
                            max_bond=208, cutoff=1e-5,
                            to_backend=utils.to_backend_cuda)
    print(f"[mps] t={time.time()-t0:.0f}s max_bond={psi.max_bond()}", flush=True)

    bits, p0s = utils.extract_bitstring(psi)
    print(f"[marginal] pred={bits}", flush=True)
    if truth:
        print(f"[marginal] H={hamming(bits, truth)}/{len(truth)}", flush=True)

    top = extract.beam_search(psi, beam=512, k=8)
    print(f"[beam] top-{len(top)}:", flush=True)
    for bs, p in top:
        line = f"   p={p:.6e} {bs}"
        if truth:
            line += f"  H={hamming(bs, truth)}"
        print(line, flush=True)

    best = max(top, key=lambda t: t[1])[0]
    conf = extract.amp2(psi, best)
    print(f"[conf] amp2(best)={conf:.6e}", flush=True)

    out = {"answer": best, "marginal": bits, "confidence_amp2": conf,
           "elapsed_s": time.time() - t0, "max_bond_in": max_bond,
           "cutoff": cutoff, "seed": seed,
           "beam": [[b, p] for b, p in top]}
    if truth:
        out["H"] = hamming(best, truth)
    print("=== RESULT JSON ===", flush=True)
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
