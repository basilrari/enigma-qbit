#!/usr/bin/env python3
"""Run the WINNER'S OWN engine on the box's GPU, with correct bit order.

Two corrections over the first version:
  1. Use unswap.mpo_to_mps() -- the engine's OWN leftover application. Our
     l2_absorb.py hand-rolled this because the stallfix file lacks the
     function; guessing the composition order moves amplitudes by orders.
  2. Carry the site->logical permutation. unswap() relabels the circuit
     (rewire_layers uses qubits=argsort(perm)) and permutes the MPO, so
     extract.py's bitstrings are in MPS-SITE order and MUST be remapped.
     Same wrapper as scripts/l2_absorb.py, which is proven.
"""
import json
import os
import sys
import time

W = os.environ.get("HQP_WIN", "/mnt/8tb_hdd2/basilrari/enigma-work/l2win")
sys.path.insert(0, W)

import utils  # noqa: E402
import unswap  # noqa: E402
import extract  # noqa: E402
from qiskit import QuantumCircuit  # noqa: E402

_PERM = None
_N = None
_ORIG_UNSWAP = unswap.unswap
_CALLS = {"n": 0}


def _tracking_unswap(mpo, *a, **kw):
    """pl[i] = the previous site whose content now sits at site i."""
    global _PERM
    mpo2, (pl, pr), stats = _ORIG_UNSWAP(mpo, *a, **kw)
    if _PERM is not None and pl is not None:
        _PERM = [_PERM[pl[i]] for i in range(_N)]
    _CALLS["n"] += 1
    return mpo2, (pl, pr), stats


def to_logical(site_bits, perm):
    return "".join(site_bits[perm.index(j)] for j in range(len(site_bits)))


def to_site(logical_bits, perm):
    return "".join(logical_bits[perm[s]] for s in range(len(logical_bits)))


def hamming(a, b):
    return sum(x != y for x, y in zip(a, b)) if a and b and len(a) == len(b) else None


def load_circuit(path):
    txt = open(path).read()
    try:
        from qiskit import qasm2
        import re
        txt2 = re.sub(r"(?<![A-Za-z0-9_])u\s*\(", "u3(", txt)
        return qasm2.loads(txt2)
    except Exception as e:
        print(f"[load] qasm2 failed ({type(e).__name__}), trying from_qasm_str", flush=True)
        return QuantumCircuit.from_qasm_str(txt)


def main():
    global _PERM, _N
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
    n = qc.num_qubits
    nodes = sum(qc.count_ops().values())
    print(f"[circuit] {n} qubits, {nodes} gates, ops={dict(qc.count_ops())}", flush=True)

    _PERM = list(range(n))
    _N = n
    unswap.unswap = _tracking_unswap

    mpo_core, L_left, L_right, stats = unswap.mpo_compress_unswap(
        qc, max_bond=max_bond, cutoff=cutoff, unswap_threshold=1e6,
        center_ratio=0.5, max_its=20, to_backend=utils.to_backend_cuda,
        seed=seed, hows=("both", "left", "right"), deadline=deadline)
    real_l = [l for l in L_left if "measure" not in l.count_ops()]
    real_r = [l for l in L_right if "measure" not in l.count_ops()]
    print(f"[unswap] t={time.time()-t0:.0f}s calls={_CALLS['n']} "
          f"L={len(real_l)} R={len(real_r)} perm={_PERM[:6]}...", flush=True)

    # the engine's OWN finalizer -- not a hand-rolled replay of the leftovers.
    # NOTE: it returns (mps, final_perm), and it merges+inverts the left layers
    # and applies them to a FRESH |0..0> MPS: left layers -> core MPO -> right.
    psi, final_perm = unswap.mpo_to_mps(mpo_core, L_left, L_right, max_bond=208,
                                        cutoff=1e-5, to_backend=utils.to_backend_cuda)
    print(f"[mps] t={time.time()-t0:.0f}s bond={psi.max_bond()} "
          f"norm={float(abs(complex(psi.norm()))):.6f} final_perm={list(final_perm)[:6]}...",
          flush=True)

    # Which site->logical convention is right? Score them all with amp2, which
    # NORMALISES, so the values are directly comparable. Target 1e-3..1e-1.
    cands = {"identity": list(range(n)), "tracked": list(_PERM), "final": list(final_perm)}
    try:
        cands["tracked^-1"] = [list(_PERM).index(j) for j in range(n)]
    except Exception:
        pass
    if truth:
        print("[diag] amp2(truth) per convention (target 1e-3..1e-1):", flush=True)
        for name, pm in cands.items():
            try:
                a2 = extract.amp2(psi, to_site(truth, pm))
                print(f"    {name:12s} {a2:.6e}", flush=True)
            except Exception as e:
                print(f"    {name:12s} failed: {type(e).__name__}", flush=True)

    bits_site, _p0 = utils.extract_bitstring(psi)
    print(f"[marginal] site={bits_site}", flush=True)
    if truth:
        print(f"[marginal] logical={to_logical(bits_site, _PERM)} "
              f"H={hamming(to_logical(bits_site, _PERM), truth)}/{n}", flush=True)

    top = extract.beam_search(psi, beam=512, k=8)
    print(f"[beam] {len(top)} candidates (site -> logical):", flush=True)
    out_top = []
    for i, (bs, p) in enumerate(top):
        bs = str(bs)
        lg = to_logical(bs, _PERM)
        line = f"  #{i+1} w={p:.6e} site={bs} logical={lg}"
        h = hamming(lg, truth) if truth else None
        if h is not None:
            line += f" H={h}/{n}"
        print(line, flush=True)
        out_top.append([lg, p, h])

    best = max(out_top, key=lambda t: t[1])
    print(f"[conf] amp2(best logical) = {extract.amp2(psi, to_site(best[0], _PERM)):.6e}", flush=True)
    print("=== RESULT JSON ===", flush=True)
    print(json.dumps({"answer_logical": best[0], "H": best[2], "weight": best[1],
                      "elapsed_s": time.time() - t0, "max_bond": max_bond,
                      "seed": seed, "n": n, "top": out_top}), flush=True)


if __name__ == "__main__":
    main()
