#!/usr/bin/env python3
"""Rebuild the L2 winner's orchestrator: absorb -> extract -> diagnose.

The published engine is intact but its orchestrator was lost. This wires it up:

  1. QASM (u + cz)  ->  every gate as a UnitaryGate, preserving the DAG layer
     structure the engine iterates over (it counts 'unitary' ops as its unit).
  2. mpo_compress_unswap(...)  ->  (mpo_core, layers_left, layers_right, stats)
     The site->logical permutation it accumulates is tracked by wrapping unswap,
     because the function itself does not return it.
  3. Unabsorbed layers are applied in the engine's own composition order
     (left leftovers inverted on side='right', right ones on side='left';
     measurement layers stripped -- they are bookkeeping, not state).
  4. The circuit operator is turned into a state by projecting the input leg to
     |0...0>, then:
       extract.amp2(mps, truth)  ->  THE decisive diagnostic
       > ~1e-8  : peak is BURIED (recoverable by exact re-scoring at higher chi)
       ~ eps    : peak was DISCARDED by the truncation (fix absorption, not search)
  5. extract.beam_search(...) -> candidate, Hamming distance vs known truth.

Run:  l2_absorb.py --qasm X.qasm [--max-bond 8192 --cutoff 1e-3 --budget 36000]
                   [--early-stop 30] [--device cpu|cuda:0] [--truth BITS]
"""
import argparse
import json
import logging
import os
import sys
import time

import numpy as np

L2 = "/mnt/8tb_hdd2/basilrari/enigma-work/l2win"
sys.path.insert(0, L2)

from qiskit import QuantumCircuit                                    # noqa: E402
from qiskit.circuit.library import UnitaryGate                        # noqa: E402
from qiskit.quantum_info import Operator                              # noqa: E402
import quimb.tensor as qtn                                            # noqa: E402

import utils as UT                                                    # noqa: E402
import extract as EX                                                  # noqa: E402
import unswap_stallfix as US                                          # noqa: E402
from utils import quimb_circuit                                   # noqa: E402

Circuit = qtn.Circuit

# --- permutation tracking: wrap unswap so the site<->logical map is kept -----
_PERM = None
_N = None
_ORIG_UNSWAP = US.unswap
_UNSWAP_CALLS = {"n": 0}


def _tracking_unswap(mpo, *a, **kw):
    mpo2, (pl, pr), stats = _ORIG_UNSWAP(mpo, *a, **kw)
    global _PERM
    if _PERM is not None and pl is not None:
        # pl[i] = the previous site whose content now sits at site i
        _PERM = [_PERM[pl[i]] for i in range(_N)]
    _UNSWAP_CALLS["n"] += 1
    return mpo2, (pl, pr), stats


def load_qasm(path):
    """QASM2 via qiskit; the samples mix QASM3 constructs under a QASM2 header.

    'u(theta,phi,lambda)' is not in qelib1.inc (that is u3) -- identical gate,
    identical argument order, so rewriting is loss-free.
    """
    import re
    txt = open(path).read()
    if "OPENQASM 3" in txt:
        txt = txt.replace("OPENQASM 3.0;", "OPENQASM 2.0;")
        txt = txt.replace("stdgates.inc", "qelib1.inc")
        txt = re.sub(r"qubit\[(\d+)\]\s+(\w+)\s*;", r"qreg \2[\1];", txt)
    txt = re.sub(r"(?<![A-Za-z0-9_])u\s*\(", "u3(", txt)
    from qiskit import qasm2
    return qasm2.loads(txt)


def unitarize(qc):
    """u/cz -> one UnitaryGate per gate. DAG layers are unchanged.

    Non-unitary bookkeeping instructions (measure/barrier/reset/delay) are
    DROPPED: they do not change the state, and trying to build a UnitaryGate
    from them raises "Cannot apply Operation: measure".  The real samples happen
    to carry none, but relying on that is luck, not correctness.
    """
    skip = {"measure", "barrier", "reset", "delay", "id"}
    out = QuantumCircuit(qc.num_qubits)
    for inst in qc.data:
        if inst.operation.name in skip:
            continue
        qidx = [qc.find_bit(q).index for q in inst.qubits]
        out.append(UnitaryGate(Operator(inst.operation).data, label=inst.operation.name), qidx)
    return out


def is_measure_layer(layer):
    return all(op.operation.name in ("measure", "barrier") for op in getattr(layer, "data", []))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qasm", required=True)
    ap.add_argument("--truth", default=None)
    ap.add_argument("--max-bond", type=int, default=1024, help="greedy absorb cap (winners: 1024)")
    ap.add_argument("--cutoff", type=float, default=0.002, help="greedy absorb cutoff (winners: 0.002 loose)")
    ap.add_argument("--cutoff-final", type=float, default=1e-5,
                    help="final reconstruction cutoff (winners: 1e-5; dossier says tighten to 1e-6/1e-7)")
    ap.add_argument("--max-bond-final", type=int, default=4096, help="final reconstruction cap (winners: 4096)")
    ap.add_argument("--seed", type=int, default=0, help="ordering seed (independent orderings for consensus)")
    ap.add_argument("--budget", type=float, default=36000.0, help="wall seconds for absorption")
    ap.add_argument("--early-stop", type=int, default=30)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--beam", type=int, default=512)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--center-ratio", type=float, default=0.5)
    ap.add_argument("--save-mps", default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if os.environ.get("HQP_QUIET"):
        logging.getLogger().setLevel(logging.WARNING)

    t0 = time.time()
    qc = load_qasm(args.qasm)
    n = qc.num_qubits
    uqc = unitarize(qc)
    print(f"# circuit {os.path.basename(args.qasm)}: {n} qubits, "
          f"{len(qc.data)} gates -> {uqc.count_ops().get('unitary', 0)} unitary blocks, "
          f"{len(list(UT.iter_layers(uqc)))} layers", flush=True)

    global _PERM, _N
    _N = n
    _PERM = list(range(n))
    US.unswap = _tracking_unswap

    to_backend = None
    if args.device != "cpu":
        to_backend = lambda x: UT.to_backend_cuda(x, args.device)  # noqa: E731

    deadline = t0 + args.budget
    mpo, L_left, L_right, stats = US.mpo_compress_unswap(
        uqc, max_bond=args.max_bond, cutoff=args.cutoff, unswap_threshold=1e6,
        early_stopping_gates=args.early_stop, center_ratio=args.center_ratio,
        to_backend=to_backend, seed=args.seed, deadline=deadline)
    t_abs = time.time() - t0
    real_left = [l for l in L_left if not is_measure_layer(l)]
    real_right = [l for l in L_right if not is_measure_layer(l)]
    tot_gates = sum(sum(ops.values()) for ops in [dict(l.count_ops()) for l in UT.iter_layers(uqc)])
    left_gates = sum(sum(dict(l.count_ops()).values()) for l in real_left + real_right)
    absorbed = 1.0 - left_gates / max(1, tot_gates)
    bonds = [int(mpo.bond_size(i, i + 1)) for i in range(len(mpo.sites) - 1)]
    print(f"# absorb: {t_abs:.0f}s | unswap calls {_UNSWAP_CALLS['n']} | "
          f"leftover layers L={len(real_left)} R={len(real_right)} "
          f"({left_gates}/{tot_gates} gates left => {absorbed*100:.1f}% absorbed)\n"
          f"# bond max={max(bonds) if bonds else 0} mean={np.mean(bonds):.1f} "
          f"profile={bonds[:12]}...", flush=True)

    # --- apply the leftovers exactly as the engine's loop would have ---------
    # The trailing pass reconstructs the peak: the winners keep the greedy absorb
    # loose (0.002) but the FINAL reconstruction tight (1e-5). Their own L3 data
    # says bond is not the limiter -- cutoff fidelity is -- so this is the knob.
    q2c = lambda c: quimb_circuit(c.decompose("unitary"), Circuit, to_backend=to_backend)  # noqa: E731
    for lay in L_left:
        if is_measure_layer(lay):
            continue
        mpo = US.apply_circuit(mpo, q2c(lay.inverse()), side="right",
                               max_bond=args.max_bond_final, cutoff=args.cutoff_final)
    for lay in L_right:
        if is_measure_layer(lay):
            continue
        mpo = US.apply_circuit(mpo, q2c(lay), side="left",
                               max_bond=args.max_bond_final, cutoff=args.cutoff_final)

    # --- MPO -> state: project the input leg to |0...0> ----------------------
    try:
        lower = [mpo.lower_ind(i) for i in range(n)]
    except Exception:
        per = [i for i in mpo.tensors[0].inds if str(i).startswith("b")]
        lower = per if len(per) == n else [i for i in mpo.tensors[0].inds if str(i).startswith("k")]
    tn = mpo.isel({i: 0 for i in lower})

    # Move arrays to host BEFORE the conversion, and do NOT run apply_to_arrays
    # afterwards: that returns a bare TensorNetwork and silently drops the
    # MatrixProductState methods (site_ind / normalize / copy) that the
    # winner's amp2 needs.  probe_conv2.py pinned this down.
    try:
        import torch

        for _t in tn.tensors:
            if isinstance(_t.data, torch.Tensor):
                _t.modify(data=_t.data.detach().cpu().numpy())
    except ImportError:
        pass

    mps = qtn.MatrixProductState.from_TN(tn, site_ind_id="k{}", cyclic=False)
    print(f"# state object: {type(mps).__name__} copy={hasattr(mps, 'copy')} "
          f"site_ind={hasattr(mps, 'site_ind')} normalize={hasattr(mps, 'normalize')}",
          flush=True)
    try:
        nrm = float(abs(complex(mps.norm())))
    except Exception as _e:
        nrm = float("nan")
        print(f"# (norm check skipped: {type(_e).__name__})")

    # Bond profile is DIAGNOSTIC ONLY -- three fallbacks and then give up.
    # A cosmetic line must never be able to abort a 60-minute run.
    bonds = []
    try:
        bonds = [int(mps.bond_size(i, i + 1)) for i in range(len(mps.sites) - 1)]
    except Exception:
        try:
            ts = list(mps.tensors)

            def _sh(a, b):
                c = set(a.inds) & set(b.inds)
                return int(a.ind_size(next(iter(c)))) if c else 1

            bonds = [_sh(ts[i], ts[i + 1]) for i in range(len(ts) - 1)]
        except Exception:
            pass
    bmax = max(bonds) if bonds else -1
    print(f"# state: bond max={bmax} (n={len(bonds)}) norm={nrm:.6f} "
          f"({'OK, unitary => 1.0' if (nrm == nrm and abs(nrm - 1) < 1e-3) else 'norm n/a'})",
          flush=True)

    # --- ★ THE DIAGNOSTIC ★ (print BEFORE any save, so a save failure can
    # never cost us the run's verdict) -------------------------------------
    if args.truth:
        truth_site = "".join(args.truth[i] for i in _PERM)  # logical -> site order
        a2 = EX.amp2(mps, truth_site)
        a2_rev = EX.amp2(mps, truth_site[::-1])
        print(f"\n### amp2(truth) = {a2:.6e}   (reversed site order: {a2_rev:.6e})")
        print(f"### uniform = {2.0**-n:.6e}  ->  truth is {a2/2.0**-n:.3e}x uniform")
        print(f"### VERDICT: {'BURIED (recoverable)' if a2 > 1e-8 else 'DISCARDED (eps-level)'}"
              f"   [perm={_PERM[:8]}...]", flush=True)

    # beam_search(mps, beam, k) -> [(bitstring_site_order, prob), ...]  (one list)
    try:
        res = EX.beam_search(mps, beam=args.beam, k=args.topk)
        if isinstance(res, list) and res and isinstance(res[0], (tuple, list)) and len(res[0]) == 2:
            cand, w = [b for b, _p in res], [p for _b, p in res]
        elif isinstance(res, tuple) and len(res) == 2:
            cand, w = list(res[0]), list(res[1])
        else:  # unknown shape: keep whatever strings we can see
            cand, w = res, []
    except Exception as _e:
        cand, w = [], []
        print(f"# beam_search failed: {type(_e).__name__}: {_e}")
    cand = [cand] if isinstance(cand, str) else list(cand)
    print("\n### beam_search candidates (site order):")
    for i, c in enumerate(cand[:8]):
        ww = w[i] if hasattr(w, "__len__") else w
        line = f"  #{i+1} w={ww:.6e} bits={c}"
        if args.truth:
            logical = "".join(str(c)[_PERM.index(j)] for j in range(n)) if len(str(c)) == n else None
            if logical:
                h = sum(1 for a, b in zip(logical, args.truth) if a != b)
                hrev = sum(1 for a, b in zip(logical, args.truth[::-1]) if a != b)
                line += f"  H={h}/{n} (reversed-truth {hrev})"
        print(line, flush=True)

    print(f"\n# total {time.time()-t0:.0f}s", flush=True)

    if args.save_mps:
        try:
            import pickle

            with open(args.save_mps, "wb") as fh:
                pickle.dump(mps, fh, protocol=4)
            print(f"# saved MPS (pickle) to {args.save_mps}", flush=True)
        except Exception as _e:
            print(f"# save failed: {type(_e).__name__}: {_e}", flush=True)
            try:
                arrs = np.array([np.asarray(t.data) for t in mps.tensors], dtype=object)
                np.save(args.save_mps + ".arrays.npy", arrs, allow_pickle=True)
                print(f"# fallback: saved raw arrays x{len(arrs)}")
            except Exception as _e2:
                print(f"# fallback save also failed: {type(_e2).__name__}")


if __name__ == "__main__":
    main()
