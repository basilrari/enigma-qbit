#!/usr/bin/env python3
"""Probe: convert the projected MPO into a real MPS, and settle bit order.

Runs on the box. Uses d0 (x q0; x q1; x q2 -> |00111>) as ground truth: exactly
one of amp2(mps,'00111') / amp2(mps,'11100') must be 1.0.
"""
import sys
import numpy as np

L2 = "/mnt/8tb_hdd2/basilrari/enigma-work/l2win"
sys.path.insert(0, L2)
sys.argv = ["x", "--qasm", "/mnt/8tb_hdd2/basilrari/enigma-work/verify/d0_s0_trivial.qasm",
            "--truth", "11100", "--max-bond", "256", "--budget", "200", "--beam", "32", "--topk", "4"]
import quimb.tensor as qtn                                    # noqa: E402
import l2_absorb as LA                                        # noqa: E402
import extract as EX                                          # noqa: E402

qc = LA.load_qasm("/mnt/8tb_hdd2/basilrari/enigma-work/verify/d0_s0_trivial.qasm")
uqc = LA.unitarize(qc)
n = uqc.num_qubits
LA._N, LA._PERM = n, list(range(n))
import unswap_stallfix as US                                  # noqa: E402
US.unswap = LA._tracking_unswap

mpo, L_left, L_right, stats = US.mpo_compress_unswap(
    uqc, max_bond=256, cutoff=1e-3, unswap_threshold=1e6, early_stopping_gates=30,
    center_ratio=0.5, to_backend=None, seed=0)
print("mpo type:", type(mpo).__name__)
print("core inds:", mpo.tensors[0].inds)
low = None
try:
    low = [mpo.lower_ind(i) for i in range(n)]
    print("lower_ind OK:", low)
except Exception as e:
    print("lower_ind FAILED:", e)
    low = [i for i in mpo.tensors[0].inds if str(i).startswith("b")]
proj = mpo.isel({i: 0 for i in low})
print("proj type:", type(proj).__name__, "| tensors:", [t.inds for t in proj.tensors][:3])

cands = {}
try:
    cands["MPS(tensors)"] = qtn.MatrixProductState([t.copy() for t in proj.tensors])
except Exception as e:
    print("MatrixProductState(tensors) FAILED:", repr(e)[:200])
try:
    cands["from_TN"] = qtn.MatrixProductState.from_TN(proj, site_ind_id="k{}")
except Exception as e:
    print("from_TN FAILED:", repr(e)[:200])
try:
    cands["view_as_"] = proj.view_as_(qtn.MatrixProductState, inplace=False)
except Exception as e:
    print("view_as_ FAILED:", repr(e)[:200])

for name, m in cands.items():
    try:
        a1 = EX.amp2(m, "00111")
        a2 = EX.amp2(m, "11100")
        print(f"  {name}: amp2('00111')={a1:.6f}  amp2('11100')={a2:.6f}  "
              f"-> {'q0-FIRST (site0=q0)' if a1 > 0.5 else ('q0-LAST' if a2 > 0.5 else 'neither?!')}")
    except Exception as e:
        print(f"  {name}: amp2 FAILED: {repr(e)[:160]}")
print("norm:", float(np.sqrt(abs(proj.norm() ** 2))))
