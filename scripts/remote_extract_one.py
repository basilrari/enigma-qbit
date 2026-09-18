#!/usr/bin/env python3
"""Certified extraction from one saved absorbed MPS.

Runs ON THE BOX (needs quimb + the engine). Usage:
    python remote_extract_one.py <mps.npy> <truth> [k] [cap]

The dossier's warning: the winners' beam search (beam 512, k=8) can prune the
true peak at depth 10-20. topk_exact is a certified branch-and-bound -- it
returns the TRUE top-k of the absorbed state or nothing, never a wrong claim.

Prints a single parseable line:
    RESULT bits=... p=... h=... nodes=... cert=<yes|no>
plus the top-k list.
"""
import sys

import numpy as np
import quimb.tensor as qtn

sys.path.insert(0, "/mnt/8tb_hdd2/basilrari/enigma-work")
from exact_extract import as_mps, topk_exact


def main():
    path, truth = sys.argv[1], sys.argv[2]
    k = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    cap = int(sys.argv[4]) if len(sys.argv) > 4 else 400000

    # the orchestrator pickles the MPS; fall back to an object-array .npy
    try:
        import pickle

        with open(path, "rb") as fh:
            obj = pickle.load(fh)
        mps = obj if hasattr(obj, "tensors") else qtn.MatrixProductState(list(obj), site_ind_id="k{}")
    except Exception as _pe:
        print(f"# (not a pickle: {type(_pe).__name__})")
        tens = np.load(path, allow_pickle=True)
        arrs = [np.asarray(t) for t in tens] if tens.dtype == object else list(tens)
        mps = qtn.MatrixProductState(arrs, site_ind_id="k{}")
    mps = as_mps(mps)
    try:
        n = int(mps.L)
    except Exception:
        n = len(mps.tensors)
    print(f"# sites={n} truth_len={len(truth)}")

    res = topk_exact(mps, k=k, cap=cap, verbose=False)
    tru = np.zeros(n, dtype=int)
    for i, ch in enumerate(truth[:n]):
        tru[i] = int(ch)
    import extract as _EX

    p_tru = _EX.amp2(mps, "".join(str(int(x)) for x in tru))
    h_tru = int(np.sum(np.array([int(c) for c in res[0][0]]) != tru)) if res else -1
    print(f"# amp2(truth)={p_tru:.6e}  h(top1,truth)={h_tru}")
    for bits, p in res:
        b = np.array([int(c) for c in bits])
        h = int(np.sum(b != tru))
        print(f"TOP bits={bits} p={p:.6e} h={h}")
    if res:
        b, p = res[0]
        print(f"RESULT bits={b} p={p:.6e} h={h_tru} cert=yes")
    else:
        print("RESULT cert=no")


if __name__ == "__main__":
    main()
