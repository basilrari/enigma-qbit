import sys

import numpy as np
import quimb.tensor as qtn

sys.path.insert(0, "/mnt/8tb_hdd2/basilrari/enigma-work")
from exact_extract import _site_data  # noqa: E402

n, D = 12, 4
rng = np.random.default_rng(1)
arrs = []
for i in range(n):
    dl = min(D, 2 ** i, 2 ** (n - i))
    dr = min(D, 2 ** (i + 1), 2 ** (n - 1 - i))
    arrs.append(rng.normal(size=(dl, 2, dr)))
print("intended dl/dr:", [(min(D, 2 ** i, 2 ** (n - i)), min(D, 2 ** (i + 1), 2 ** (n - 1 - i))) for i in range(n)])
mps = qtn.MatrixProductState(arrs, site_ind_id="k{}")

# which physical index does each tensor actually carry?
print("\ntensor -> physical index it carries (by quimb site_ind):")
for i in range(n):
    ix = str(mps.site_ind(i))
    tb = [t for t in mps.tensors if ix in [str(x) for x in t.inds]]
    t = tb[0] if tb else None
    nm = [str(x) for x in t.inds] if t else []
    sz = [int(t.ind_size(x)) for x in t.inds] if t else []
    print(f"  site {i}: tens#{mps.tensors.index(t)}  inds={nm} sizes={sz}")

A, nm, phys, bins, bouts = _site_data(mps)
print("\ni  phys  canonical shape   bins(/size)            bouts(/size)")
for i in range(n):
    def sz(x):
        return "-" if x is None else f"{x}({A[i].shape[nm[i].index(x)]})"
    print(f"{i}  {phys[i]:5s} {str(A[i].shape):16s} {sz(bins[i]):22s} {sz(bouts[i])}")

v = np.ones(1, dtype=complex)
print("\nwalk:")
for i in range(n):
    va = v.reshape(-1)
    print(f"  site {i}: v.size={va.size}  tensor={A[i].shape}  in={bins[i]}")
    if bins[i] is not None and va.size != A[i].shape[0]:
        print("   ^^ MISMATCH -- stopping here")
        break
    from exact_extract import _step

    v = _step(A, nm, phys, bins, bouts, i, v, "0")
