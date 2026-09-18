import sys
import traceback

import numpy as np
import quimb.tensor as qtn

sys.path.insert(0, "/mnt/8tb_hdd2/basilrari/enigma-work")
from exact_extract import as_mps, topk_exact  # noqa: E402

n, D = 6, 2
rng = np.random.default_rng(1)
arrs = []
for i in range(n):
    dl = min(D, 2 ** i, 2 ** (n - i))
    dr = min(D, 2 ** (i + 1), 2 ** (n - 1 - i))
    arrs.append(rng.normal(size=(dl, 2, dr)))
print("shapes:", [a.shape for a in arrs])

mps = qtn.MatrixProductState([qtn.Tensor(a, inds=(f"b{i}", f"k{i}", f"b{i+1}")) for i, a in enumerate(arrs)],
                            site_ind_id="k{}")
print("type(mps)      :", type(mps).__name__)
print("type(t[0])     :", type(mps.tensors[0]).__name__)
print("type(t[0].data):", type(mps.tensors[0].data).__name__)
print("t[0].inds      :", [str(x) for x in mps.tensors[0].inds])
print("site_ind(0)    :", end=" ")
try:
    print(mps.site_ind(0))
except Exception:
    print("FAILS:", traceback.format_exc().strip().splitlines()[-1])

print("-- as_mps --")
m2 = as_mps(mps)
print("type:", type(m2).__name__)
try:
    m2 = m2.copy()
    m2.right_canonicalize()
    print("right_canonicalize OK")
except Exception:
    print("right_canonicalize FAILS:")
    traceback.print_exc()

try:
    print("topk:", topk_exact(mps, k=2, cap=10000, verbose=True, canonicalize=False))
except Exception:
    print("topk_exact FAILS:")
    traceback.print_exc()
