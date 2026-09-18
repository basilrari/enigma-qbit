import pickle
import sys

import numpy as np
import quimb.tensor as qtn

sys.path.insert(0, "/mnt/8tb_hdd2/basilrari/enigma-work")
sys.path.insert(0, "/mnt/8tb_hdd2/basilrari/enigma-work/l2win")
from exact_extract import as_mps, topk_exact  # noqa: E402

W = "/mnt/8tb_hdd2/basilrari/enigma-work"

print("=== 1) known-answer MPS (squeezed bond-1 tensors, all-ones truth) ===")
mps = as_mps(pickle.load(open(f"{W}/mps/known_12q_known5.npy", "rb")))
res = topk_exact(mps, k=3, cap=20000, verbose=True)
for b, p in res:
    print(f"   {b}  p={p:.6e}")
ok1 = bool(res) and res[0][0] == "1" * 12 and abs(res[0][1] - 1.0) < 1e-9
print("   PASS" if ok1 else "   FAIL")


def dense_of(arrs):
    """Pure-numpy <-> full amplitude vector, so the test never trusts quimb."""
    t = arrs[0]
    for a in arrs[1:]:
        t = np.tensordot(t, a, axes=([-1], [0]))
    return t.reshape(-1)


print("\n=== 2) random bond-4 MPS: exact B&B vs numpy brute force ===")
n, D = 12, 4
rng = np.random.default_rng(7)
arrs = []
for i in range(n):
    # bond between sites i-1 and i is min(D, 2^i, 2^(n-i)); must taper at BOTH ends
    dl = min(D, 2 ** i, 2 ** (n - i))
    dr = min(D, 2 ** (i + 1), 2 ** (n - 1 - i))
    arrs.append(rng.normal(size=(dl, 2, dr)) + 1j * rng.normal(size=(dl, 2, dr)))
v = dense_of(arrs)
v = v / np.linalg.norm(v)
arrs[0] = arrs[0] / np.linalg.norm(dense_of(arrs))  # renormalise through the MPS itself

mps2 = as_mps(qtn.MatrixProductState(arrs, site_ind_id="k{}"))  # raw arrays, not Tensors
print("   mps sites:", len(mps2.tensors), "bond sizes:", [int(t.shape[0]) for t in mps2.tensors[1:4]])

probs = np.abs(dense_of(arrs)) ** 2
probs = probs / probs.sum()
order = np.argsort(-probs)[:3]
bf = [("".join(str((int(i) >> (n - 1 - k)) & 1) for k in range(n)), float(probs[i])) for i in order]
res2 = topk_exact(mps2, k=3, cap=300000, verbose=True)
print("   brute:", [(b, f"{p:.8f}") for b, p in bf])
print("   exact:", [(b, f"{p:.8f}") for b, p in res2])
ok2 = len(res2) == 3 and all(res2[j][0] == bf[j][0] for j in range(3))
dl = max(abs(res2[j][1] - bf[j][1]) for j in range(3)) if len(res2) == 3 else 9.9
print(f"   bits match: {ok2}   max |dp| = {dl:.3e}")
print("\nRESULT", "PASS" if (ok1 and ok2 and dl < 1e-6) else "FAIL")
