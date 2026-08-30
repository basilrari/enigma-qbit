#!/bin/bash
# DECISIVE: real Halko top-k path (bypassing the full-SVD shortcut) vs full
# economy SVD, on a realistic decaying 1024x1024 complex bond matrix.
cd /mnt/8tb_hdd2/basilrari/enigma-work/verify || exit 1
PY=/mnt/8tb_hdd2/basilrari/enigma-work/l2venv/bin/python
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
OPENBLAS_MAX_THREADS=4 OMP_DYNAMIC=FALSE "$PY" - <<'EOF'
import numpy as np, time, hqp_solver as H
rng = np.random.default_rng(7)
chi = 512; n = 2*chi
s = np.exp(-0.05*np.arange(n))*10.0
U, _ = np.linalg.qr(rng.standard_normal((n,n))+1j*rng.standard_normal((n,n)))
V, _ = np.linalg.qr(rng.standard_normal((n,n))+1j*rng.standard_normal((n,n)))
M = (U*s) @ V.conj().T
np.linalg.svd(M, full_matrices=False); H._svd(M, k=chi)  # warm

def halko(M, kk, passes=15, seed=3):
    """Inline Halko top-k, no shortcut, to measure the real path."""
    r, c = M.shape
    rng = np.random.default_rng(seed)
    ell = min(kk+8, min(r,c))
    Q = rng.standard_normal((r, ell)); Q, _ = np.linalg.qr(Q)
    for _ in range(passes):
        Q = M @ (M.conj().T @ Q)
        Q, _ = np.linalg.qr(Q)
    B = Q.conj().T @ M
    Ub, S, Vh = np.linalg.svd(B, full_matrices=False)
    Uk = Q @ Ub[:, :kk]
    return Uk, S[:kk], Vh[:kk, :]

def bench(f, reps=3):
    ts=[]
    for _ in range(reps):
        t0=time.time(); f(); ts.append(time.time()-t0)
    return min(ts)*1000

tf = bench(lambda: np.linalg.svd(M, full_matrices=False))
th = bench(lambda: halko(M, chi))
Uk, Sk, Vhk = halko(M, chi)
Uf, Sf, Vf = np.linalg.svd(M, full_matrices=False)
serr = np.abs(Sk - Sf[:chi]).max()
cos  = np.min(np.abs(Uk.conj().T @ Uf[:, :chi]))
print(f"fullSVD(1024) = {tf:.0f} ms")
print(f"halko top-512 = {th:.0f} ms   (speedup {tf/th:.2f}x)")
print(f"topk s_err={serr:.2e}  min_subspace_cos={cos:.6f}")
print("DONE")
EOF
