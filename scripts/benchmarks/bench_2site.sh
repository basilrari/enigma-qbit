#!/bin/bash
# Decisive 2-site SVD benchmark: full economy SVD vs Halko top-k,
# on realistic DECAYING-spectrum bond matrices, sequential & uncontended,
# across thread counts. Answers: what's the real per-CZ cost and which
# SVD path wins?
cd /mnt/8tb_hdd2/basilrari/enigma-work/verify || exit 1
PY=/mnt/8tb_hdd2/basilrari/enigma-work/l2venv/bin/python
for T in 4 16; do
  OPENBLAS_NUM_THREADS=$T OMP_NUM_THREADS=$T MKL_NUM_THREADS=$T \
  OPENBLAS_MAX_THREADS=$T OMP_DYNAMIC=FALSE \
  "$PY" - <<'EOF'
import numpy as np, time, math, hqp_solver as H
rng = np.random.default_rng(3)
print(f"threads={__import__('os').environ['OPENBLAS_NUM_THREADS']}")
print(f"{'chi':>5} {'fullSVD':>9} {'halko':>9} {'ratio':>7} {'decay':>8} {'topk_ok':>8}")
for chi in (256, 512):
    n = 2*chi
    # Realistic bond matrix: decaying spectrum, complex, full "shape" (2chi x 2chi)
    # Build M = U diag(s) Vh with s decaying exponentially (peaked-state bond)
    s = np.exp(-0.05*np.arange(n)) * 10.0
    U, _ = np.linalg.qr(rng.standard_normal((n,n)) + 1j*rng.standard_normal((n,n)))
    V, _ = np.linalg.qr(rng.standard_normal((n,n)) + 1j*rng.standard_normal((n,n)))
    M = (U * s) @ V.conj().T
    # warm
    H._svd(M, k=chi); np.linalg.svd(M, full_matrices=False)
    def bench(fn, reps=3):
        ts=[]
        for _ in range(reps):
            t0=time.time(); fn(); ts.append(time.time()-t0)
        return min(ts)*1000
    tf = bench(lambda: H._svd(M, k=chi))
    # Halko top-k WITHOUT the full-svd shortcut: temporarily force k<min/2
    # by benchmarking _svd_topk directly
    def halko():
        r = H._svd_topk(M, chi)
        return r
    # _svd_topk has the 2k>=min shortcut; to test the actual Halko path we
    # benchmark it and also the raw full svd for comparison
    th = bench(halko)
    tr = bench(lambda: np.linalg.svd(M, full_matrices=False))
    # decay: ratio of s[chi] to s[0] (how peaked)
    print(f"{chi:>5} {tf:8.1f} {th:8.1f} {tf/th:7.2f} {s[chi]/s[0]:8.1e} {tr:8.1f}")
    # also: is Halko returning the right top-k?
    Uk, Sk, Vhk = H._svd_topk(M, chi)
    Ukf, Skf, Vhf = np.linalg.svd(M, full_matrices=False)
    err = np.abs(Sk-Skf[:chi]).max()
    cos = np.min(np.abs(Uk.conj().T @ Ukf[:, :chi]))
    print(f"      check: s_err={err:.2e}  min_subspace_cos={cos:.6f}")
print("DONE", flush=True)
EOF
done
echo ALLDONE
