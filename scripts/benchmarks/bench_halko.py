import numpy as np, time, hqp_solver as H
rng = np.random.default_rng(2)
print("case        fullSVD   halko(k)   ratio   dS        minsubcos")
for (L, C, k) in [(1024, 1024, 512), (1024, 1024, 256), (512, 512, 256), (768, 1024, 384)]:
    m = min(L, C)
    s = 10.0 * np.exp(-0.02 * np.arange(m, dtype=float)); s = s / np.linalg.norm(s)
    U0, _ = np.linalg.qr(rng.standard_normal((L, m)))
    V0, _ = np.linalg.qr(rng.standard_normal((C, m)))
    M = (U0 * np.sqrt(s)) @ V0.T
    t0 = time.time(); Uf, Sf, Vhf = np.linalg.svd(M, full_matrices=False); tf = time.time() - t0
    # force the Halko top-k path directly (bypass the _svd shortcut)
    t0 = time.time(); Ut, St, Vht = H._svd_topk(M, k); tt = time.time() - t0
    if Ut is None:
        print(f"{L}x{C} k={k}:  Halko returned None (fallback)")
        continue
    ds = np.abs(St[:k] - Sf[:k]).max()
    cos = np.linalg.svd(Ut[:, :k].conj().T @ Uf[:, :k])[1]
    print(f"{L}x{C} k={k:<4} {tf*1000:8.0f}ms {tt*1000:8.0f}ms {tt/tf:6.2f}  {ds:.1e}     {cos.min():.4f}")
print("done")
