import numpy as np, time, hqp_solver as H
rng = np.random.default_rng(1)
print("shape      k    fullSVD   _svd(k)   ratio   dS(topk)   minsubcos")
for (L, C) in [(512, 512), (1024, 1024), (768, 1024)]:
    k = L // 2
    m = min(L, C)
    s = 10.0 * np.exp(-0.05 * np.arange(m, dtype=float)); s = s / np.linalg.norm(s)
    U0, _ = np.linalg.qr(rng.standard_normal((L, m)))
    V0, _ = np.linalg.qr(rng.standard_normal((C, m)))
    M = (U0 * np.sqrt(s)) @ V0.T
    t0 = time.time(); Uf, Sf, Vhf = np.linalg.svd(M, full_matrices=False); tf = time.time() - t0
    t0 = time.time(); Ut, St, Vht = H._svd(M, k=k); tt = time.time() - t0
    # _svd may return full or top-k spectrum; compare the top-k slice
    ds = np.abs(St[:k] - Sf[:k]).max()
    cos = np.linalg.svd(Ut[:, :k].conj().T @ Uf[:, :k])[1]
    print(f"{L}x{C}  k={k:<3} {tf*1000:8.0f}ms {tt*1000:8.0f}ms {tt/tf:6.2f}  {ds:.1e}     {cos.min():.4f}")
print("benchmark done")
