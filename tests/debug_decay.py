"""Debug decaying-spectrum topk SVD: is Q (power-iter subspace) capturing
the true top-k LEFT subspace? Print residual, full recon, subspace cosine,
and the actual power-iter convergence."""
import numpy as np
src = open('hqp_solver.py').read()
ns = {}
exec(src[:src.index('def main(')], ns)
_svd = ns['_svd']; _svd_topk = ns['_svd_topk']
_svdrng = ns['_svd_rng']

rng = np.random.default_rng(7)
r = c = 256; k = 32
n = min(r, c)
Sv = np.array([0.2 ** i for i in range(n)])
Qr, _ = np.linalg.qr(rng.standard_normal((r, n)) + 1j * rng.standard_normal((r, n)))
Qc, _ = np.linalg.qr(rng.standard_normal((c, n)) + 1j * rng.standard_normal((c, n)))
M = Qr @ np.diag(Sv) @ Qc.conj().T
print(f"||M|| = {np.linalg.norm(M):.4f}")
print("top-5 Sv:", Sv[:5])

Uf, Sf, Vh_f = np.linalg.svd(M, full_matrices=False)
print("ref top-5 S:", Sf[:5])
Uref = Uf[:, :k]

# Run _svd_topk and also manually track Q.
res = _svd(M, k=k)
print("_svd returned:", None if res is None else ("U", res[0].shape, "S", res[1].shape, "Vh", res[2].shape))
if res is not None:
    U, S, Vh = res
    print("_svd top-5 S:", S[:5])
    # resid
    print("resid ||U S - M Vh^H|| =", np.linalg.norm(U * S[None, :] - M @ Vh.conj().T))
    print("full recon ||M - U S Vh|| =", np.linalg.norm(M - U @ np.diag(S) @ Vh))
    A = U.conj().T @ Uref
    cos = np.linalg.svd(A, compute_uv=False)
    print("subspace cos (should be ~1):", cos[:5], " min:", cos.min())
    # Are the returned U columns actual left singular vectors? Check M Vh_i ~ s_i U_i
    # i.e. U diag(S) Vh should reconstruct. Check each: U[:,i] should satisfy
    # M Vh[i,:]^H = S[i] U[:,i]
    chk = np.linalg.norm(M @ Vh[0, :].conj() - S[0] * U[:, 0])
    print("M v0^H vs S0*u0 resid:", chk)

# Manual power iteration to see convergence of the subspace.
ell = max(k, min(k + 4, n // 2))
Q = _svdrng.standard_normal((r, ell)); Q, _ = np.linalg.qr(Q)
for it in range(10):
    Qnew, _ = np.linalg.qr(M @ (M.conj().T @ Q))
    # measure angle between span(Q) and true top-k left subspace
    # project true top-k left subspace basis Uref onto Q
    # use the max principal angle
    A = Q.conj().T @ Uref   # ell x k
    cosA = np.linalg.svd(A, compute_uv=False)
    print(f"iter {it}: span(Q) vs top-k cos = {cosA[:4]}  min={cosA.min() if cosA.size else float('nan')}")
    Q = Qnew
