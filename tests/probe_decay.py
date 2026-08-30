import numpy as np
import hqp_solver as H

rng = np.random.default_rng(0)
r, c = 256, 256
s = 10.0 * np.exp(-0.1 * np.arange(r, dtype=float))
# True decaying spectrum: M = U0 diag(s) V0^T with random orthogonal U0, V0
U0, _ = np.linalg.qr(rng.standard_normal((r, r)))
V0, _ = np.linalg.qr(rng.standard_normal((c, c)))
M = (U0 * s) @ V0.T

k = 32
U, S, Vh = H._svd_topk(M, k)
Uref, Sref, Vhref = np.linalg.svd(M, full_matrices=False)

print("topk  U", U.shape, "S", S.shape, "Vh", Vh.shape)
print("S[:k]        :", S[:k])
print("Sref[:k]     :", Sref[:k])
print("recon ||M - U S Vh|| =", np.linalg.norm(M - (U * S) @ Vh))
print("recon ||M - Uref Sref Vhref|| =", np.linalg.norm(M - (Uref * Sref) @ Vhref))
print("||M|| =", np.linalg.norm(M))
print("rank(M) =", np.linalg.matrix_rank(M))
print("rank(U) =", np.linalg.matrix_rank(U))
print("rank(Vh) =", np.linalg.matrix_rank(Vh))
print("singular values of M (top 6):", Sref[:6])
print("s[k-1] vs s[k]:", Sref[k-1], Sref[k])

# subspace agreement
sv = np.linalg.svd(U.conj().T @ Uref[:, :k], compute_uv=False)
print("subspace sv:", sv)

# is U an orthonormal basis?
print("U^H U = I?", np.linalg.norm(U.conj().T @ U - np.eye(k)))
# does U span the top-k right vectors? (V side)
svV = np.linalg.svd(Vh.conj().T @ Vhref[:k, :], compute_uv=False)
print("V-subspace sv:", svV)
