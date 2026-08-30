"""Decisive test for _svd top-k. Separates three regimes cleanly:

A) WELL-CONDITIONED FULL-RANK (slow decay, s_k >> noise): top-k subspace is
   mathematically well-defined -> assert subspace agreement + singular values.
B) FAST DECAY (0.2^i, effective rank << k): vectors beyond the effective rank
   are at the noise floor and their DIRECTION is arbitrary -> assert no NaN +
   singular values only (NOT subspace, to avoid asserting on arbitrary vectors).
C) RANDOM full-rank: well-separated -> assert subspace + singular values.
D) RANK-DEFICIENT with k = true rank: top-k == full row space (unique) -> assert
   subspace + singular values.

Metric: cosines of principal angles between two orthonormal subspaces =
singular values of (U^H V). All ~1 iff same subspace.
"""
import numpy as np
src = open('hqp_solver.py').read()
ns = {}
exec(src[:src.index('def main(')], ns)
_svd = ns['_svd']

rng = np.random.default_rng(7)
NOISE_FLOOR = 1e-10  # relative: below this a singular value is numerically noise

def principal_cos(U, V):
    """Cosines of principal angles between orthonormal subspaces span(U), span(V)."""
    return np.linalg.svd(U.conj().T @ V, compute_uv=False)

def well_separated_rank(Sref_desc, s0):
    """Largest m with Sref[m-1] > NOISE_FLOOR * s0 (well above the noise floor)."""
    m = 0
    for i, s in enumerate(Sref_desc):
        if s > NOISE_FLOOR * s0:
            m = i + 1
        else:
            break
    return m

def check(name, M, k, assert_sub=True, verbose=True):
    Uref, Sref, Vhref = np.linalg.svd(M, full_matrices=False)
    Sref_desc = np.sort(np.abs(Sref))[::-1]
    res = _svd(M, k=k)
    assert res is not None, f"{name}: unexpected None (exact fallback)"
    U, S, Vh = res
    assert np.all(np.isfinite(U)) and np.all(np.isfinite(S)) and np.all(np.isfinite(Vh)), \
        f"{name}: non-finite result"
    S = np.sort(np.abs(S))[::-1]
    kcmp = min(k, len(S), len(Sref_desc))
    # singular values of the well-separated part
    m = min(kcmp, well_separated_rank(Sref_desc, Sref_desc[0]))
    s_err = np.abs(S[:m] - Sref_desc[:m]).max() / Sref_desc[0]
    # subspace on the well-separated part only (arbitrary beyond noise)
    sub_err = None
    if assert_sub and m >= 1:
        cos = principal_cos(U[:, :m], Uref[:, :m])
        sub_err = 1.0 - cos.min()
    if verbose:
        print(f"  {name:26s} k={k:4d} rank_eff={m:3d}: s_err={s_err:.2e}"
              + (f"  sub_err={sub_err:.2e}" if sub_err is not None else "  (sub skipped)"))
    return s_err, sub_err

fail = 0
print("== A: well-conditioned full-rank decay (s=exp(-0.02 i)) ==")
for (r, c, k) in [(256, 256, 64), (512, 512, 128), (128, 64, 32), (64, 128, 16)]:
    n = min(r, c)
    S = np.exp(-0.02 * np.arange(n, dtype=float))   # s_{n-1}=exp(-0.02*255)~6e-3 >> noise
    Qr, _ = np.linalg.qr(rng.standard_normal((r, n)))
    Qc, _ = np.linalg.qr(rng.standard_normal((c, n)))
    M = Qr @ np.diag(S) @ Qc.conj().T
    s_err, sub_err = check(f"cond_decay {r}x{c}", M, k, assert_sub=True)
    if s_err > 1e-9 or (sub_err is not None and sub_err > 1e-7):
        fail += 1

print("== B: fast decay (0.2^i), k > effective rank (singular values only) ==")
for (r, c, k) in [(256, 256, 32), (256, 256, 64), (512, 512, 128), (512, 512, 256)]:
    n = min(r, c)
    S = 0.2 ** np.arange(n, dtype=float)
    Qr, _ = np.linalg.qr(rng.standard_normal((r, n)))
    Qc, _ = np.linalg.qr(rng.standard_normal((c, n)))
    M = Qr @ np.diag(S) @ Qc.conj().T
    s_err, _ = check(f"fast_decay {r}x{c}", M, k, assert_sub=False)
    if s_err > 1e-9:
        fail += 1

print("== C: random full-rank (well-separated) ==")
for (r, c, k) in [(256, 256, 32), (512, 512, 128), (128, 64, 48), (64, 128, 24)]:
    M = (rng.standard_normal((r, c)) + 1j * rng.standard_normal((r, c))) / 2
    s_err, sub_err = check(f"random {r}x{c}", M, k, assert_sub=True)
    if s_err > 1e-9 or (sub_err is not None and sub_err > 1e-7):
        fail += 1

print("== D: rank-deficient, k = true rank (full row space, unique) ==")
for (r, c, k) in [(256, 256, 64), (512, 512, 128)]:
    rank = k
    S = rng.standard_normal(rank) ** 2 + 0.5
    S = S / S[0]  # normalize
    Qr, _ = np.linalg.qr(rng.standard_normal((r, rank)))
    Qc, _ = np.linalg.qr(rng.standard_normal((c, rank)))
    M = Qr @ np.diag(S) @ Qc.conj().T
    s_err, sub_err = check(f"rankdef {r}x{c}", M, k, assert_sub=True)
    if s_err > 1e-9 or (sub_err is not None and sub_err > 1e-7):
        fail += 1

print("\n" + ("ALL SVD TOPK CHECKS PASSED" if fail == 0 else f"*** {fail} FAILURES ***"))

# --- MPS oracle rigor (statevector) ---
print("\n== MPS oracle rigor ==")
import importlib.util, sys
from qiskit import QuantumCircuit
from qiskit.circuit.library import UGate
from qiskit.quantum_info import Statevector
MPS = ns['MPS']; build_mps = ns['build_mps']
rng = np.random.default_rng(12345)
worst = 0.0
def sidx(s, n): return sum(int(s[q]) * (2 ** q) for q in range(n))
for trial in range(8):
    n = 10
    qc = QuantumCircuit(n)
    for _ in range(12):
        qc.append(UGate(*rng.uniform(0, 2 * np.pi, 3)), [int(rng.integers(0, n))])
    for _ in range(8):
        a, b = rng.choice(n, 2, replace=False)
        qc.cz(int(a), int(b))
    sv = Statevector(qc); probs = np.abs(sv) ** 2
    T, order, qo = build_mps(qc, 16, verbose=False)
    m = MPS(T, qo, None, f"t{trial}")
    qo = list(qo)
    def mk(seed):
        s = ['0'] * n
        for k2 in range(n): s[qo[k2]] = str(seed[k2])
        return ''.join(s)
    for r2 in range(6):
        s = mk(list(rng.integers(0, 2, n)))
        worst = max(worst, abs(float(probs[sidx(s, n)]) - m.P(s)))
    seed0 = list(rng.integers(0, 2, n)); s0 = mk(seed0)
    fl = m.P_single_flip_all(s0)
    for k2 in range(n):
        fs = list(seed0); fs[k2] = 1 - fs[k2]
        worst = max(worst, abs(float(probs[sidx(mk(fs), n)]) - fl[k2]))
    p1 = m.marginal_p1()
    for k2 in range(n):
        q = qo[k2]
        worst = max(worst, abs(float(sum(probs[x] for x in range(2 ** n) if (x >> q) & 1)) - p1[k2]))
print(f"RIGOR worst = {worst:.2e}")
if worst > 1e-8:
    fail += 1

print("\n" + ("ALL OK (SVD + ORACLE)" if fail == 0 else f"*** {fail} FAILURES ***"))
sys.exit(1 if fail else 0)
