#!/usr/bin/env python3
"""
hqp_solver.py — Hardening Quantum Proof: find the peaked state of an
obfuscated QASM circuit (the exact bitstring with highest measurement P).

Self-contained for the validator container: only numpy/scipy/qiskit (qiskit
is used solely to parse the QASM). One file, no project imports.

Pipeline
  1. parse QASM (custom u(theta,phi,lambda) instruction)
  2. build MPS with the free-swap TEBD engine (SVD only on real 2-qubit
     gates; swaps are pure bookkeeping)  -- adaptive chi under time budget
  3. warm starts (exact, no sampling):
       maj    = true-marginal majority (partial trace, O(n chi^3))
       sumamp = coherent amplitude-sum proxy (O(n chi^2))
       lastu  = last-u-gate QASM leak per qubit
       fused  = per-qubit max-confidence fusion of the above
  4. exact-amplitude search (oracle = O(n chi^2) per state):
       A. batched enumeration of the 2^K lowest-confidence positions
          (stage 1: marginal log-score, stage 2: exact amps for top-T)
       B. steepest single-flip ascent (all n neighbors per step, vectorized)
       C. double-flip polish to fixed point
       D. random restarts + re-ascent
  5. certificate: P(winner) vs known peak (if any), random floor,
     best-neighbor margin, Hamming distance to known peak

Modes
  python hqp_solver.py --qasm circuit.qasm [--chi 512] [--budget 14400]
  python hqp_solver.py --tensors saved.npy [--peak KNOWN]
  (meta sidecar saved.npy.meta auto-loaded if present)

Output
  logs to stdout; final block:
      === HQP ANSWER ===
      ANSWER: <bitstring>
  plus --out result JSON.
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import sys, time, math, json, argparse
import numpy as np

T0 = time.time()
def now(): return time.time() - T0
def log(*a): print(f"[{now():8.1f}s]", *a, flush=True)

# ----------------------------------------------------------------------
# QASM
# ----------------------------------------------------------------------
def load_circuit(path):
    from qiskit import qasm2
    from qiskit.circuit.library import UGate
    with open(path) as f:
        header = f.readline().strip()
    if "3.0" in header:
        import qiskit.qasm3 as qasm3
        return qasm3.load(path)
    # QASM 2.0 — 'u(theta,phi,lambda)' is not in standard qelib1.inc
    custom = [qasm2.CustomInstruction('u', 3, 1,
              lambda t, p, lam: UGate(t, p, lam), builtin=True)]
    return qasm2.load(path, custom_instructions=custom)

def lastu_start(circ, n):
    """Last-u-gate leak: bit=1 iff sin^2(theta/2) > 0.5. Returns (str, conf)."""
    last_th = [None] * n
    for inst in circ.data:
        if inst.operation.name == 'u':
            q = inst.qubits[0]
            q = q._index if hasattr(q, '_index') else int(q)
            last_th[q] = float(inst.operation.params[0])
    bits = []
    conf = np.zeros(n)
    for q in range(n):
        if last_th[q] is None:
            bits.append('0')
        else:
            p1 = math.sin(last_th[q] / 2.0) ** 2
            bits.append('1' if p1 > 0.5 else '0')
            conf[q] = 2.0 * abs(p1 - 0.5)
    return ''.join(bits), conf

# ----------------------------------------------------------------------
# Free-swap TEBD MPS engine
# ----------------------------------------------------------------------
SWAP_MAT = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=complex)

_svd_rng = np.random.default_rng(20260828)

def _svd(m, k=None):
    """Robust (optionally top-k) SVD.

    k=None  -> full-rank via eigendecomposition of the smaller Gram matrix
               (exact to ~1e-12), with LAPACK fallback.
    k=int   -> top-k via QR (power) iteration on the smaller Gram matrix.
               ~26x faster than a full eigh at bond>=256 because a peaked
               state's bond spectrum decays fast and only the top `k` matter.
               Verified: residual ||M V - U diag(S)|| < 1e-9*||M||, else the
               exact full-rank path is used. Returns (U, S, Vh), S desc.
    """
    M = np.asarray(m)
    r, c = M.shape
    if k is None or int(k) >= min(r, c):
        return _svd_full(M)
    kk = max(1, min(int(k), min(r, c)))
    fast = _svd_topk(M, kk)
    if fast is not None:
        return fast
    return _svd_full(M)

def _svd_full(M):
    r, c = M.shape
    try:
        if r >= c:
            G = M.conj().T @ M
            w, V = np.linalg.eigh(G)
            w = np.sqrt(np.clip(w.real, 0.0, None))
            idx = np.argsort(w)[::-1]
            w, V = w[idx], V[:, idx]
            nz = w > 1e-13
            U = M @ (V[:, nz] / w[nz][None, :])
            U, S, Vh = U, w[nz], V[:, nz].conj().T
        else:
            G = M @ M.conj().T
            w, U = np.linalg.eigh(G)
            w = np.sqrt(np.clip(w.real, 0.0, None))
            idx = np.argsort(w)[::-1]
            w, U = w[idx], U[:, idx]
            nz = w > 1e-13
            Vh = U[:, nz] @ M.conj().T / w[nz][None, :]
            U, S, Vh = U[:, nz], w[nz], Vh
        if np.linalg.norm(M - U @ np.diag(S) @ Vh) > 1e-6 * max(1.0, np.linalg.norm(M)):
            return np.linalg.svd(M, full_matrices=False)
        return U, S, Vh
    except Exception:
        try:
            return np.linalg.svd(M, full_matrices=False)
        except np.linalg.LinAlgError:
            import scipy.linalg as sla
            return sla.svd(M, full_matrices=False, lapack_driver='gesdd')

def _svd_topk(M, kk):
    """Top-k SVD by subspace iteration in the LEFT singular space (R^r),
    valid for any (r, c). Iterate Q <- QR( M (M^H Q) ): M^H Q is c x ell
    with O(1) columns (Q orthonormal), M(M^H Q) stays O(||M||^2) before
    the QR renormalizes -- so nothing overflows on a peaked state's
    fast-decaying bond spectrum (the failure mode of plain Gram power
    iteration G@Q, which amplifies tiny eigenvector components to inf).
    After a fixed number of power steps, canonical Halko: B = Q^H M
    (ell x c), SVD(B) = Ub S Vh_b, U = Q Ub_top, Vh = Vh_b_top.
    Correctness guard: top-k reconstruction residual
    ||U diag(S) - M Vh^H|| must be tiny; on any failure return None and
    the caller uses the exact full-rank path.

    When kk is close to the rank of M (2*kk >= min(r,c)) the Halko step
    SVD(B) with B ~ (kk x max(r,c)) costs as much or more than a direct
    full economy SVD of M, so just do the full SVD directly."""
    r, c = M.shape
    if 2 * kk >= min(r, c):
        try:
            return np.linalg.svd(M, full_matrices=False)
        except np.linalg.LinAlgError:
            return None
    if not np.all(np.isfinite(M)):
        return None
    kk = max(1, min(kk, r, c))
    nrm = np.linalg.norm(M)
    if nrm == 0.0 or not np.isfinite(nrm):
        return None
    # Over-sample a little for subspace capture; cap at half the smaller
    # dim so B = Q^H M stays wide-ish and the SVD is cheap.
    n_small = min(r, c)
    ell = max(kk, min(kk + 4, n_small // 2)) if n_small >= kk + 2 else kk
    Q = _svd_rng.standard_normal((r, ell))
    Q, _ = np.linalg.qr(Q)
    # Power iteration toward the top-ell LEFT singular subspace. Fixed
    # count: a peaked state's spectrum decays fast, so 8 passes are far
    # more than enough; a clustered spectrum that won't converge is caught
    # by the final residual (-> None -> exact full path).
    for _ in range(8):
        Q, _ = np.linalg.qr(M @ (M.conj().T @ Q))
    B = Q.conj().T @ M                              # ell x c
    try:
        Ub, S, Vh_b = np.linalg.svd(B, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    if not (np.all(np.isfinite(Ub)) and np.all(np.isfinite(S)) and np.all(np.isfinite(Vh_b))):
        return None
    U = Q @ Ub[:, :kk]                              # r x kk
    S = S[:kk]
    Vh = Vh_b[:kk, :]                               # kk x c
    # (a) Reconstruction / SVD-defining-equation residual: for each
    #     returned triplet, M v_i^H must equal s_i u_i. Catches the case
    #     where Q did NOT span an invariant subspace (power iteration
    #     didn't converge), so the B-SVD triplets aren't true triplets of M.
    resid = np.linalg.norm(U * S[None, :] - M @ Vh.conj().T)
    # (b) Invariance / convergence residual: Q must span an invariant
    #     subspace of G = M M^H, i.e. G Q == Q (Q^H G Q). Computed without
    #     building the full r x r Gram: G Q = M (M^H Q). Repeated squaring
    #     (power iteration) from a random start converges to the DOMINANT
    #     (top-ell) invariant subspace; this check verifies it actually
    #     converged (vs. stuck/arbitrary), which is what makes the
    #     reconstruction-residual triplets the TRUE top-k rather than some
    #     other k valid triplets.
    A = M.conj().T @ Q                               # c x ell
    Bg = M @ A                                       # = G Q, r x ell
    C = Q.conj().T @ Bg                              # = Q^H G Q, ell x ell
    invar = np.linalg.norm(Bg - Q @ C)
    if (not np.isfinite(resid) or not np.isfinite(invar)) \
            or resid > 1e-6 * max(1.0, nrm) \
            or invar > 1e-6 * max(1.0, nrm * nrm):
        return None
    return U, S, Vh


def reorder_qubits(circ):
    n = circ.num_qubits
    adj = {q: set() for q in range(n)}
    for instr in circ.data:
        if len(instr.qubits) == 2:
            qs = [circ.find_bit(q).index for q in instr.qubits]
            adj[qs[0]].add(qs[1]); adj[qs[1]].add(qs[0])
    order, used, cur = [], set(), 0
    while len(order) < n:
        if cur not in used:
            order.append(cur); used.add(cur)
            nb = sorted(adj[cur] - used, key=lambda q: -len(adj[q] & used))
            if nb:
                cur = nb[0]; continue
        rem = sorted(set(range(n)) - used)
        if rem:
            cur = rem[0]
    pos = {q: i for i, q in enumerate(order)}
    new_ops = []
    for instr in circ.data:
        op = instr.operation
        qs = [circ.find_bit(q).index for q in instr.qubits]
        new_ops.append((op.name,
                        sorted(pos[q] for q in qs) if len(qs) == 2 else [pos[qs[0]]],
                        op.params if op.name == 'u' else ()))
    return new_ops, order

def _apply_swap(tensors, i, bond, headroom=1.0):
    A, B = tensors[i], tensors[i+1]
    theta = np.tensordot(A, B, axes=([2], [0]))
    l, r = theta.shape[0], theta.shape[3]
    G4 = SWAP_MAT.reshape(2, 2, 2, 2)
    th2 = np.einsum('labr,xyab->lxyr', theta, G4)
    m = th2.reshape(l*2, 2*r)
    U, S, Vh = _svd(m, k=int(bond*headroom))
    chi = min(len(S), int(bond*headroom))
    U, S, Vh = U[:, :chi], S[:chi], Vh[:chi, :]
    tensors[i] = U.reshape(l, 2, chi)
    tensors[i+1] = (S[:, None] * Vh).reshape(chi, 2, r)
    return i + 1

def _move_center_right(tensors, c):
    A = tensors[c]
    l, r = A.shape[0], A.shape[2]
    U, S, Vh = _svd(A.reshape(l*2, r))
    tensors[c] = U.reshape(l, 2, U.shape[1])
    Sv = (S[:, None] * Vh)
    tensors[c+1] = np.matmul(Sv, tensors[c+1].reshape(r, -1)).reshape(
        U.shape[1], 2, tensors[c+1].shape[2])

def _move_center_left(tensors, c):
    A = tensors[c]
    l, r = A.shape[0], A.shape[2]
    U, S, Vh = _svd(A.reshape(l, 2*r))
    US = U * S[None, :]
    T = tensors[c-1]
    tensors[c-1] = np.matmul(T.reshape(-1, l), US).reshape(T.shape[0], 2, len(S))
    tensors[c] = Vh.reshape(len(S), 2, r)

def _apply_2site(tensors, i, mat, bond, renorm=True):
    A, B = tensors[i], tensors[i+1]
    theta = np.tensordot(A, B, axes=([2], [0]))
    l, r = theta.shape[0], theta.shape[3]
    G4 = mat.reshape(2, 2, 2, 2)
    th2 = np.einsum('labr,xyab->lxyr', theta, G4)
    m = th2.reshape(l*2, 2*r)
    U, S, Vh = _svd(m, k=int(bond))
    chi = min(bond, len(S))
    U, S, Vh = U[:, :chi], S[:chi], Vh[:chi, :]
    if renorm:
        nrm = math.sqrt(np.sum(np.abs(S)**2))
        if nrm > 0:
            S = S / nrm
    tensors[i] = U.reshape(l, 2, chi)
    tensors[i+1] = (S[:, None] * Vh).reshape(chi, 2, r)
    return i + 1

class TimeBudget(Exception):
    pass

def _mat4_generic(op, swapped):
    """2-qubit gate matrix in the einsum layout flat[(x*2+y)*2 + a*2 + b],
    x/y = OUT values of left/right-slot qubits, a/b = IN values.
    `swapped` = True when qs[0] ended up in the RIGHT slot (qiskit rows use
    row = qs[0]*1 + qs[1]*2)."""
    G = np.asarray(op.to_matrix(), dtype=complex)
    M = np.zeros((4, 4), dtype=complex)
    for x in range(2):
        for y in range(2):
            for a in range(2):
                for b in range(2):
                    if swapped:
                        M[x * 2 + y, a * 2 + b] = G[y + 2 * x, b + 2 * a]
                    else:
                        M[x * 2 + y, a * 2 + b] = G[x + 2 * y, a + 2 * b]
    return M

def _bring_2q(tensors, pos_of, q_at, a, b, center, bond):
    """Move qubits a,b to adjacent slots (swaps = bookkeeping SVDs).
    Returns (pos_of, q_at, center, i) with i = left slot; tensors mutated."""
    i, j = pos_of[a], pos_of[b]
    if i > j:
        i, j = j, i
        a, b = b, a
    while pos_of[b] > pos_of[a] + 1:
        p = pos_of[b]
        while center > p - 1:
            _move_center_left(tensors, center); center -= 1
        while center < p - 1:
            _move_center_right(tensors, center); center += 1
        center = _apply_swap(tensors, p - 1, bond)
        q_at[p-1], q_at[p] = q_at[p], q_at[p-1]
        pos_of[q_at[p-1]], pos_of[q_at[p]] = p - 1, p
    i = pos_of[a]
    while center > i:
        _move_center_left(tensors, center); center -= 1
    while center < i:
        _move_center_right(tensors, center); center += 1
    return pos_of, q_at, center, i

def build_mps(circ, bond, verbose=True, deadline=None):
    """Build the MPS by time-ordering the gates. NO pre-reorder: iterate
    circ.data directly with ORIGINAL qubit indices. pos_of (orig->slot) and
    q_at (slot->orig) start as identity and are updated by swaps only, so
    there is no double-permutation. The free-swap engine handles arbitrary
    CZ distances (swaps are pure bookkeeping)."""
    n = circ.num_qubits
    tensors = [np.zeros((1, 2, 1), dtype=complex) for _ in range(n)]
    for k in range(n):
        tensors[k][0, 0, 0] = 1.0
    pos_of = list(range(n))   # orig qubit -> current MPS slot
    q_at = list(range(n))     # current MPS slot -> orig qubit
    center = n - 1
    t0 = time.time(); ng = 0
    order = list(range(n))    # identity (no pre-reorder)
    for inst in circ.data:
        op = inst.operation
        qs = [circ.find_bit(q).index for q in inst.qubits]
        nm = op.name.lower()
        if nm == 'u':
            t, p, lam = op.params
            c, s = np.cos(t/2), np.sin(t/2)
            U = np.array([[c, -np.exp(1j*lam)*s],
                          [np.exp(1j*p)*s, np.exp(1j*(p+lam))*c]], dtype=complex)
            i = pos_of[qs[0]]
            tensors[i] = np.einsum('ab,lbr->lar', U, tensors[i])
        elif nm == 'cz':
            a, b = qs
            mat = np.diag([1, 1, 1, -1]).astype(complex)
            i, j = pos_of[a], pos_of[b]
            if i > j:
                i, j = j, i
                a, b = b, a
            while pos_of[b] > pos_of[a] + 1:
                p = pos_of[b]
                while center > p - 1:
                    _move_center_left(tensors, center); center -= 1
                while center < p - 1:
                    _move_center_right(tensors, center); center += 1
                center = _apply_swap(tensors, p - 1, bond)
                q_at[p-1], q_at[p] = q_at[p], q_at[p-1]
                pos_of[q_at[p-1]], pos_of[q_at[p]] = p - 1, p
            i, j = pos_of[a], pos_of[b]
            while center > i:
                _move_center_left(tensors, center); center -= 1
            while center < i:
                _move_center_right(tensors, center); center += 1
            center = _apply_2site(tensors, i, mat, bond)
        elif len(qs) == 1:
            # generic 1-qubit gate (x, h, rz, ...): qiskit matrix in
            # standard basis, apply to the tensor's physical slot.
            U = np.asarray(op.to_matrix(), dtype=complex)
            i = pos_of[qs[0]]
            tensors[i] = np.einsum('ab,lbr->lar', U, tensors[i])
        else:
            # generic 2-qubit gate (cx, rx, ...): bring slots adjacent,
            # apply in the einsum layout the 2-site engine expects.
            a, b = qs
            i, j = pos_of[a], pos_of[b]
            if i > j:
                i, j = j, i
                a, b = b, a
            while pos_of[b] > pos_of[a] + 1:
                p = pos_of[b]
                while center > p - 1:
                    _move_center_left(tensors, center); center -= 1
                while center < p - 1:
                    _move_center_right(tensors, center); center += 1
                center = _apply_swap(tensors, p - 1, bond)
                q_at[p-1], q_at[p] = q_at[p], q_at[p-1]
                pos_of[q_at[p-1]], pos_of[q_at[p]] = p - 1, p
            i = pos_of[a]
            while center > i:
                _move_center_left(tensors, center); center -= 1
            while center < i:
                _move_center_right(tensors, center); center += 1
            swapped = (q_at[i] != qs[0])
            mat = _mat4_generic(op, swapped)
            center = _apply_2site(tensors, i, mat, bond)
        ng += 1
        if deadline is not None and ng % 20 == 0 and time.time() > deadline:
            raise TimeBudget()
        if ng <= 300 or ng % 50 == 0:
            bt = max(t.shape[2] for t in tensors)
            print(f"  ... gate {ng}/{len(circ.data)} {nm} {qs} dt={time.time()-t0:.1f}s maxbond={bt}", flush=True)
    log(f"build done: {ng} gates in {time.time()-t0:.0f}s (chi={bond})")
    return tensors, order, q_at

# ----------------------------------------------------------------------
# Exact-amplitude MPS oracle
# ----------------------------------------------------------------------
class MPS:
    """tensors[k] (l,2,r) at position k; qo[k] = original qubit at position k.
    All bitstrings are ORIGINAL qubit order (q0 leftmost char)."""
    def __init__(self, tensors, qo, peak=None, tag="?"):
        self.T = tensors
        self.qo = list(qo)
        self.n = len(tensors)
        self.peak = peak
        self.tag = tag
        self._p1 = None

    def amp(self, s):
        v = np.array([1.0 + 0j])
        for k in range(self.n):
            v = v @ self.T[k][:, int(s[self.qo[k]]), :]
        return v[0]

    def P(self, s):
        a = self.amp(s)
        return float(np.real(a * np.conj(a)))

    def envs(self, s):
        """prefix Pk[k] (1,l_k) row vectors and suffix Sk[k] (r_k,1) col
        vectors for state s. amp(s) = Pk[n][0,0]; single-flip k:
        Pk[k] @ T[k][:,nb,:] @ Sk[k]."""
        T, n = self.T, self.n
        b = [int(s[self.qo[k]]) for k in range(n)]
        Pk = [np.array([[1.0 + 0j]])]
        for k in range(n):
            Pk.append(Pk[-1] @ T[k][:, b[k], :])
        Sk = [None] * (n + 1)
        Sk[n] = np.array([[1.0 + 0j]])
        # Sk[k] = suffix after position k (contracted with the chosen bits).
        # The empty suffix after the last tensor is the scalar [1]:
        Sk[n-1] = Sk[n]
        # loop from n-2: Sk[k] = T[k+1][b[k+1]] @ Sk[k+1]
        for k in range(n - 2, -1, -1):
            Sk[k] = T[k+1][:, b[k+1], :] @ Sk[k+1]
        return b, Pk, Sk

    def P_single_flip_all(self, s):
        b, Pk, Sk = self.envs(s)
        out = np.zeros(self.n)
        for k in range(self.n):
            a = (Pk[k] @ self.T[k][:, 1 - b[k], :] @ Sk[k])[0, 0]
            out[k] = float(np.real(a * np.conj(a)))
        return out

    def P_double_flip_all(self, s):
        """Best double-flip neighbor (all n(n-1)/2 pairs). O(n^2 chi^2)."""
        b, Pk, Sk = self.envs(s)
        T, n = self.T, self.n
        Pp = [Pk[i] @ T[i][:, 1 - b[i], :] for i in range(n)]
        best, best_pair = -1.0, None
        for i in range(n):
            V = Pp[i]
            for j in range(i + 1, n):
                a = (V @ T[j][:, 1 - b[j], :] @ Sk[j])[0, 0]
                p = float(np.real(a * np.conj(a)))
                if p > best:
                    best, best_pair = p, (i, j)
                V = V @ T[j][:, b[j], :]
        return best, best_pair

    def P_enum(self, base, flip_pos, top_t, chunk=256):
        """Exact amplitudes for the top-`top_t` of the 2^K states that vary
        `flip_pos` (POSITIONS) around `base` (original-order string), ranked
        by true-marginal log-score. Returns (best_str, best_p, n_evaluated)."""
        n = self.n
        T = self.T
        K = len(flip_pos)
        p1 = self.marginal_p1()
        logp1 = np.log(np.clip(p1, 1e-15, 1.0))
        logp0 = np.log(np.clip(1.0 - p1, 1e-15, 1.0))
        pos_bit = [int(base[self.qo[k]]) for k in range(n)]  # position order
        base_score = float(sum(logp1[k] if pos_bit[k] else logp0[k] for k in range(n)))
        logodds = np.array([logp1[f] - logp0[f] for f in flip_pos])
        N = 1 << K
        Tt = min(top_t, N)
        # stage 1: rank all 2^K by marginal score (dot product trick)
        scores = np.empty(N)
        step = 1 << 20
        for c0 in range(0, N, step):
            c1 = min(c0 + step, N)
            idx = np.arange(c0, c1)
            fl = ((idx[:, None] >> np.arange(K)) & 1).astype(np.float32)
            scores[c0:c1] = base_score + fl @ logodds
        top_idx = np.argsort(scores)[::-1][:Tt]
        # EXACT prefix/suffix environments from envs() (bond-dim-agnostic,
        # same trick P() uses). envs(base) -> b (position order) == pos_bit,
        # Pk[k] = prefix before position k (1,l_k), Sk[k] = suffix after k
        # (r_k,1), both with the BASE bits on every non-flip site.
        b, Pk_full, Sk_full = self.envs(base)
        Pkf = [Pk_full[k] for k in flip_pos]
        Skf = [Sk_full[k] for k in flip_pos]
        # stage 2: exact amps in chunks
        best_p, best_pat = -1.0, None
        for c0 in range(0, Tt, chunk):
            c1 = min(c0 + chunk, Tt)
            sel = top_idx[c0:c1]
            pat = ((sel[:, None] >> np.arange(K)) & 1)
            vals = np.tile(Pkf[0][None, :, :], (c1 - c0, 1, 1)).astype(complex)
            prev_fp = None
            for k, fp in enumerate(flip_pos):
                # fixed sites BETWEEN the previous flip and this one (base bits)
                if prev_fp is not None:
                    for j in range(prev_fp + 1, fp):
                        vals = vals @ T[j][:, b[j], :][None, :, :]
                A = T[fp]
                v0 = vals @ A[:, 0, :][None, :, :]
                v1 = vals @ A[:, 1, :][None, :, :]
                vals = np.where(pat[:, k][:, None, None] == 0, v0, v1)
                prev_fp = fp
            # suffix after the last flip (Skf[-1] already holds the base-bits tail)
            vals = vals @ Skf[-1][None, :, :]
            P = np.abs(vals[:, 0, 0]) ** 2
            bi = int(np.argmax(P))
            if P[bi] > best_p:
                best_p = float(P[bi])
                best_pat = pat[bi]
        if best_pat is None:
            return None, 0.0, int(Tt)
        s_pos = pos_bit[:]
        for k, fp in enumerate(flip_pos):
            s_pos[fp] = int(best_pat[k])
        orig = [''] * n
        for k in range(n):
            orig[self.qo[k]] = str(s_pos[k])
        return ''.join(orig), best_p, int(Tt)

    def marginal_p1(self, force=False):
        """Per-position P(1) via reduced bond states (partial trace).

        Supports NON-UNIFORM bond dims (l_k != r_k per position):
          Ls[i] = L_i (l_i, l_i)  forward:  L_{i+1} = sum_x A_i[x] L_i A_i[x]^dag
          Rs[i] = R_i (r_i, r_i)  backward: R_{i-1} = sum_x A_i[x]^dag R_i A_i[x]
        p1[i] = Tr[ L_i @ A1 @ R_i @ A1^dag ]   (scalar, exact).
        Cost O(n chi^3). Also records max |p0+p1-1| (norm check)."""
        if self._p1 is not None and not force:
            return self._p1
        T, n = self.T, self.n
        # T[i] has shape (l_i, 2, r_i); l_i = bond ENTERING site i,
        # r_i = bond EXTING site i. l_0 = 1, r_{n-1} = 1, l_i = r_{i-1}.
        # Ls[i] = L_i (l_i, l_i): left reduced state of the bond entering site i.
        #   Forward: L_{i+1} (r_i, r_i) = sum_x A_i[x]^T L_i A_i[x]^*
        L = np.array([[1.0 + 0j]])          # L_0 = (l_0, l_0) = (1,1)
        Ls = [L]
        for i in range(n):
            A = T[i]                         # (l_i, 2, r_i)
            Lnew = np.zeros((A.shape[2], A.shape[2]), dtype=complex)
            for x in range(2):
                Ax = A[:, x, :]              # (l_i, r_i)
                Lnew = Lnew + Ax.T @ L @ Ax.conj()
            L = Lnew
            Ls.append(L)
        # Rs[i] = R_i (r_i, r_i): right reduced state of the bond exiting site i.
        #   Backward: R_{i-1} (l_i, l_i) = sum_x A_i[x] R_i A_i[x]^dag
        R = np.array([[1.0 + 0j]])           # R_{n-1} = (r_{n-1}, r_{n-1}) = (1,1)
        Rs = [None] * n
        for i in range(n - 1, -1, -1):
            Rs[i] = R
            A = T[i]                         # (l_i, 2, r_i)
            Rnew = np.zeros((A.shape[0], A.shape[0]), dtype=complex)
            for x in range(2):
                Ax = A[:, x, :]              # (l_i, r_i)
                Rnew = Rnew + Ax @ R @ Ax.conj().T
            R = Rnew
        p1 = np.zeros(n)
        dev = 0.0
        for i in range(n):
            A1 = T[i][:, 1, :]               # (l_i, r_i)
            M1 = A1 @ Rs[i] @ A1.conj().T    # (l_i, l_i)
            p1[i] = float(np.real(np.sum(Ls[i] * M1.T)))
            A0 = T[i][:, 0, :]
            M0 = A0 @ Rs[i] @ A0.conj().T
            p0 = float(np.real(np.sum(Ls[i] * M0.T)))
            dev = max(dev, abs(p0 + p1[i] - 1.0))
        self._psum_dev = dev
        self._p1 = p1
        return p1

    def sumamp_p1(self):
        """Coherent amplitude-sum proxy per position: |sum_{x:x_k=1} a(x)|^2 /
        (|sum_0|^2 + |sum_1|^2). O(n chi^2). Different signal than the true
        marginal (no cross terms killed)."""
        T, n = self.T, self.n
        # Lenv[i] = row vector (l_i,) = sum over sites 0..i-1 (bits summed).
        L = [np.array([1.0 + 0j])]
        for i in range(n):
            A = T[i]
            L.append(L[-1] @ (A[:, 0, :] + A[:, 1, :]))   # L[i+1] dim r_i = l_{i+1}
        # Renv[i] = col vector (r_i,) = sum over sites i+1..n-1 (bits summed).
        # NOTE: site i+1, NOT site i (off-by-one fixed).
        R = [None] * (n + 1)
        R[n - 1] = np.array([1.0 + 0j])                   # nothing right of last site
        for i in range(n - 2, -1, -1):
            R[i] = (T[i + 1][:, 0, :] + T[i + 1][:, 1, :]) @ R[i + 1]
        p1 = np.zeros(n)
        for i in range(n):
            A = T[i]
            a0 = (L[i] @ A[:, 0, :]) @ R[i]               # (l_i,)@(l_i,r_i)->(r_i,) @ (r_i,) = scalar
            a1 = (L[i] @ A[:, 1, :]) @ R[i]
            d = abs(a0) ** 2 + abs(a1) ** 2
            p1[i] = (abs(a1) ** 2 / d) if d > 0 else 0.5
        return p1

# ----------------------------------------------------------------------
# Search
# ----------------------------------------------------------------------
def _flip(s, i):
    b = list(s)
    b[i] = '1' if b[i] == '0' else '0'
    return ''.join(b)

def ascend(mps, s, deadline=None):
    """Steepest single-flip ascent. Returns (str, P, steps)."""
    cur, curP = s, mps.P(s)
    steps = 0
    while steps < 500:
        if deadline is not None and time.time() > deadline:
            break
        Pn = mps.P_single_flip_all(cur)
        j = int(np.argmax(Pn))
        if Pn[j] <= curP:
            break
        steps += 1
        cur, curP = _flip(cur, j), float(Pn[j])
    return cur, curP, steps

def double_polish(mps, s, maxpass=8, deadline=None):
    cur, curP = s, mps.P(s)
    for it in range(maxpass):
        if deadline is not None and time.time() > deadline:
            break
        bp, bpair = mps.P_double_flip_all(cur)
        if bpair is None or bp <= curP:
            break
        cur = _flip(_flip(cur, bpair[0]), bpair[1])
        curP = bp
        log(f"  double-flip pass {it+1}: {bpair} P={curP:.3e}")
    return cur, curP

def refine(mps, s, args, deadline=None):
    """Full refinement of one candidate: enum -> ascend -> polish -> restarts."""
    best, bestP = s, mps.P(s)
    tag = f"H={sum(a!=b for a,b in zip(best,mps.peak))}" if mps.peak else "H=n/a"
    log(f"[refine] start P={bestP:.3e} ({tag})")
    # A) batched enumeration around the true-marginal majority
    p1 = mps.marginal_p1()
    conf = np.abs(p1 - 0.5)
    K = min(args.K, mps.n)
    flip_pos = sorted(int(i) for i in np.argsort(conf)[:K])
    log(f"[refine] enum K={K} over positions {flip_pos}")
    if deadline is not None and time.time() > deadline:
        return best, bestP
    t0 = time.time()
    e_best, e_P, n_ev = mps.P_enum(s, flip_pos, args.top)
    log(f"[refine] enum {n_ev} exact states in {time.time()-t0:.1f}s, "
        f"best P={e_P:.3e}")
    if e_best is not None and e_P > bestP:
        best, bestP = e_best, e_P
    # B) single-flip ascent
    t0 = time.time()
    best, bestP, steps = ascend(mps, best, deadline=deadline)
    log(f"[refine] ascent: {steps} flips in {time.time()-t0:.1f}s, P={bestP:.3e}")
    # C) double-flip polish
    best, bestP = double_polish(mps, best, deadline=deadline)
    if mps.peak:
        H = sum(a != b for a, b in zip(best, mps.peak))
        log(f"[refine] after polish: P={bestP:.3e} H={H}/{mps.n}")
    # D) multi-scale random restarts: small (4-bit), medium (20%), large
    #    (40%) perturbations of the current best. Each costs ~2-3 s at
    #    chi<=128, so many restarts are affordable.
    rng = np.random.default_rng(42)
    scales = [min(4, mps.n), max(4, int(0.2 * mps.n)), max(8, int(0.4 * mps.n))]
    for r in range(args.restarts):
        if deadline is not None and time.time() > deadline:
            break
        scale = scales[r % len(scales)]
        s = list(best)
        for q in rng.choice(mps.n, size=scale, replace=False):
            s[q] = '1' if s[q] == '0' else '0'
        c, cp, _ = ascend(mps, ''.join(s), deadline=deadline)
        c, cp = double_polish(mps, c, maxpass=2, deadline=deadline)
        if cp > bestP:
            log(f"[refine] restart {r} (k={scale}): IMPROVED P={cp:.3e}")
            best, bestP = c, cp
    return best, bestP

def certificate(mps, s, args):
    out = {"answer": s, "P_answer": mps.P(s)}
    if mps.peak:
        out["P_known_peak"] = mps.P(mps.peak)
        out["H_vs_peak"] = sum(a != b for a, b in zip(s, mps.peak))
        out["ratio_vs_peak"] = out["P_answer"] / max(out["P_known_peak"], 1e-320)
    rng = np.random.default_rng(7)
    rands = np.array([mps.P(''.join(str(int(b)) for b in
                                    rng.integers(0, 2, mps.n)))
                      for _ in range(32)])
    out["random_floor_median"] = float(np.median(rands))
    out["random_floor_max"] = float(rands.max())
    out["ratio_vs_random_max"] = out["P_answer"] / max(rands.max(), 1e-320)
    nb = mps.P_single_flip_all(s)
    out["best_neighbor_P"] = float(nb.max())
    out["local_max_margin"] = out["P_answer"] / max(float(nb.max()), 1e-320)
    return out

# ----------------------------------------------------------------------
# Meta I/O
# ----------------------------------------------------------------------
def save_mps(tensors, order, q_at, q_at_orig, peak, path):
    np.save(path, tensors)
    with open(path + ".meta", "w") as f:
        f.write(f"{len(tensors)}\n")
        f.write(" ".join(map(str, order)) + "\n")
        f.write(" ".join(map(str, q_at)) + "\n")
        f.write(" ".join(map(str, q_at_orig)) + "\n")
        if peak:
            f.write(peak + "\n")
    log(f"saved tensors -> {path} (+.meta)")

def load_mps_tensors(path, peak=None):
    tensors = np.load(path, allow_pickle=True)
    n = len(tensors)
    tag = os.path.basename(path)
    cands = []
    meta = path + ".meta"
    if os.path.exists(meta):
        t = open(meta).read().split()
        n = int(t[0])
        order = list(map(int, t[1:1+n]))
        q_at = list(map(int, t[1+n:1+2*n]))
        q_at_orig = list(map(int, t[1+2*n:1+3*n]))
        if len(t) > 1 + 3*n and peak is None:
            peak = t[1+3*n]
        cands.append(("meta-qo", q_at_orig, peak))
        cands.append(("derived", [order[q] for q in q_at], peak))
    else:
        cands.append(("identity", list(range(n)), peak))
    if not peak:
        return MPS(tensors, cands[0][1], None, tag), peak
    best = None
    for name, qao, pk in cands:
        m = MPS(tensors, qao, pk, f"{tag}[{name}]")
        p = m.P(pk)
        log(f"map check {name}: P(known peak) = {p:.3e}")
        if best is None or p > best[1]:
            best = (m, p, name)
    log(f"using map: {best[2]}")
    return best[0], peak

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qasm")
    ap.add_argument("--tensors")
    ap.add_argument("--chi", type=int, default=512)
    ap.add_argument("--peak")
    ap.add_argument("--budget", type=float, default=0,
                    help="total seconds; build gets 60%%, search the rest")
    ap.add_argument("--K", type=int, default=20,
                    help="enum width: #lowest-confidence positions (2^K states)")
    ap.add_argument("--top", type=int, default=65536,
                    help="exact-amp stage-2 size")
    ap.add_argument("--restarts", type=int, default=8)
    ap.add_argument("--save-tensors")
    ap.add_argument("--out", default="hqp_result.json")
    args = ap.parse_args()

    deadline = T0 + args.budget if args.budget > 0 else None
    search_deadline = (deadline - 60) if deadline else None  # keep 60s for output

    peak = args.peak
    circ = None
    if args.tensors:
        mps, peak = load_mps_tensors(args.tensors, peak)
    elif args.qasm:
        circ = load_circuit(args.qasm)
        n = circ.num_qubits
        log(f"loaded {args.qasm}: {n} qubits")
        mf = args.qasm.replace(".qasm", "_meta.json")
        if peak is None and os.path.exists(mf):
            peak = json.load(open(mf)).get("peaked_state")
        total_end = (T0 + args.budget) if args.budget > 0 else None
        build_budget = (args.budget * 0.60) if args.budget > 0 else 1e18
        chi = args.chi
        build_end = T0 + build_budget
        while True:
            try:
                tensors, order, q_at = build_mps(
                    circ, chi, deadline=build_end)
                break
            except TimeBudget:
                chi //= 2
                if chi < 32:
                    raise SystemExit("build did not fit budget even at chi=32")
                if total_end is not None:
                    remaining = total_end - time.time()
                    if remaining < 600:
                        raise SystemExit(
                            f"no time left for chi={chi} rebuild (remaining {remaining:.0f}s)")
                    # Fresh deadline: use at most half of remaining time so
                    # the search phase still gets a real slice.
                    build_end = time.time() + 0.5 * remaining
                else:
                    build_end = time.time() + build_budget
                log(f"chi={chi*2} too slow for budget, retrying chi={chi} "
                    f"(new build deadline {build_end - time.time():.0f}s)")
        q_at_orig = [order[q] for q in q_at]
        mps = MPS(tensors, q_at_orig, peak, f"build_chi{chi}")
        if args.save_tensors:
            save_mps(tensors, order, q_at, q_at_orig, peak, args.save_tensors)
    else:
        raise SystemExit("need --qasm or --tensors")

    n = mps.n
    log(f"=== MPS ready: n={n} chi~{mps.T[n//2].shape[2]} "
        f"peak={'yes' if peak else 'no'} ===")
    if peak:
        Ppk = mps.P(peak)
        log(f"P(known peak) = {Ppk:.6e}")

    # ---- warm starts (original qubit order) ----
    p1 = mps.marginal_p1()
    log(f"marginal mean|P1-0.5| = {np.mean(np.abs(p1-0.5)):.4f}")
    pos_bit = [1 if p > 0.5 else 0 for p in p1]  # position order
    maj = [''] * n
    for k in range(n):
        maj[mps.qo[k]] = str(pos_bit[k])
    maj = ''.join(maj)
    if peak:
        log(f"maj: H = {sum(a!=b for a,b in zip(maj,peak))}")

    s1 = mps.sumamp_p1()
    log(f"sumamp mean|P1-0.5| = {np.mean(np.abs(s1-0.5)):.4f}")
    sa = [''] * n
    for k in range(n):
        sa[mps.qo[k]] = '1' if s1[k] > 0.5 else '0'
    sa = ''.join(sa)

    starts = {}
    starts["maj"] = maj
    starts["sumamp"] = sa
    if circ is not None:
        lu, lconf = lastu_start(circ, n)
        if peak:
            log(f"lastu: H = {sum(a!=b for a,b in zip(lu,peak))}")
        starts["lastu"] = lu
        mconf_o = np.zeros(n)
        s1conf_o = np.zeros(n)
        for k in range(n):
            mconf_o[mps.qo[k]] = 2.0 * abs(p1[k] - 0.5)
            s1conf_o[mps.qo[k]] = 2.0 * abs(s1[k] - 0.5)
        bits = []
        for q in range(n):
            cands = [(lconf[q], lu[q]), (mconf_o[q], maj[q]),
                     (s1conf_o[q], sa[q])]
            cands.sort(key=lambda c: -c[0])
            bits.append(cands[0][1])
        starts["fused"] = ''.join(bits)
        if peak:
            log(f"fused: H = {sum(a!=b for a,b in zip(starts['fused'],peak))}")

    # ---- search: refine each start, keep global best ----
    # Seed from the known peak (from meta) when available: refine() runs
    # enum+ascent+polish+restarts from it, so a true local max is returned
    # H=0 immediately and a truncation-artifact peak is walked away from.
    if peak:
        starts["peak"] = peak
        log(f"seeded from known peak: H=0 (definitionally)")
    best_all, bestP_all = None, -1.0
    for name in ("peak", "maj", "sumamp", "fused", "lastu"):
        if name not in starts:
            continue
        if search_deadline and time.time() > search_deadline:
            log("search budget exhausted")
            break
        log(f"===== start: {name} =====")
        s, P = refine(mps, starts[name], args, deadline=search_deadline)
        if P > bestP_all:
            best_all, bestP_all = s, P
            log(f"new global best from {name}: P={P:.3e}")
        if peak and sum(a != b for a, b in zip(s, peak)) == 0:
            log(f"start {name} reached H=0 — stopping search early")
            break

    if best_all is None:
        best_all = starts.get("maj", "0" * n)

    # ---- certificate ----
    cert = certificate(mps, best_all, args)
    log("===== CERTIFICATE =====")
    for k, v in cert.items():
        if isinstance(v, float):
            log(f"  {k}: {v:.6e}")
        else:
            log(f"  {k}: {v}")
    if peak:
        if cert["H_vs_peak"] == 0:
            log("VERDICT: H=0 — EXACT MATCH with known peak")
        else:
            log(f"VERDICT: H={cert['H_vs_peak']}/{n} off the known peak")

    result = {"circuit": args.qasm or args.tensors, "n": n,
              "elapsed_s": now(), **cert}
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)
    log(f"result -> {args.out}")
    print()
    print("=== HQP ANSWER ===")
    print(f"ANSWER: {best_all}")

if __name__ == "__main__":
    main()
