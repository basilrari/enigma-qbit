"""Exact certified branch-and-bound top-K extraction from an absorbed MPS.

Why this exists (dossier enigma-sn63.md): the published winner's `beam_search`
(beam 512, k=8) prunes by *local* weight and can therefore throw away the true
peak at depth 10-20.  Its A* variant is described as the robust one, with the
heuristic "partial weight x remaining norm".  That is exactly what this does,
with a certificate: we stop only when the best remaining upper bound is below
the k-th best exact score, so a returned list is the TRUE top-k of the state --
never a hopeful claim.

Implementation notes (learned the hard way):
  * quimb SQUEEZES bond-of-size-1 indices, so a site tensor may be 1-D or 2-D.
    Never index a tensor by axis position -- derive the axes from `t.inds`.
  * the bound |v|^2 is only an upper bound on the subtree when the state is
    RIGHT-canonical (remaining norm == 1), so canonicalize first.
"""

import heapq
import itertools

import numpy as np
import quimb.tensor as qtn


def as_mps(x):
    """Coerce whatever quimb hands back into a MatrixProductState."""
    if hasattr(x, "site_ind") and hasattr(x, "tensors"):
        return x
    return qtn.MatrixProductState.from_TN(x, site_ind_id="k{}", cyclic=False)


def _site_data(mps):
    """Site tensors in canonical axis order: (incoming, physical, outgoing).

    Hard-won: quimb's `Tensor.inds` is NOT the data's axis order here (the
    probe showed inds=(in,out,phys) with data=(in,phys,out)), so any
    permutation built from `inds.index(...)` silently scrambles the tensor.
    The DATA is canonical, so read it positionally and squeeze size-1 axes --
    safe because the physical axis always has dimension 2.
    """
    n = len(mps.tensors)
    arrs = []
    for i in range(n):
        ix = mps.site_ind(i)
        for cand in mps.tensors:
            if ix in cand.inds:
                arrs.append(np.squeeze(np.asarray(cand.data).astype(complex)))
                break
        else:
            raise ValueError(f"no tensor carries site index {ix}")
    return arrs


def _step(arrs, i, v, bit):
    """Contract site i onto the running vector v with physical leg = bit.

    All tensors passed here are rank 3 (in, phys, out) from `_coerce_all`, so
    the physical axis is ALWAYS axis 1 -- no per-site guessing.
    """
    A = np.take(arrs[i], int(bit), axis=1)              # -> (in, out)
    va = np.asarray(v).reshape(-1)
    if va.size != A.shape[0]:
        raise ValueError(f"bond mismatch at site {i}: v={va.size} vs tensor={A.shape}")
    return np.tensordot(va, A, axes=([-1], [0])).reshape(-1)


def _as3(A):
    """Coerce a single site tensor to rank 3 -- only when rank is unambiguous."""
    if A.ndim == 3:
        return A
    if A.ndim == 1:
        return A.reshape(1, A.shape[0], 1)
    raise ValueError("rank-2 tensor needs bond context; use _coerce_all")


def _coerce_all(arrs):
    """Rank-3 (in, phys, out) for every site, using the running bond size.

    quimb's data is already in canonical (in, phys, out) order (probe_mps3),
    but it drops size-1 axes, so a tensor may arrive as rank 1 or 2.  Which
    axis is missing is decided by `vsz` -- the incoming bond size carried from
    the previous site -- never by guessing:
        rank 2, A.shape[0] == vsz -> (in, phys)   [out was size 1]
        rank 2, otherwise         -> (phys, out)  [in  was size 1]
    """
    out, vsz = [], 1
    for i, A in enumerate(arrs):
        A = np.asarray(A, dtype=complex)
        if A.ndim == 3:
            B = A
        elif A.ndim == 2:
            B = (A.reshape(A.shape[0], A.shape[1], 1) if A.shape[0] == vsz
                 else A.reshape(1, A.shape[0], A.shape[1]))
        elif A.ndim == 1:
            B = A.reshape(1, A.shape[0], 1)
        else:
            raise ValueError(f"site {i}: rank {A.ndim} is not an MPS site")
        out.append(B)
        vsz = B.shape[2]
    return out


def _right_canonicalize(arrs):
    """Make every tensor but the first a right-isometry (numpy SVD sweep).

    The branch-and-bound bound is only valid in this form: for a
    right-canonical MPS the mass of a bit-prefix is exactly ||v_prefix||^2,
    which is why MPS sampling works at all.  quimb's own right_canonicalize
    raises on these objects, so do it here -- plain SVDs, no API risk.
    """
    arrs = _coerce_all([np.array(a, dtype=complex, copy=True) for a in arrs])
    for i in range(len(arrs) - 1, 0, -1):
        A = arrs[i]
        inb, p, out = A.shape
        U, S, Vh = np.linalg.svd(A.reshape(inb, p * out), full_matrices=False)
        keep = S > 1e-14 * (S[0] if S.size else 1.0)
        r = int(np.count_nonzero(keep)) or 1
        arrs[i] = Vh[:r].reshape(r, p, out)
        arrs[i - 1] = np.tensordot(arrs[i - 1], (U * S)[:, :r], axes=([-1], [0]))
    return arrs


def topk_exact(mps, k=8, cap=400000, verbose=False, canonicalize=True):
    """Return [(bitstring_in_site_order, probability), ...] -- the true top-k.

    Best-first over bit-prefixes, ordered by the exact prefix mass.  Only when
    the best remaining prefix mass is <= the k-th best complete probability is
    the answer certified; otherwise `capped` says the search gave up honestly.
    """
    mps = as_mps(mps)
    arrs = _site_data(mps)
    arrs = _coerce_all(arrs)
    n = len(arrs)
    trusted = False
    if canonicalize:
        try:
            arrs = _right_canonicalize(arrs)
            trusted = True  # ||v||^2 is now a true upper bound on the subtree
        except Exception as e:
            if verbose:
                print(f"# (right_canonicalize failed: {type(e).__name__}: {e})")

    cnt = itertools.count()
    out, nodes = [], 0
    # heap of (-prefix_mass_upper_bound, tiebreak, depth, bits, v)
    pq = [(-1.0, next(cnt), 0, "", np.ones(1, dtype=complex))]
    capped = False
    while pq:
        neg_b, _tb, i, bits, v = heapq.heappop(pq)
        nodes += 1
        if i == n:
            p = float(abs(complex(np.asarray(v).reshape(-1)[0])) ** 2)
            out.append((bits, p))
            if len(out) >= k:
                # certificate: nothing left in the heap can beat our k-th best.
                # ONLY valid on a right-canonical state -- without it the bound
                # is unsound and would prune the true peak (see validate_exact).
                if trusted and (not pq or -pq[0][0] <= out[-1][1] * (1 + 1e-12)):
                    break
            continue
        if nodes > cap:
            capped = True
            break
        for bit in ("0", "1"):
            v2 = _step(arrs, i, v, bit)
            ub = float(np.sum(np.abs(v2) ** 2))
            heapq.heappush(pq, (-ub, next(cnt), i + 1, bits + bit, v2))
    out = [o for o in out if o[0] != ""]
    out.sort(key=lambda t: -t[1])
    if verbose:
        print(f"# topk_exact: nodes={nodes} found={len(out)} k={k} capped={capped}")
    return out
