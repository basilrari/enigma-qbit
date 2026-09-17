#!/usr/bin/env python3
"""EXACT MPS SAMPLER + heavy-hitter ranking.

Our solver has only ever HILL-CLIMBED the MPS amplitude landscape
(ascend / double_polish / marginals), and every warm start derives from the
SAME marginal vector, so all restarts share one basin.  The challenge's own
reference solver does something different: it runs Aer MPS and takes
10,000 SHOTS, then reports the most frequent bitstring.

This script does the equivalent - exact sequential conditional sampling from
our MPS - and then reports the heavy hitters, which is the operation that is
robust to truncation (heavy-hitter MASS survives truncation; fine amplitude
ORDERING does not).

Sampling is gauge-independent: we use suffix norm environments
    R_n = 1,   R_k = sum_v T_k^v R_{k+1} (T_k^v)^dagger   (shape l_k x l_k)
so that P(b_<k) = v_k R_k v_k^dagger with v_k the prefix row-vector, and
P(b_k = v | b_<k) is obtained from v_k T_k^v R_{k+1} (v_k T_k^v)^dagger.
No canonical form required, no left-right sweep.

usage: sample_probe.py <circuit_id> <chi> <nsamples> [batch]
"""
import sys, os, json, time, math
import numpy as np

VER = "/mnt/8tb_hdd2/basilrari/enigma-work/verify"
sys.path.insert(0, VER)
import hqp_solver as HS

# Optional: force the exact full SVD, disabling the randomized top-k
# subspace-iteration fast path in _svd().  Used to test whether the
# approximation is silently suppressing the planted peak at large chi.
if os.environ.get("HQP_EXACT_SVD") == "1":
    HS._svd = lambda m, k=None: HS._svd_full(m)
    print("[cfg] HQP_EXACT_SVD=1 -> exact full SVD forced, top-k path disabled",
          flush=True)
if os.environ.get("HQP_POWER_STEPS"):
    _n = int(os.environ["HQP_POWER_STEPS"])
    _orig = HS._svd_topk

    def _patched(M, kk):
        import numpy as _np
        r, c = M.shape
        if 2 * kk >= min(r, c):
            try:
                return _np.linalg.svd(M, full_matrices=False)
            except _np.linalg.LinAlgError:
                return None
        if not _np.all(_np.isfinite(M)):
            return None
        kk = max(1, min(kk, r, c))
        nrm = _np.linalg.norm(M)
        if nrm == 0.0 or not _np.isfinite(nrm):
            return None
        n_small = min(r, c)
        ell = max(kk, min(kk + 4, n_small // 2)) if n_small >= kk + 2 else kk
        Q = HS._svd_rng.standard_normal((r, ell))
        Q, _ = _np.linalg.qr(Q)
        for _ in range(_n):
            Q, _ = _np.linalg.qr(M @ (M.conj().T @ Q))
        B = Q.conj().T @ M
        try:
            Ub, S, Vh_b = _np.linalg.svd(B, full_matrices=False)
        except _np.linalg.LinAlgError:
            return None
        U = Q @ Ub[:, :kk]
        S = S[:kk]
        Vh = Vh_b[:kk, :]
        resid = _np.linalg.norm(U * S[None, :] - M @ Vh.conj().T)
        if not _np.isfinite(resid) or resid > 1e-6 * max(1.0, nrm):
            return None
        return U, S, Vh

    HS._svd_topk = _patched
    print(f"[cfg] HQP_POWER_STEPS={_n} -> top-k power iterations overridden", flush=True)

KNOWN = {
    "d1_s1_4043cafb": "0001001101001111101001001110010001111010100000",
    "d1_s2_adeddcf3": "111001111111010110011001010100001001101011001101",
    "d2_s1_39b370e4": "1110101100010111001000000011101111001000",
    "d2_s2_1efabaf4": "11111101001110010011000110110000111010100010",
    "d3_s1_2674779a": "011100110011000101111001110011100000101101100100",
    "d3_s2_c09ba537": "110110111110110110110100101000010000101110111010",
}


def hamming(a, b):
    return sum(1 for x, y in zip(a, b) if x != y)


def suffix_norms(T):
    """R[k] shape (l_k, l_k); R[n] = [[1]].  Cost O(n * chi^3) once."""
    n = len(T)
    R = [None] * (n + 1)
    R[n] = np.ones((1, 1), dtype=complex)
    for k in range(n - 1, -1, -1):
        acc = None
        for v in (0, 1):
            A = T[k][:, v, :]
            term = A @ R[k + 1] @ A.conj().T
            acc = term if acc is None else acc + term
        R[k] = acc
    return R


def sample_batch(T, R, n, N, rng):
    """Vectorised over N.  Returns uint8 array (N, n) in MPS POSITION order."""
    bits = np.zeros((N, n), dtype=np.uint8)
    V = np.ones((N, 1), dtype=complex)
    for k in range(n):
        Tk = T[k]
        A0 = V @ Tk[:, 0, :]          # (N, r)
        A1 = V @ Tk[:, 1, :]
        Rk = R[k + 1]
        p0 = np.einsum('ni,ij,nj->n', A0, Rk, A0.conj()).real
        p1 = np.einsum('ni,ij,nj->n', A1, Rk, A1.conj()).real
        p0 = np.maximum(p0, 0.0)
        p1 = np.maximum(p1, 0.0)
        tot = p0 + p1
        tot[tot <= 0] = 1.0
        b = (rng.random(N) >= (p0 / tot)).astype(np.uint8)
        bits[:, k] = b
        V = np.where(b[:, None] == 0, A0, A1)
    return bits


def main():
    cid = sys.argv[1]
    chi = int(sys.argv[2])
    NS = int(sys.argv[3])
    batch = int(sys.argv[4]) if len(sys.argv) > 4 else 4000
    tag_suffix = sys.argv[5] if len(sys.argv) > 5 else ""
    t0 = time.time()
    known = KNOWN[cid]
    path = f"{VER}/{cid}.qasm"

    print(f"=== {cid} chi={chi} nsamples={NS} batch={batch} ===", flush=True)
    print(f"known answer ({len(known)} bits): {known}", flush=True)

    circ = HS.load_circuit(path)
    n = circ.num_qubits
    print(f"qubits={n} gates={len(circ.data)}  known_len={len(known)}", flush=True)
    assert n == len(known), "known-answer length must match qubit count"

    print("building MPS ...", flush=True)
    tb = time.time()
    tensors, order, q_at = HS.build_mps(circ, chi, verbose=False)
    print(f"build done in {time.time()-tb:.0f}s  maxbond={max(t.shape[2] for t in tensors)}",
          flush=True)
    mps = HS.MPS(tensors, q_at, peak=None, tag=cid)

    print("suffix environments ...", flush=True)
    R = suffix_norms(tensors)
    print(f"  done {time.time()-tb:.0f}s", flush=True)

    # MPS's own exact-ish score for the known answer (for reference only)
    Pk = mps.P(known)
    print(f"MPS P(known answer) = {Pk:.6e}", flush=True)

    rng = np.random.default_rng(12345)
    from collections import Counter
    cnt = Counter()
    done = 0
    ts = time.time()
    while done < NS:
        nb = min(batch, NS - done)
        bits = sample_batch(tensors, R, n, nb, rng)
        for row in bits:
            # MPS position order -> ORIGINAL qubit order via q_at
            s = [0] * n
            for k in range(n):
                s[q_at[k]] = int(row[k])
            cnt["".join(map(str, s))] += 1
        done += nb
        if done % (batch * 5) == 0 or done >= NS:
            el = time.time() - ts
            rate = done / max(el, 1e-9)
            print(f"  sampled {done}/{NS}  {el:.0f}s  ({rate:.0f}/s)  distinct={len(cnt)}",
                  flush=True)

    print(f"\nsampling done in {time.time()-ts:.0f}s, distinct={len(cnt)}", flush=True)
    hits = cnt.get(known, 0)
    print(f"*** TRUE ANSWER APPEARED {hits} times  (freq {hits/NS:.5f}) ***", flush=True)

    top = cnt.most_common(25)
    print("\ntop 25 heavy hitters:", flush=True)
    print(f"{'rank':>5} {'count':>7} {'freq':>9} {'H':>3}  bits", flush=True)
    for i, (s, c) in enumerate(top, 1):
        print(f"{i:>5} {c:>7} {c/NS:>9.5f} {hamming(s,known):>3}  {s}", flush=True)

    # rank of the true answer if it appeared
    if hits:
        rk = 1 + sum(1 for s, c in cnt.items() if c > hits)
        print(f"\nrank of TRUE ANSWER in histogram: {rk} of {len(cnt)}", flush=True)

    out = {
        "cid": cid, "chi": chi, "nsamples": NS, "qubits": n,
        "tag": tag_suffix, "known": known, "true_hits": hits,
        "true_freq": hits / NS,
        "distinct": len(cnt), "maxbond": int(max(t.shape[2] for t in tensors)),
        "mps_P_known": Pk,
        "top": [{"bits": s, "count": c, "h": hamming(s, known)} for s, c in top],
        "elapsed_s": time.time() - t0,
    }
    if hits:
        out["true_rank"] = 1 + sum(1 for s, c in cnt.items() if c > hits)
    with open(f"{VER}/sample_{cid}_c{chi}{tag_suffix}.json", "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {VER}/sample_{cid}_c{chi}{tag_suffix}.json  "
          f"total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
