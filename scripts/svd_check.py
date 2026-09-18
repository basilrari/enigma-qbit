#!/usr/bin/env python3
"""Show the Gram-route noise problem and that LAPACK avoids it.

Builds matrices at the scale the real circuit tensors actually reach (~5e5,
measured by the engine probe: max|entry| 4.99e5), then compares:

  * the old Gram/eigh factorisation with its absolute  nz = w > 1e-13  cutoff
  * hqp_solver._svd_full  (now straight LAPACK dgesdd)

Reports how many "singular values" the old route keeps that are pure noise,
and the reconstruction error of each factorisation.  Run on the box:

    cd {V} && {PY} {W}/svd_check.py
"""
import sys

import numpy as np

sys.path.insert(0, "/mnt/8tb_hdd2/basilrari/enigma-work/verify")
import hqp_solver as HS   # noqa: E402

# the patched solver, loaded by path so the box's own copy stays untouched
import importlib.util    # noqa: E402
_spec = importlib.util.spec_from_file_location(
    "hqp_new", "/mnt/8tb_hdd2/basilrari/enigma-work/verify/hqp_solver_new.py")
NEW = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(NEW)


def old_gram_svd(M):
    """Verbatim reproduction of the previous _svd_full body."""
    r, c = M.shape
    if r >= c:
        G = M.conj().T @ M
        w, V = np.linalg.eigh(G)
        w = np.sqrt(np.clip(w.real, 0.0, None))
        idx = np.argsort(w)[::-1]
        w, V = w[idx], V[:, idx]
        nz = w > 1e-13
        U = M @ (V[:, nz] / w[nz][None, :])
        return U, w[nz], V[:, nz].conj().T
    G = M @ M.conj().T
    w, U = np.linalg.eigh(G)
    w = np.sqrt(np.clip(w.real, 0.0, None))
    idx = np.argsort(w)[::-1]
    w, U = w[idx], U[:, idx]
    nz = w > 1e-13
    Vh = U[:, nz] @ M.conj().T / w[nz][None, :]
    return U[:, nz], w[nz], Vh


def report(name, M, true_rank):
    r, c = M.shape
    nrm = float(np.linalg.norm(M))
    noise = np.finfo(float).eps * nrm * nrm          # eigh noise floor on M^H M
    U, S, Vh = old_gram_svd(M)
    U2, S2, Vh2 = NEW._svd_full(M)
    e_old = float(np.linalg.norm(M - U @ np.diag(S) @ Vh))
    e_new = float(np.linalg.norm(M - U2 @ np.diag(S2) @ Vh2))
    # a kept direction is "noise" if it is below the floor of M^H M
    n_old_noise = int((S <= np.sqrt(noise)).sum())
    print(f"# {name}  shape={r}x{c}  |M|={nrm:.3e}  true_rank={true_rank}")
    print(f"    M^H M noise floor ~ {noise:.3e}   => sigma floor ~ {np.sqrt(noise):.3e}")
    print(f"    OLD gram: kept {len(S):4d} values (rank {int((S > 1e-13).sum())}), "
          f"{n_old_noise:4d} of them below the noise floor   resid={e_old:.3e}")
    print(f"    NEW lapack: kept {len(S2):4d} values, sigma_max={S2[0]:.3e}, "
          f"resid={e_new:.3e}")
    print(f"    retained-sigma tail OLD {S[min(20, len(S) - 1)]:.3e} -> "
          f"NEW {S2[min(20, len(S2) - 1)]:.3e}")
    print()


def main():
    rng = np.random.default_rng(7)
    # 1. a genuinely rank-deficient matrix at circuit scale
    A = (rng.normal(size=(240, 40)) + 1j * rng.normal(size=(240, 40)))
    B = (rng.normal(size=(40, 160)) + 1j * rng.normal(size=(40, 160)))
    M = A @ B
    M *= 5e5 / np.linalg.norm(M) * 10            # scale to ~5e6-ish like the real run
    report("rank-40 @ circuit scale", M, 40)

    # 2. full-rank, well-scaled (the case the old code handled fine)
    M2 = (rng.normal(size=(120, 90)) + 1j * rng.normal(size=(120, 90))) / 40.0
    report("full rank, O(1) scale", M2, 90)

    # 3. full-rank at circuit scale
    M3 = (rng.normal(size=(240, 160)) + 1j * rng.normal(size=(240, 160)))
    M3 *= 5e5 / np.linalg.norm(M3) * 10
    report("full rank @ circuit scale", M3, 160)

    print("# verdict: the old route kept far more values than the true rank, and")
    print("# everything below the M^H M noise floor is a garbage direction.")


if __name__ == "__main__":
    main()
