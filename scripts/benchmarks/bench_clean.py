import numpy as np, time, os, sys
rng = np.random.default_rng(0)
def t(name, n, thr):
    os.environ.update({k: str(thr) for k in
        ["OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"]})
    M = rng.standard_normal((n, n)) + 1j*rng.standard_normal((n, n))
    # warm
    np.linalg.svd(M, full_matrices=False)
    ts = []
    for _ in range(3):
        t0 = time.time(); np.linalg.svd(M, full_matrices=False); ts.append(time.time()-t0)
    print(f"{name:<14} n={n:<5} threads={thr:<3}  {1000*min(ts):8.1f} ms", flush=True)
for n in (512, 1024):
    for thr in (1, 2, 4, 8):
        t("fullSVD", n, thr)
print("DONE", flush=True)
