import numpy as np, time, os, sys
n = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
iters = 5
rng = np.random.default_rng(0)
M = rng.standard_normal((n, n)) * (1j if False else 1.0)
# complex decaying-ish
Mc = M + 1j * M
# warmup
_ = np.linalg.svd(Mc, full_matrices=False)
ts = []
for _ in range(iters):
    t0 = time.time()
    np.linalg.svd(Mc, full_matrices=False)
    ts.append(time.time() - t0)
t = min(ts)
print(f"n={n}  best {t*1000:.0f}ms  ({1e6/(t*n**2):.0f} MFLOP-ish)  threads={os.environ.get('OPENBLAS_NUM_THREADS','default')}")
