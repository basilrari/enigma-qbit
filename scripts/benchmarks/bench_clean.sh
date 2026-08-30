#!/bin/bash
# Clean per-thread-count SVD bench (env set BEFORE python starts).
cd /mnt/8tb_hdd2/basilrari/enigma-work/verify || exit 1
PY=/mnt/8tb_hdd2/basilrari/enigma-work/l2venv/bin/python
for T in 1 2 4 8; do
  OPENBLAS_NUM_THREADS=$T OMP_NUM_THREADS=$T MKL_NUM_THREADS=$T \
  OPENBLAS_MAX_THREADS=$T OMP_DYNAMIC=FALSE \
  "$PY" - <<'EOF'
import numpy as np, time
rng = np.random.default_rng(0)
for n in (512, 1024):
    M = rng.standard_normal((n, n)) + 1j*rng.standard_normal((n, n))
    np.linalg.svd(M, full_matrices=False)  # warm
    ts = []
    for _ in range(3):
        t0 = time.time(); np.linalg.svd(M, full_matrices=False); ts.append(time.time()-t0)
    import os
    print(f"n={n:<5} threads={os.environ['OPENBLAS_NUM_THREADS']:<3}  {1000*min(ts):8.1f} ms", flush=True)
EOF
done
echo ALLDONE
