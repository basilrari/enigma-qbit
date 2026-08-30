#!/bin/bash
# Probe: real SVD cost at the bond sizes each chi hits, 1-thread, UNDER CURRENT LOAD.
# chi=64 -> 128x128 SVDs; chi=128 -> 256x256; chi=256 -> 512x512.
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
cd /mnt/8tb_hdd2/basilrari/enigma-work/verify
echo "load: $(uptime | sed 's/.*load/load')"
PY=/mnt/8tb_hdd2/basilrari/enigma-work/l2venv/bin/python
$PY - <<'PYEOF' 2>&1 | tail -8
import numpy as np, time
rng = np.random.default_rng(0)
def t(n, rep=3):
    M = rng.standard_normal((n,n)) + 1j*rng.standard_normal((n,n))
    # warm
    np.linalg.svd(M, full_matrices=False)
    ts=[]
    for _ in range(rep):
        s=time.perf_counter(); np.linalg.svd(M, full_matrices=False); ts.append(time.perf_counter()-s)
    return min(ts)
for n in (128,256,512):
    print(f"SVD {n}x{n} (1-thread, under load) = {t(n)*1000:.0f} ms")
PYEOF
echo DONE
