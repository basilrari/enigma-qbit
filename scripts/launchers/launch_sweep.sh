#!/bin/bash
# Clean chi-sweep runner, sequential, thread-capped.
# Root cause of prior slowness: OpenBLAS (DYNAMIC_ARCH) oversubscription --
# 6 concurrent jobs x up to 64 threads thrashing 32 cores. Benchmarks:
#   512x512 SVD: 156ms (1 thr) vs 2383ms (default, oversubscribed)
#   1024x1024 SVD: ~733ms (2 thr)
# So: cap threads to 4 per job, run jobs sequentially, highest chi first
# (most information per unit time).
cd /mnt/8tb_hdd2/basilrari/enigma-work/verify || exit 1
PY=/mnt/8tb_hdd2/basilrari/enigma-work/l2venv/bin/python
HQP=hqp_solver.py

export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export OPENBLAS_MAX_THREADS=4 OMP_DYNAMIC=FALSE

run () {  # run <name> <qasm> <chi> <budget>
  echo "=== $(date +%T) START $1 chi=$3 budget=$4s ==="
  "$PY" "$HQP" --qasm "$2" --chi "$3" --budget "$4" \
    --out "/tmp/${1}.json" > "/tmp/${1}.log" 2>&1
  echo "=== $(date +%T) DONE  $1 (exit $?) ==="
}

run d1_c512 d1_s1_4043cafb.qasm 512 7200
run d1_c256 d1_s1_4043cafb.qasm 256 3600
run d2_c512 d2_s1_39b370e4.qasm 512 7200
run d2_c256 d2_s1_39b370e4.qasm 256 3600
run d3_c512 d3_s1_2674779a.qasm 512 7200
run d3_c256 d3_s1_2674779a.qasm 256 3600
echo "ALLDONE $(date +%T)"
