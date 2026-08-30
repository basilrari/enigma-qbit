#!/bin/bash
# Wave 2 (11:50): fixed deadline cascade + thread caps.
#  d1 -> chi=512 self-healing (falls back to 256/128 with FRESH deadlines)
#  d2/d3 -> chi=256 directly (512 infeasible at current box load ~71)
# 3 jobs x 4 OpenBLAS threads = 12 threads on 32 cores.
cd /mnt/8tb_hdd2/basilrari/enigma-work/verify || exit 1
PY=/mnt/8tb_hdd2/basilrari/enigma-work/l2venv/bin/python
HQP=hqp_solver.py
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export OPENBLAS_MAX_THREADS=4 OMP_DYNAMIC=FALSE

run () {  # run <name> <qasm> <chi> <budget>
  echo "=== $(date '+%H:%M:%S') START $1 chi=$3 budget=$4s ==="
  "$PY" "$HQP" --qasm "$2" --chi "$3" --budget "$4" \
    --out "/tmp/${1}.json" > "/tmp/${1}.log" 2>&1
  local rc=$?
  echo "=== $(date '+%H:%M:%S') END   $1 rc=$rc ==="
  tail -18 "/tmp/${1}.log" | sed "s|^|  [$1] |"
}

run d1_c256 d1_s1_4043cafb.qasm 256 7200 & P1=$!
run d2_c256 d2_s1_39b370e4.qasm 256 7200 & P2=$!
run d3_c256 d3_s1_2674779a.qasm 256 10800 & P3=$!

wait $P1 $P2 $P3
echo "=== $(date '+%H:%M:%S') WAVE2 COMPLETE ==="
