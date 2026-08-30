#!/bin/bash
# Parallel chi=256 wave (3 jobs x 4 OpenBLAS threads = 12 threads on 32 cores).
# chi=512 builds are infeasible under current shared-box load (load ~38,
# 9.5s/gate -> 5.5h > budget); revisit c512 tonight when the box is quieter.
cd /mnt/8tb_hdd2/basilrari/enigma-work/verify || exit 1
PY=/mnt/8tb_hdd2/basilrari/enigma-work/l2venv/bin/python
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 NUMEXPR_NUM_THREADS=4

run() {
  local name=$1 qasm=$2 budget=$3
  echo "[$(date '+%H:%M:%S')] START $name (budget=${budget}s)"
  $PY hqp_solver.py --qasm "$qasm" --chi 256 --budget "$budget" > "/tmp/${name}.log" 2>&1
  local rc=$?
  echo "[$(date '+%H:%M:%S')] END $name rc=$rc"
  tail -14 "/tmp/${name}.log" | sed "s|^|  [$name] |"
}

run d1_c256 d1_s1_4043cafb.qasm 3600 & P1=$!
run d2_c256 d2_s1_39b370e4.qasm 3600 & P2=$!
run d3_c256 d3_s1_2674779a.qasm 5400 & P3=$!

wait $P1 $P2 $P3
echo "[$(date '+%H:%M:%S')] WAVE COMPLETE"
