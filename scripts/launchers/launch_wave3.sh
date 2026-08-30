#!/bin/bash
# Wave 3: d1/d2/d3 @ chi=256, fixed _svd_full rank-deficient bug, thread-capped.
# Parallel (3 jobs x 4 OpenBLAS threads = 12 threads, fits 32 cores under load).
cd /mnt/8tb_hdd2/basilrari/enigma-work/verify || exit 1
PY=/mnt/8tb_hdd2/basilrari/enigma-work/l2venv/bin/python
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
       VECLIB_MAXIMUM_THREADS=4 NUMEXPR_NUM_THREADS=1 OPENMP_NUM_THREADS=4
echo "wave3 start $(date) load: $(uptime | sed 's/.*load/load')"

run() {
  local tag=$1 qasm=$2 budget=$3
  echo "=== $(date +%T) START $tag chi=256 budget=${budget}s ==="
  "$PY" hqp_solver.py --qasm "$qasm" --chi 256 --budget "$budget" --out "/tmp/$tag.json" \
      > "/tmp/$tag.log" 2>&1
  echo "=== $(date +%T) DONE $tag exit=$? ==="
  tail -18 "/tmp/$tag.log" | sed "s|^|  [$tag] |"
}

run d1_c256 d1_s1_4043cafb.qasm 7200 & P1=$!
run d2_c256 d2_s1_39b370e4.qasm 7200 & P2=$!
run d3_c256 d3_s1_2674779a.qasm 10800 & P3=$!

echo "P1=$P1 P2=$P2 P3=$P3"
wait $P1 $P2 $P3
echo "=== $(date +%T) wave3 COMPLETE ==="
for tag in d1_c256 d2_c256 d3_c256; do
  echo "----- $tag tail -----"
  tail -12 "/tmp/$tag.log"
done
