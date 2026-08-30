#!/bin/bash
cd /mnt/8tb_hdd2/basilrari/enigma-work/verify || exit 1
PY=/mnt/8tb_hdd2/basilrari/enigma-work/l2venv/bin/python
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
echo "wave4 start $(date) load: $(uptime | sed 's/.*load/load')"
run() {
  local tag=$1 qasm=$2 chi=$3 budget=$4
  echo "=== $(date +%T) START $tag chi=$chi budget=${budget}s ==="
  "$PY" hqp_solver.py --qasm "$qasm" --chi "$chi" --budget "$budget" --out "/tmp/$tag.json" > "/tmp/$tag.log" 2>&1
  echo "=== $(date +%T) DONE $tag exit=$? ==="
  tail -16 "/tmp/$tag.log" | sed "s|^|  [$tag] |"
}
run d1_c128 d1_s1_4043cafb.qasm 128 14400 & P1=$!
run d2_c128 d2_s1_39b370e4.qasm 128 14400 & P2=$!
run d3_c128 d3_s1_2674779a.qasm 128 14400 & P3=$!
echo "P1=$P1 P2=$P2 P3=$P3"
wait $P1 $P2 $P3
echo "=== $(date +%T) wave4 COMPLETE ==="
for tag in d1_c128 d2_c128 d3_c128; do
  echo "----- $tag -----"; tail -10 "/tmp/$tag.log"
done
