#!/bin/bash
# Wave 5 (Mon 09:30): d2/d3 back to chi=256 (their H=18/27 local maxima are
# truncation artifacts at chi=128 — 100-200x above the true peak). d1 re-verified
# at chi=256 (already SOLVED H=0 at chi=128). Single-thread BLAS per job.
# Search upgrades: seed from known peak + multi-scale restarts (4/8/16-bit) x24.
cd /mnt/8tb_hdd2/basilrari/enigma-work/verify || exit 1
PY=/mnt/8tb_hdd2/basilrari/enigma-work/l2venv/bin/python
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

run() { # tag qasm chi budget
  local tag=$1 qasm=$2 chi=$3 budget=$4
  echo "=== $(date +%T) START $tag chi=$chi budget=${budget}s ==="
  "$PY" hqp_solver.py --qasm "$qasm" --chi "$chi" --budget "$budget" --restarts 24 --out "/tmp/$tag.json" \
      > "/tmp/${tag}.log" 2>&1
  echo "=== $(date +%T) DONE $tag exit=$? ==="
  grep -E "VERDICT|ANSWER|H=0" "/tmp/${tag}.log" | tail -4
}

run d1_c256 d1_s1_4043cafb.qasm 256 18000 & P1=$!
run d2_c256 d2_s1_39b370e4.qasm 256 18000 & P2=$!
run d3_c256 d3_s1_2674779a.qasm 256 18000 & P3=$!
echo "launched d1=$P1 d2=$P2 d3=$P3 at $(date +%T) — waiting"
wait $P1 $P2 $P3
echo "=== $(date +%T) ALL DONE ==="
