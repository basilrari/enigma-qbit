#!/usr/bin/env bash
# run_test.sh — run the HQP solver on a sample circuit
# usage: run_test.sh <circuit.qasm> <chi> <K> <top> <restarts>
cd /home/basilsclaw/enigma-solve
CIRC="$1"
./venv/bin/python hqp_solver.py --qasm "$CIRC" \
  --chi "${2:-128}" --K "${3:-16}" --top "${4:-16384}" \
  --restarts "${5:-3}" --out "/tmp/hqp_$(basename $CIRC .qasm).json" 2>&1
