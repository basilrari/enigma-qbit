#!/usr/bin/env python3
"""test_phys.py — verify MPS physics vs full statevector on a small circuit."""
import os, sys
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
os.environ.setdefault("OMP_NUM_THREADS","1")
os.environ.setdefault("MKL_NUM_THREADS","1")
sys.path.insert(0, "/home/basilsclaw/enigma-solve")
import numpy as np
import hqp_solver as H

# small 5-qubit circuit with CZ gates (exercises swaps) + u gates
qasm = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
u(0.3,0.7,1.1) q[0];
u(1.9,0.2,0.8) q[1];
u(0.5,2.1,0.4) q[2];
u(2.4,0.6,1.5) q[3];
u(1.2,0.9,0.1) q[4];
cz q[0],q[3];
u(0.7,1.4,2.0) q[1];
u(1.8,0.3,0.9) q[2];
cz q[2],q[4];
u(0.4,1.7,0.2) q[0];
u(1.1,0.5,1.9) q[3];
cz q[1],q[4];
u(0.9,1.2,0.7) q[2];
u(2.2,0.8,1.3) q[4];
"""
open("/tmp/test_circ.qasm","w").write(qasm)
circ = H.load_circuit("/tmp/test_circ.qasm")
n = circ.num_qubits
print(f"circuit: {n} qubits")

# 1) full statevector (ground truth)
from qiskit import QuantumCircuit, Statevector
qc = QuantumCircuit(n)
for inst in circ.data:
    op = inst.operation
    qs = [circ.find_bit(q).index for q in inst.qubits]
    if op.name == 'u':
        from qiskit.circuit.library import UGate
        qc.append(UGate(*op.params), [qs[0]])
    elif op.name == 'cz':
        qc.cz(qs[0], qs[1])
    else:
        qc.append(op, qs)
sv = Statevector(qc)
ampl = sv.data
# argmax in QISKIT order (q0 = least significant bit)
peak_idx = int(np.argmax(np.abs(ampl)**2))
sv_peak = format(peak_idx, f'0{n}b')[::-1]  # reverse to q0=leftmost char
print(f"statevector peak (q0=leftmost): {sv_peak}  P={np.abs(ampl[peak_idx])**2:.4e}")

# 2) MPS oracle
mps = H.MPS(*H.build_mps(circ, 16, verbose=False))
# the oracle returns bitstring in ORIGINAL qubit order (q0 leftmost char)
# compare: enumerate a few, but for a peaked circuit the MPS P(argmax) should
# match the sv peak. Since we can't argmax the MPS easily, instead: check that
# the MPS P(sv_peak) is high AND that a couple of neighbors are lower.
p_svpeak = mps.P(sv_peak)
print(f"MPS P(sv_peak) = {p_svpeak:.4e}")
# if the MPS is correct, P(sv_peak) should be close to the sv peak P (for chi>=2^depth)
# and the peak should be unique (neighbors much lower)
Pn = mps.P_single_flip_all(sv_peak)
print(f"MPS best single-flip neighbor P = {Pn.max():.4e} (ratio {p_svpeak/max(Pn.max(),1e-300):.1f}x)")

# 3) run the full search and see if it recovers the peak
class A: pass
a = A(); a.K=3; a.top=64; a.restarts=0
best, bestP = H.refine(mps, '0'*n, a)
H.finalize_output(mps, best, bestP, "test", 0, a, "/tmp/test_phys.json", deadline=None)
print(f"\nFULL PIPELINE peak: {best}")
print(f"statevector peak  : {sv_peak}")
print(f"MATCH: {best == sv_peak}")
