import sys
import numpy as np
src = open('hqp_solver.py').read()
head = src[:src.index('def main(')]
ns = {}
exec(head, ns)
MPS = ns['MPS']; load_circuit = ns['load_circuit']; build_mps = ns['build_mps']
def h(b): return int(b, 2)

# d0 trivial = product state of x gates. Exact state is known analytically.
circ = load_circuit('d0_s0_trivial.qasm')
nq = circ.num_qubits
# which input qubits get an x
xgates = set()
for inst in circ.data:
    if inst.operation.name == 'x':
        xgates.add(inst.qubits[0])
print("nqubits:", nq, "x-gates:", sorted(xgates), "count:", len(xgates))

T, order, qo = build_mps(circ, 16, verbose=False)
m = MPS(T, order, qo)

# Exact product state in OUTPUT ordering qo: bit k of the answer = 1 iff
# qo[k] (the input qubit at output position k) has an x gate.
ans_bits = [1 if qo[k] in xgates else 0 for k in range(nq)]
known = h(''.join(str(b) for b in ans_bits))
print("known (product state, output order):", bin(known)[2:].zfill(nq))

p1 = m.marginal_p1()
print("p1 range: [%.4f, %.4f]" % (p1.min(), p1.max()))
print("p1 is clean 0/1:", np.all((p1 < 0.01) | (p1 > 0.99)))

mar = h(''.join('1' if x > 0.5 else '0' for x in p1))
print("marginal bitstring:", bin(mar)[2:].zfill(nq))
print("marginal == known :", mar == known)
print("P(known) exact    :", m.P(known))
print("P(marginal)        :", m.P(mar))
print("num 1s in marginal:", bin(mar).count('1'), "(expect %d)" % len(xgates))
