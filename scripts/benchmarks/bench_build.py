#!/usr/bin/env python3
"""bench_build.py — time the MPS build gate-by-gate to find the hot spot."""
import sys, time, importlib.util
import numpy as np

sys.path.insert(0, "/home/basilsclaw/enigma-solve")
import hqp_solver as H

path = sys.argv[1] if len(sys.argv) > 1 else "/home/basilsclaw/enigma-solve/samples/d1_s1_4043cafb.qasm"
chi = int(sys.argv[2]) if len(sys.argv) > 2 else 128

circ = H.load_circuit(path)
n = circ.num_qubits
n_cz = sum(1 for i in circ.data if len(i.qubits) == 2)
n_u = sum(1 for i in circ.data if len(i.qubits) == 1)
print(f"circuit: {n} qubits, {len(circ.data)} gates ({n_u} u-gates, {n_cz} cz)", flush=True)

tensors = [np.zeros((1, 2, 1), dtype=complex) for _ in range(n)]
for k in range(n):
    tensors[k][0, 0, 0] = 1.0
ops, order = H.reorder_qubits(circ)
print(f"reorder done, pos_of identity: {order[:6]}...", flush=True)

pos_of = list(range(n))
q_at = list(range(n))
center = n - 1
t0 = time.time()
ng = 0
swaps = 0
moves = 0
t_gate = time.time()
for name, qs, params in ops:
    if name == 'u':
        t, p, lam = params[0]/2, params[1], params[2]
        c, s = np.cos(t), np.sin(t)
        U = np.array([[c, -np.exp(1j*lam)*s],
                      [np.exp(1j*p)*s, np.exp(1j*(p+lam))*c]], dtype=complex)
        i = pos_of[qs[0]]
        tensors[i] = np.einsum('ab,lbr->lar', U, tensors[i])
    elif name == 'cz':
        a, b = qs
        mat = np.diag([1, 1, 1, -1]).astype(complex)
        i, j = pos_of[a], pos_of[b]
        if i > j:
            i, j = j, i
            a, b = b, a
        while pos_of[b] > pos_of[a] + 1:
            p2 = pos_of[b]
            while center > p2 - 1:
                H._move_center_left(tensors, center); center -= 1; moves += 1
            while center < p2 - 1:
                H._move_center_right(tensors, center); center += 1; moves += 1
            center = H._apply_swap(tensors, p2-1, chi)
            q_at[p2-1], q_at[p2] = q_at[p2], q_at[p2-1]
            pos_of[q_at[p2-1]], pos_of[q_at[p2]] = p2-1, p2
            swaps += 1
        i, j = pos_of[a], pos_of[b]
        while center > i:
            H._move_center_left(tensors, center); center -= 1; moves += 1
        while center < i:
            H._move_center_right(tensors, center); center += 1; moves += 1
        center = H._apply_2site(tensors, i, mat, chi)
    ng += 1
    if ng % 100 == 0:
        dt = time.time() - t_gate
        t_gate = time.time()
        print(f"  gate {ng}/{len(ops)}: {dt:6.1f}s for 100  (swaps={swaps} moves={moves}) "
              f"total={time.time()-t0:.0f}s", flush=True)
print(f"BUILD DONE: {ng} gates in {time.time()-t0:.1f}s | swaps={swaps} moves={moves} chi={chi}", flush=True)

# quick oracle sanity: P of all-zeros
mps = H.MPS(tensors, [order[q] for q in q_at])
p0 = mps.P('0' * n)
print(f"P(0...0) = {p0:.3e}", flush=True)
