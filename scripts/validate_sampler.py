#!/usr/bin/env python3
"""VALIDATE the MPS sampler against exact brute-force statevector.

Builds a small random circuit, computes the exact statevector |psi> by
naive matrix application, then samples from the MPS built by HS.build_mps
and checks the empirical distribution against |<s|psi>|^2.

If total-variation distance is small (~1/sqrt(N)) the sampler is correct.
"""
import sys, os, math, random
import numpy as np

VER = "/mnt/8tb_hdd2/basilrari/enigma-work/verify"
sys.path.insert(0, VER)
import hqp_solver as HS

np.random.seed(7)
random.seed(7)

n = 6
NG = 40
lines = ["OPENQASM 2.0;", 'include "qelib1.inc";', f"qreg q[{n}];", f"creg c[{n}];"]
gates = []
for _ in range(NG):
    if random.random() < 0.55:
        q = random.randrange(n)
        t, p, l = (random.random() * math.pi,
                   (random.random() * 2 - 1) * math.pi,
                   (random.random() * 2 - 1) * math.pi)
        gates.append(("u", (q,), (t, p, l)))
        lines.append(f"u({t!r},{p!r},{l!r}) q[{q}];")
    else:
        a, b = random.sample(range(n), 2)
        gates.append(("cz", (a, b), None))
        lines.append(f"cz q[{a}],q[{b}];")

tmp = "/tmp/_valid_circ.qasm"
open(tmp, "w").write("\n".join(lines) + "\n")

# ---- brute force statevector, ORIGINAL qubit order, q0 = leftmost bit ----
psi = np.zeros(1 << n, dtype=complex)
psi[0] = 1.0
CZ = np.diag([1, 1, 1, -1]).astype(complex)
for kind, qs, prm in gates:
    if kind == "u":
        (q,) = qs
        t, p, l = prm
        c, s = math.cos(t / 2), math.sin(t / 2)
        U = np.array([[c, -np.exp(1j * l) * s],
                      [np.exp(1j * p) * s, np.exp(1j * (p + l)) * c]])
        # q0 is the LEFTMOST (most significant) bit
        bitpos = n - 1 - q
        P = psi.reshape([2] * n)
        P = np.moveaxis(P, bitpos, 0)
        P = np.tensordot(U, P, axes=([1], [0]))
        P = np.moveaxis(P, 0, bitpos)
        psi = P.reshape(-1)
    else:
        a, b = qs
        ba, bb = n - 1 - a, n - 1 - b
        idx = np.arange(1 << n)
        ia = (idx >> ba) & 1
        ib = (idx >> bb) & 1
        psi = psi * np.where((ia == 1) & (ib == 1), -1.0, 1.0)

prob = np.abs(psi) ** 2
prob /= prob.sum()
print(f"brute force: n={n} gates={NG} sum|psi|^2={np.sum(np.abs(psi)**2):.6f}")

# ---- MPS + sampling ----
circ = HS.load_circuit(tmp)
print("loaded circuit qubits =", circ.num_qubits, "gates =", len(circ.data))
tensors, order, q_at = HS.build_mps(circ, 32, verbose=False)
print("maxbond =", max(t.shape[2] for t in tensors))

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sample_probe import suffix_norms, sample_batch

R = suffix_norms(tensors)
rng = np.random.default_rng(3)
N = 200000
emp = np.zeros(1 << n)
B = 5000
done = 0
while done < N:
    nb = min(B, N - done)
    bits = sample_batch(tensors, R, n, nb, rng)
    for row in bits:
        s = [0] * n
        for k in range(n):
            s[q_at[k]] = int(row[k])
        v = 0
        for q, b in enumerate(s):
            v |= b << q
        emp[v] += 1
    done += nb
emp /= emp.sum()

tv = 0.5 * np.sum(np.abs(emp - prob))
print(f"\nsamples={N}  TOTAL VARIATION DISTANCE = {tv:.5f}")
print(f"  (expected ~ sqrt(2^n/(pi*N)) = {math.sqrt((1<<n)/(math.pi*N)):.5f} for a perfect sampler)")
print(f"  max |emp - exact| = {np.max(np.abs(emp-prob)):.6f}")
top_exact = np.argsort(prob)[-8:][::-1]
print("\n  bits     exact      empirical")
for v in top_exact:
    s = "".join(str((v >> q) & 1) for q in range(n))
    print(f"  {s}  {prob[v]:.6f}  {emp[v]:.6f}")
print("\nVERDICT:", "SAMPLER CORRECT" if tv < 4 * math.sqrt((1 << n) / (math.pi * N)) else "SAMPLER WRONG")
