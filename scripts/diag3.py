#!/usr/bin/env python3
"""diag3: separate MPS-truncation error from sampler error.

For one random 6-qubit circuit, compare against the exact statevector:
  (1) TVD of  mps.P(s)          -> is the MPS itself faithful?
  (2) TVD of  sampled histogram  -> is the sampler faithful to the MPS?
  (3) per-position conditional P(b_k=0 | greedy prefix) sampler vs exact
      -> localises any sampler bias.
"""
import sys, os, math, random
import numpy as np

VER = "/mnt/8tb_hdd2/basilrari/enigma-work/verify"
sys.path.insert(0, VER)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hqp_solver as HS
from sample_probe import suffix_norms, sample_batch

n = 6
NG = 40
np.random.seed(7)
random.seed(7)
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
tmp = "/tmp/_d3_circ.qasm"
open(tmp, "w").write("\n".join(lines) + "\n")

psi = np.zeros(1 << n, dtype=complex)
psi[0] = 1.0
for kind, qs, prm in gates:
    if kind == "u":
        (q,) = qs
        t, p, l = prm
        c, s = math.cos(t / 2), math.sin(t / 2)
        U = np.array([[c, -np.exp(1j * l) * s],
                      [np.exp(1j * p) * s, np.exp(1j * (p + l)) * c]])
        bp = n - 1 - q
        P = np.moveaxis(psi.reshape([2] * n), bp, 0)
        P = np.tensordot(U, P, axes=([1], [0]))
        psi = np.moveaxis(P, 0, bp).reshape(-1)
    else:
        a, b = qs
        idx = np.arange(1 << n)
        ia = (idx >> a) & 1
        ib = (idx >> b) & 1
        psi = psi * np.where((ia == 1) & (ib == 1), -1.0, 1.0)
prob = np.abs(psi) ** 2
prob /= prob.sum()
# little-endian string for statevector index v
strs = ["".join(str((v >> q) & 1) for q in range(n)) for v in range(1 << n)]

circ = HS.load_circuit(tmp)
for chi in (32, 64):
    T, order, q_at = HS.build_mps(circ, chi, verbose=False)
    mps = HS.MPS(T, q_at)
    P = np.array([mps.P(s) for s in strs])
    Pn = P / P.sum()
    print(f"\nchi={chi} maxbond={max(t.shape[2] for t in T)}")
    print(f"  (1) TVD(mps.P, exact) = {0.5*np.sum(np.abs(Pn-prob)):.6f}")
    R = suffix_norms(T)
    rng = np.random.default_rng(3)
    N = 200000
    emp = np.zeros(1 << n)
    done = 0
    while done < N:
        nb = min(5000, N - done)
        bits = sample_batch(T, R, n, nb, rng)
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
    print(f"  (2) TVD(sampled, exact) = {0.5*np.sum(np.abs(emp-prob)):.6f}   "
          f"TVD(sampled, mps) = {0.5*np.sum(np.abs(emp-Pn)):.6f}")

# (3) conditional check on the greedy path
T, order, q_at = HS.build_mps(circ, 64, verbose=False)
R = suffix_norms(T)
print("\n(3) per-position conditional P(0 | prefix), sampler(MPS) vs exact:")
V = np.ones((1, 1), dtype=complex)
prefix = []
worst = 0.0
for k in range(n):
    Tk = T[k]
    A0 = V @ Tk[:, 0, :]
    A1 = V @ Tk[:, 1, :]
    p0 = (A0 @ R[k + 1] @ A0.conj().T)[0, 0].real
    p1 = (A1 @ R[k + 1] @ A1.conj().T)[0, 0].real
    q = q_at[k]
    e0 = e1 = 0.0
    for v in range(1 << n):
        s = strs[v]
        if [int(s[q_at[j]]) for j in range(len(prefix))] != prefix:
            continue
        if s[q] == '0':
            e0 += prob[v]
        else:
            e1 += prob[v]
    m0 = p0 / max(p0 + p1, 1e-30)
    x0 = e0 / max(e0 + e1, 1e-30)
    worst = max(worst, abs(m0 - x0))
    print(f"   pos {k} (qubit {q}): sampler P(0)={m0:.6f}  exact P(0)={x0:.6f}  d={abs(m0-x0):.2e}")
    b = 0 if p0 > p1 else 1
    prefix.append(b)
    V = A0 if b == 0 else A1
print(f"  worst |delta| = {worst:.3e}")
