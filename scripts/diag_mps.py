#!/usr/bin/env python3
"""Isolate the bug: is the MPS itself wrong, or is the sampler wrong?

Compares exact |<s|psi>|^2 (brute force) against mps.P(s) for ALL 2^n
bitstrings on a small random circuit.  Then compares the sampler's
conditional probabilities against the exact conditionals.
"""
import sys, os, math, random
import numpy as np

VER = "/mnt/8tb_hdd2/basilrari/enigma-work/verify"
sys.path.insert(0, VER)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hqp_solver as HS
from sample_probe import suffix_norms, sample_batch

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
tmp = "/tmp/_diag_circ.qasm"
open(tmp, "w").write("\n".join(lines) + "\n")

# brute force
psi = np.zeros(1 << n, dtype=complex)
psi[0] = 1.0
for kind, qs, prm in gates:
    if kind == "u":
        (q,) = qs
        t, p, l = prm
        c, s = math.cos(t / 2), math.sin(t / 2)
        U = np.array([[c, -np.exp(1j * l) * s],
                      [np.exp(1j * p) * s, np.exp(1j * (p + l)) * c]])
        bitpos = n - 1 - q
        P = np.moveaxis(psi.reshape([2] * n), bitpos, 0)
        P = np.tensordot(U, P, axes=([1], [0]))
        psi = np.moveaxis(P, 0, bitpos).reshape(-1)
    else:
        a, b = qs
        idx = np.arange(1 << n)
        ia = (idx >> (n - 1 - a)) & 1
        ib = (idx >> (n - 1 - b)) & 1
        psi = psi * np.where((ia == 1) & (ib == 1), -1.0, 1.0)
prob = np.abs(psi) ** 2
prob /= prob.sum()

circ = HS.load_circuit(tmp)
for chi in (32, 8):
    tensors, order, q_at = HS.build_mps(circ, chi, verbose=False)
    mps = HS.MPS(tensors, q_at)
    bond = max(t.shape[2] for t in tensors)
    print(f"\n--- chi={chi}  maxbond={bond}  q_at={q_at} ---")
    P = np.array([mps.P(format(v, f'0{n}b')) for v in range(1 << n)])
    Pn = P / P.sum()
    print(f"  sum mps.P = {P.sum():.6f}")
    print(f"  max|Pn - exact| = {np.max(np.abs(Pn-prob)):.6f}")
    print(f"  TVD(mps, exact) = {0.5*np.sum(np.abs(Pn-prob)):.5f}")
    o = np.argsort(prob)[::-1][:6]
    for v in o:
        print(f"    {v:0{n}b}  exact={prob[v]:.5f}  mps={Pn[v]:.5f}  ratio={Pn[v]/max(prob[v],1e-12):8.3f}")

# ---- now check the SAMPLER's conditional probs against exact ----
tensors, order, q_at = HS.build_mps(circ, 32, verbose=False)
R = suffix_norms(tensors)
print("\n--- sampler conditional check (prefix = exact most likely first bits) ---")
V = np.ones((1, 1), dtype=complex)
prefix = []
for k in range(n):
    Tk = tensors[k]
    A0 = V @ Tk[:, 0, :]
    A1 = V @ Tk[:, 1, :]
    p0 = float(np.real(A0 @ R[k + 1] @ A0.conj().T))
    p1 = float(np.real(A1 @ R[k + 1] @ A1.conj().T))
    # exact conditional for the same prefix
    q = q_at[k]
    e0 = e1 = 0.0
    for v in range(1 << n):
        s = format(v, f'0{n}b')
        if [int(s[q_at[j]]) for j in range(len(prefix))] != prefix:
            continue
        if s[q] == '0':
            e0 += prob[v]
        else:
            e1 += prob[v]
    print(f"  pos {k} (qubit {q}): mps P(0)={p0/(p0+p1):.6f}  exact P(0)={e0/max(e0+e1,1e-12):.6f}")
    b = 0 if p0 > p1 else 1
    prefix.append(b)
    V = A0 if b == 0 else A1
