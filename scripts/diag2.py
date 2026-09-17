#!/usr/bin/env python3
"""Minimal systematic test of build_mps vs exact statevector.

Test 1: single-qubit gates only            (no swaps, no 2-site)
Test 2: one ADJACENT cz                    (2-site, no swap)
Test 3: one DISTANT cz                     (forces swaps)
Also prints the singular-value spectrum seen by _svd for each 2-site step.
"""
import sys, os, math, random
import numpy as np

VER = "/mnt/8tb_hdd2/basilrari/enigma-work/verify"
sys.path.insert(0, VER)
import hqp_solver as HS


def brute(n, gates_dir):
    psi = np.zeros(1 << n, dtype=complex)
    psi[0] = 1.0
    for kind, qs, prm in gates_dir:
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
            ia = (idx >> (n - 1 - a)) & 1
            ib = (idx >> (n - 1 - b)) & 1
            psi = psi * np.where((ia == 1) & (ib == 1), -1.0, 1.0)
    pr = np.abs(psi) ** 2
    return pr / pr.sum()


def run(name, n, gd):
    lines = ["OPENQASM 2.0;", 'include "qelib1.inc";', f"qreg q[{n}];", f"creg c[{n}];"]
    for kind, qs, prm in gd:
        if kind == "u":
            lines.append(f"u({prm[0]!r},{prm[1]!r},{prm[2]!r}) q[{qs[0]}];")
        else:
            lines.append(f"cz q[{qs[0]}],q[{qs[1]}];")
    tmp = f"/tmp/_t_{name}.qasm"
    open(tmp, "w").write("\n".join(lines) + "\n")
    exact = brute(n, gd)
    circ = HS.load_circuit(tmp)
    T, order, q_at = HS.build_mps(circ, 64, verbose=False)
    mps = HS.MPS(T, q_at)
    # little-endian: statevector index bit q == qubit q's bit
    strs = ["".join(str((v >> q) & 1) for q in range(n)) for v in range(1 << n)]
    P = np.array([mps.P(s) for s in strs])
    Pn = P / P.sum()
    tvd = 0.5 * np.sum(np.abs(Pn - exact))
    print(f"\n=== {name}: n={n} gates={len(gd)} maxbond={max(t.shape[2] for t in T)} q_at={q_at}")
    print(f"    sum mps.P={P.sum():.6f}   TVD = {tvd:.6f}   {'OK' if tvd < 1e-6 else '*** WRONG ***'}")
    if tvd >= 1e-6:
        print(f"    {'bits':>6} {'exact':>10} {'mps':>10}")
        for v in np.argsort(exact)[::-1][:6]:
            print(f"    {v:0{n}b} {exact[v]:10.6f} {Pn[v]:10.6f}")
    return tvd


random.seed(11)
np.random.seed(11)

# Test 1: single-qubit gates only
gd1 = []
for i in range(6):
    q = i % 3
    gd1.append(("u", (q,), (0.3 + 0.4 * i, -1.1 + 0.5 * i, 0.7 - 0.3 * i)))
run("1_1q_only_n3", 3, gd1)

# Test 2: adjacent cz
gd2 = gd1 + [("cz", (0, 1), None)]
run("2_adjacent_cz", 3, gd2)

# Test 3: distant cz  (forces a swap)
gd3 = gd1 + [("cz", (0, 2), None)]
run("3_distant_cz", 3, gd3)

# Test 4: exactly one u then one adjacent cz
gd4 = [("u", (0,), (0.7, -0.4, 1.3)), ("cz", (0, 1), None)]
run("4_u_then_cz", 3, gd4)

# Test 5: one u then distant cz
gd5 = [("u", (0,), (0.7, -0.4, 1.3)), ("cz", (0, 2), None)]
run("5_u_then_distant_cz", 3, gd5)

# Test 6: bigger random 5-qubit
random.seed(5)
gd6 = []
for _ in range(18):
    if random.random() < 0.5:
        gd6.append(("u", (random.randrange(5),),
                    (random.random() * math.pi,
                     (random.random() * 2 - 1) * math.pi,
                     (random.random() * 2 - 1) * math.pi)))
    else:
        a, b = random.sample(range(5), 2)
        gd6.append(("cz", (a, b), None))
run("6_rand_n5", 5, gd6)

# ---- singular-value spectrum of a 2-site SVD like _apply_2site does ----
print("\n\n=== singular value magnitudes seen by _svd_full (absolute cutoff 1e-13) ===")
for scale, label in [(1.0, "norm-1 state"), (1e-8, "norm-1e-8 (48q-like amplitudes)"),
                     (1e-15, "norm-1e-15")]:
    M = np.random.randn(64, 128) * scale + 1j * np.random.randn(64, 128) * scale
    U, S, Vh = HS._svd_full(M)
    print(f"  {label:34s} max_sv={S.max():10.3e}  n_returned={len(S):4d}  "
          f"min_sv={S.min():10.3e}")
