#!/usr/bin/env python3
"""Validate the whole L2 chain on a synthetic instance with a KNOWN answer.

Construction: a random block U, then U^dagger (so the block is the identity),
then X on every qubit. The output is therefore |1...1>, in ANY bit-order
convention (all-ones is reversal-invariant) -- so this test cannot be fooled by
an endianness mistake, which is exactly the trap that has bitten this work twice.

If the rebuilt orchestrator + extractor are correct, this must print
    amp2(truth) ~ 1.0, H=0, and the beam/A* must return the all-ones string.
"""
import json
import os
import random

import paramiko

W = "/mnt/8tb_hdd2/basilrari/enigma-work"
V = f"{W}/verify"
N = 12
LAYERS = 5
INST = "known_12q"
TRUTH = "1" * N


def emit(path):
    rng = random.Random(20260918)
    gates = []
    for _ in range(LAYERS):
        for q in range(N):
            gates.append(("u3", q, rng.uniform(0, 6.28), rng.uniform(0, 6.28), rng.uniform(0, 6.28)))
        for q in range(N - 1):
            gates.append(("cz", q, q + 1))
    inv = []
    for g in reversed(gates):
        if g[0] == "u3":
            inv.append(("u3", g[1], -g[2], -g[4], -g[3]))  # u3(t,p,l)^dag = u3(-t,-l,-p)
        else:
            inv.append(g)
    tail = [("x", q) for q in range(N)]
    lines = ['OPENQASM 2.0;', 'include "qelib1.inc";', f"qreg q[{N}];", f"creg c[{N}];"]
    for g in gates + inv + tail:
        if g[0] == "u3":
            lines.append(f"u3({g[2]:.12f},{g[3]:.12f},{g[4]:.12f}) q[{g[1]}];")
        elif g[0] == "cz":
            lines.append(f"cz q[{g[1]}],q[{g[2]}];")
        else:
            lines.append(f"x q[{g[1]}];")
    lines += [f"measure q[{i}] -> c[{i}];" for i in range(N)]
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return len(gates)


ngates = emit(f"/home/basilsclaw/enigma-solve/samples/{INST}.qasm")
print(f"generated {INST}: {N}q, block {ngates} gates x2 + {N} X gates; truth={TRUTH}")

pw = open(os.path.expanduser("~/.srvpro_env")).read().split("'")[1]
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("140.123.105.18", username="basilrari", password=pw)
s = c.open_sftp()
s.put(f"/home/basilsclaw/enigma-solve/samples/{INST}.qasm", f"{V}/{INST}.qasm")
with s.open(f"{V}/{INST}_meta.json", "w") as fh:
    fh.write(json.dumps({"name": INST, "known_bitstring": TRUTH, "qubits": N, "n_qubits": N}))
s.put("/home/basilsclaw/enigma-solve/scripts/l2_absorb.py", f"{W}/l2_absorb.py")
for extra in ("exact_extract.py",):
    p = f"/home/basilsclaw/enigma-solve/scripts/{extra}"
    if os.path.exists(p):
        s.put(p, f"{W}/{extra}")
s.close()
print("uploaded qasm + meta + l2_absorb.py + exact_extract.py")
c.close()
