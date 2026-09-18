#!/usr/bin/env python3
"""Validate exact_extract.topk_exact against brute force, and poll the L2 runs."""
import os
import paramiko

PW = open(os.path.expanduser("~/.srvpro_env")).read().split("'")[1]
W = "/mnt/8tb_hdd2/basilrari/enigma-work"
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("140.123.105.18", username="basilrari", password=PW)
s = c.open_sftp()
s.put("/home/basilsclaw/enigma-solve/scripts/exact_extract.py", f"{W}/exact_extract.py")

test = '''
import numpy as np, random, sys
sys.path.insert(0, "__W__")
import quimb.tensor as qtn
from exact_extract import topk_exact, as_mps

random.seed(7); np.random.seed(7)
n, D = 10, 20
dims = [1] + [min(D, 2 ** i, 2 ** (n - i)) for i in range(1, n)] + [1]
tens = [np.random.randn(dims[i], 2, dims[i + 1]) for i in range(n)]
mps = qtn.MatrixProductState(tens, site_ind_id="k{}")
dense = np.asarray(mps.to_dense()).ravel()
probs = np.abs(dense) ** 2
probs = probs / probs.sum()
idx = np.argsort(probs)[::-1][:5]
print("brute force top5 (site0 = MSB of the string):")
for j in idx:
    print("   P=%.6e  %s" % (probs[j], format(int(j), "0%db" % n)))

res = topk_exact(mps, k=5, cap=200000)
print("exact_extract top5:")
ok = True
for r, (bits, p) in enumerate(res):
    j = idx[r]
    b = format(int(j), "0%db" % n)
    match = (bits == b) and abs(p - probs[j]) < 1e-9 * max(1e-30, probs[j])
    ok &= match
    print("   P=%.6e  %s   %s" % (p, bits, "OK" if match else "MISMATCH vs " + b))
print("SELFTEST:", "PASS" if ok else "FAIL")
'''.replace("__W__", W)
with open("/tmp/selftest_extract.py", "w") as fh:
    fh.write(test)
s.put("/tmp/selftest_extract.py", f"{W}/selftest_extract.py")
s.close()

_i, o, e = c.exec_command(f"cd {W} && {W}/l2venv/bin/python {W}/selftest_extract.py 2>&1 | "
                          f"grep -vE 'FutureWarning|tensor_network_ag_compress|warnings.warn'", timeout=500)
o.channel.recv_exit_status()
print(o.read().decode() + e.read().decode())

_i, o, e = c.exec_command(
    "for f in d1_s1_4043cafb d2_s1_39b370e4 d3_s1_2674779a; do echo -n \"$f: \"; "
    f"grep -aE 'absorb:|state:|###|Traceback' {W}/logs/l2_$f.log | tail -3 | tr '\\n' ' '; "
    f"echo \\\" [$(tail -c 300 {W}/logs/l2_$f.log | tr -d '\\n' | tail -c 120)]\\\"; done", timeout=120)
o.channel.recv_exit_status()
print(o.read().decode() + e.read().decode())
c.close()
