#!/usr/bin/env python3
"""Upload the fixed solver, prove save/load round-trips, then relaunch work."""
import os
import paramiko

HOST, USER, PASS = os.environ.get("SRVPRO_PASS", ""), "basilrari", "iloveesl"
VER = "/mnt/8tb_hdd2/basilrari/enigma-work/verify"
LOCAL = "/home/basilsclaw/enigma-solve"
PY = "/mnt/8tb_hdd2/basilrari/enigma-work/l2venv/bin/python"
ENV = ("export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "
       "NUMEXPR_NUM_THREADS=1 && ")

TEST = r'''
import sys, numpy as np
sys.path.insert(0, "/mnt/8tb_hdd2/basilrari/enigma-work/verify")
import hqp_solver as HS
rng = np.random.default_rng(3)
nq = 10
L = ["OPENQASM 2.0;", 'include "qelib1.inc";', f"qreg q[{nq}];", f"creg c[{nq}];"]
for d in range(4):
    for q in range(nq):
        L.append(f"u({rng.uniform(0,6.28)},{rng.uniform(0,6.28)},{rng.uniform(0,6.28)}) q[{q}];")
    for i in range(0, nq - 1, 2):
        L.append(f"cz q[{i}],q[{i+1}];")
open("/tmp/_sl.qasm", "w").write("\n".join(L) + "\n")
circ = HS.load_circuit("/tmp/_sl.qasm")
tensors, order, q_at = HS.build_mps(circ, 32, verbose=False)
HS.save_mps(tensors, order, q_at, list(q_at), None, "/tmp/_sl.npy")
m1 = HS.MPS(tensors, q_at, None, "orig")
m2, _ = HS.load_mps_tensors("/tmp/_sl.npy")
bad = 0
for x in range(64):
    s = format(x, f"0{nq}b")
    if abs(m1.amp(s) - m2.amp(s)) > 1e-12:
        bad += 1
print("ROUNDTRIP", "PASS" if bad == 0 else f"FAIL {bad}/64", flush=True)
'''

JOBS = [
    ("argmax_probe.py", "d1_s1_4043cafb", 64),
    ("argmax_probe.py", "d2_s1_39b370e4", 128),
    ("argmax_probe.py", "d2_s1_39b370e4", 192),
    ("argmax_probe.py", "d3_s1_2674779a", 128),
    ("argmax_probe.py", "d3_s1_2674779a", 192),
]

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, username=USER, password=PASS)
sf = c.open_sftp()
sf.put(f"{LOCAL}/hqp_solver.py", f"{VER}/hqp_solver.py")
sf.put(f"{LOCAL}/scripts/argmax_probe.py", f"{VER}/argmax_probe.py")
with sf.open(f"{VER}/_test_saveload.py", "w") as f:
    f.write(TEST)
sf.close()
print("uploaded hqp_solver.py + argmax_probe.py")

i, o, e = c.exec_command(f"cd {VER} && {ENV} {PY} _test_saveload.py", timeout=300)
o.channel.recv_exit_status()
out = o.read().decode()
print([l for l in out.splitlines() if "ROUNDTRIP" in l or "WARNING" in l])
err = e.read().decode().strip()
if err:
    print("STDERR:", err[-800:])

for name, cid, chi in JOBS:
    cmd = (f"cd {VER} && {ENV} setsid nohup {PY} {name} {cid} {chi} 8 "
           f"> argmax_{cid}_c{chi}.log 2>&1 < /dev/null & echo ok")
    i, o, e = c.exec_command(cmd, timeout=60)
    o.channel.recv_exit_status()
print("argmax jobs launched")

for cid in ("d2_s1_39b370e4", "d3_s1_2674779a"):
    cmd = (f"cd {VER} && {ENV} setsid nohup {PY} hqp_solver.py "
           f"--qasm {cid}.qasm --chi 256 --samples 40000 --hh-top 24 "
           f"--restarts 12 --budget 14400 --peak $(python3 -c \"import "
           f"json;print(json.load(open('_peaks.json'))['{cid}'])\") "
           f"--save-tensors {cid}_c256 --out full_{cid}_c256.json "
           f"> full_{cid}_c256.log 2>&1 < /dev/null & echo ok")
    i, o, e = c.exec_command(cmd, timeout=60)
    o.channel.recv_exit_status()
print("full-pipeline jobs relaunched")

i, o, e = c.exec_command("ps -eo pid,etime,pcpu,args --sort=-pcpu | grep "
                         "python3 | grep -v grep | head -12", timeout=60)
o.channel.recv_exit_status()
print(o.read().decode().strip()[:1500])
c.close()
