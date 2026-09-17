#!/usr/bin/env python3
"""Validate the SVD-guard optimisation and collect the gap-curve results."""
import os
import time

import paramiko

V = "/mnt/8tb_hdd2/basilrari/enigma-work/verify"
PY = "/mnt/8tb_hdd2/basilrari/enigma-work/l2venv/bin/python"
PW = open(os.path.expanduser("~/.srvpro_env")).read().split("'")[1]

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("140.123.105.18", username="basilrari", password=PW)


def run(x, t=540):
    i, o, e = c.exec_command(x, timeout=t)
    o.channel.recv_exit_status()
    return o.read().decode()


s = c.open_sftp()
s.put("hqp_solver.py", f"{V}/hqp_solver.py")
s.close()
print("uploaded new solver")

# Exactness gate: same circuit, same chi, same command as the recorded baseline
# run, which produced ex#1 P=2.516320e-03 H=0.  Any change in the numerics
# will move that probability; identical to ~1e-12 means the edit is inert.
cmd = (f"cd {V} && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "
       f"NUMEXPR_NUM_THREADS=1 setsid nohup {PY} hqp_solver.py "
       f"--qasm d1_s1_4043cafb.qasm --chi 128 --exact-top 3 "
       f"> {V}/abnew_c128.log 2>&1 < /dev/null &")
c.exec_command(cmd)
print("launched new-engine validation on d1_s1 chi=128\n")

time.sleep(150)
print("--- new engine timing at gate 300 (old engine: dt=22.7s) ---")
print(run(f"grep -a 'gate 300/' {V}/abnew_c128.log | tail -1").strip() or "(not yet)")

print("\n--- gap curve so far ---")
for lg in ["gap_d2_c64", "gap_d2_c96", "gap_d2_c192", "gap_d2_c256",
           "gap_d2_c384", "gap_d3_c96", "gap_d3_c192"]:
    pat = r"P\(known\)|ex#|certified|ratio|H=[0-9]+/|uniform|VERDICT|ANSWER"
    body = run(f"cd {V} && grep -aE \"{pat}\" {lg}.log | tail -6")
    gate = run(f"cd {V} && grep -aoE 'gate [0-9]+/[0-9]+' {lg}.log | tail -1")
    print(f"[{lg}] {gate.strip()}")
    if body.strip():
        print("   " + body.strip().replace("\n", "\n   "))

print("\n--- jobs alive ---")
print(run("ps -eo etime,args | grep 'hqp_solver.py --qasm' | grep -v grep "
          "| wc -l").strip(), "solver jobs running")
c.close()
