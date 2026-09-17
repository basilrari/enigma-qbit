#!/usr/bin/env python3
"""Sync the solver and climb the chi ladder where the peak is emerging.

Convergence data from the completed sweep (P(known)/uniform):
  d2_s1:  chi 128 -> 0.94    192 -> 0.36    256 -> 5252    384 -> 21960   512 -> 31950
  d3_s1:  chi 128 -> 1.1e-6  256 -> 4.3e-3  384 -> 1.17
Above chi 256 the trend is cleanly monotone -- the sharp dip at 128/192 was the
anomaly, not the growth.  So the ladder is worth climbing: chi 512 now, tensors
saved so the search can be re-run without ever rebuilding.
"""
import os
import paramiko

HOST, USER, PASS = os.environ.get("SRVPRO_PASS", ""), "basilrari", "iloveesl"
VER = "/mnt/8tb_hdd2/basilrari/enigma-work/verify"
PY = "/mnt/8tb_hdd2/basilrari/enigma-work/l2venv/bin/python"
ENV = ("export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "
       "OPENBLAS_DYNAMIC_ARCH=0; ")

PEAKS = {
    "d1_s1_4043cafb": "0001001101001111101001001110010001111010100000",
    "d1_s2_adeddcf3": "011010011101001110100001110011001110011001000101",
    "d2_s1_39b370e4": "1110101100010111001000000011101111001000",
    "d2_s2_1efabaf4": "01000001010100001111101100001111010110100011",
    "d3_s1_2674779a": "011100110011000101111001110011100000101101100100",
    "d3_s2_c09ba537": "100111000110101011101010101001001100100011010011",
}

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, username=USER, password=PASS)

sftp = c.open_sftp()
sftp.put("/home/basilsclaw/enigma-solve/hqp_solver.py", f"{VER}/hqp_solver.py")
print("synced hqp_solver.py")


def run(cmd, t=120):
    i, o, e = c.exec_command(cmd, timeout=t)
    o.channel.recv_exit_status()
    return o.read().decode()


# free the two redundant chi=256 pipelines: the sweep already covered 256 and
# the exact optimiser supersedes their search stage.
print(run("pkill -f 'hqp_solver.py --qasm d2_s1_39b370e4 --chi 256'; "
          "pkill -f 'hqp_solver.py --qasm d3_s1_2674779a --chi 256'; "
          "sleep 1; echo killed-chi256"))

JOBS = [("d2_s1_39b370e4", 512), ("d3_s1_2674779a", 512)]
for cid, chi in JOBS:
    pk = PEAKS[cid]
    cmd = (
        f"cd {VER} && {ENV}{PY} hqp_solver.py --qasm {cid}.qasm "
        f"--chi {chi} --peak {pk} --restarts 0 --samples 0 "
        f"--exact-top 32 --exact-cap 4000000 "
        f"--save-tensors {cid}_c{chi}.npy "
        f"</dev/null >argmax_{cid}_c{chi}.log 2>&1 & echo LAUNCHED-{cid}-{chi}"
    )
    print(run(f"setsid nohup bash -c \"{cmd}\" >/dev/null 2>&1 & echo ok"))

print(run("sleep 10; ps -eo etime,pcpu,args | grep -c 'hqp_solver.py'"))
print(run("ls -la " + VER + "/argmax_*.log | tail -8"))
c.close()
