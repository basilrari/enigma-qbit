import os
import paramiko

HOST, USER, PASS = os.environ.get("SRVPRO_PASS", ""), "basilrari", "iloveesl"
VER = "/mnt/8tb_hdd2/basilrari/enigma-work/verify"
PY = "/mnt/8tb_hdd2/basilrari/enigma-work/l2venv/bin/python"

PEAK = {
    "d2_s1_39b370e4": "1110101100010111001000000011101111001000",
    "d3_s1_2674779a": "011100110011000101111001110011100000101101100100",
}

# Full pipeline: heavy-hitter sampling -> refine() each seed -> certificate.
JOBS = [
    ("d2_s1_39b370e4", 256),
    ("d3_s1_2674779a", 256),
]

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, username=USER, password=PASS)
sh = ["#!/bin/bash", f"cd {VER}",
      "export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1"]
for cid, chi in JOBS:
    sh.append(f"setsid nohup {PY} hqp_solver.py --qasm {cid}.qasm --chi {chi} "
              f"--samples 40000 --hh-top 24 --restarts 6 --budget 18000 "
              f"--peak {PEAK[cid]} "
              f"--save-tensors {VER}/mps_{cid}_c{chi}.npz "
              f"--out {VER}/full_{cid}_c{chi}.json "
              f"> {VER}/full_{cid}_c{chi}.log 2>&1 < /dev/null &")
sh.append("sleep 3; ps aux | grep -c [h]qp_solver")
sf = c.open_sftp()
with sf.open(f"{VER}/_launch_full.sh", "w") as f:
    f.write("\n".join(sh) + "\n")
sf.close()
i, o, e = c.exec_command(f"bash {VER}/_launch_full.sh", timeout=60)
o.channel.recv_exit_status()
print("live hqp_solver procs:", o.read().decode().strip())
c.close()
