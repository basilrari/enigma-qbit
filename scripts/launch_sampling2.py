import os
import paramiko

HOST, USER, PASS = os.environ.get("SRVPRO_PASS", ""), "basilrari", "iloveesl"
VER = "/mnt/8tb_hdd2/basilrari/enigma-work/verify"
PY = "/mnt/8tb_hdd2/basilrari/enigma-work/l2venv/bin/python"

JOBS = [
    ("d2_s1_39b370e4", 128, 200000),
    ("d2_s1_39b370e4", 256, 200000),
    ("d3_s1_2674779a", 128, 200000),
    ("d3_s1_2674779a", 256, 200000),
]

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, username=USER, password=PASS)

# build a launcher script ON the box, then run it detached
sh = ["#!/bin/bash", f"cd {VER}",
      "export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1"]
for cid, chi, ns in JOBS:
    tag = f"{cid}_c{chi}"
    sh.append(f"setsid nohup {PY} sample_probe.py {cid} {chi} {ns} 4000 "
              f"> {VER}/sample_{tag}.log 2>&1 < /dev/null &")
sh.append("sleep 2")
sh.append("ps aux | grep [s]ample_probe | wc -l")
sf = c.open_sftp()
with sf.open(f"{VER}/_launch_sampling.sh", "w") as f:
    f.write("\n".join(sh) + "\n")
sf.close()

i, o, e = c.exec_command(f"bash {VER}/_launch_sampling.sh", timeout=60)
o.channel.recv_exit_status()
print("live sample_probe procs:", o.read().decode().strip())
c.close()
