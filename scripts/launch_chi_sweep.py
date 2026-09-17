import paramiko

HOST, USER, PASS = "140.123.105.18", "basilrari", "iloveesl"
VER = "/mnt/8tb_hdd2/basilrari/enigma-work/verify"
PY = "/mnt/8tb_hdd2/basilrari/enigma-work/l2venv/bin/python"

# chi-sweep: how does the MPS probability of the KNOWN peak scale with bond
# dimension?  This is the measurement that decides whether "more chi" can
# ever work, and what chi would be needed.
JOBS = [
    ("d2_s1_39b370e4", 64, 20000),
    ("d2_s1_39b370e4", 192, 20000),
    ("d2_s1_39b370e4", 384, 20000),
    ("d2_s1_39b370e4", 512, 20000),
    ("d3_s1_2674779a", 384, 20000),
    ("d3_s1_2674779a", 512, 20000),
]

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, username=USER, password=PASS)
sh = ["#!/bin/bash", f"cd {VER}",
      "export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1"]
for cid, chi, ns in JOBS:
    tag = f"{cid}_c{chi}"
    sh.append(f"setsid nohup {PY} sample_probe.py {cid} {chi} {ns} 4000 "
              f"> {VER}/sample_{tag}.log 2>&1 < /dev/null &")
sh.append("sleep 3; ps aux | grep [s]ample_probe | wc -l")
sf = c.open_sftp()
with sf.open(f"{VER}/_launch_sweep.sh", "w") as f:
    f.write("\n".join(sh) + "\n")
sf.close()
i, o, e = c.exec_command(f"bash {VER}/_launch_sweep.sh", timeout=60)
o.channel.recv_exit_status()
print("live sample_probe procs now:", o.read().decode().strip())
c.close()
