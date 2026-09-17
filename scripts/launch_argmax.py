import os
import paramiko

HOST, USER, PASS = os.environ.get("SRVPRO_PASS", ""), "basilrari", "iloveesl"
VER = "/mnt/8tb_hdd2/basilrari/enigma-work/verify"
PY = "/mnt/8tb_hdd2/basilrari/enigma-work/l2venv/bin/python"
ENV = ("export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "
       "NUMEXPR_NUM_THREADS=1 && ")

JOBS = [
    ("d1_s1_4043cafb", 64),
    ("d2_s1_39b370e4", 128),
    ("d2_s1_39b370e4", 192),
    ("d3_s1_2674779a", 128),
    ("d3_s1_2674779a", 192),
]

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, username=USER, password=PASS)

for lf in ("full_d2_s1_39b370e4_c256.log", "full_d3_s1_2674779a_c256.log"):
    i, o, e = c.exec_command(
        f"tail -3 {VER}/{lf}; grep -c . {VER}/{lf}", timeout=60)
    o.channel.recv_exit_status()
    print(f"--- {lf} ---")
    print(o.read().decode().strip()[-600:])
    print()

n = 0
for cid, chi in JOBS:
    cmd = (f"cd {VER} && {ENV} setsid nohup {PY} argmax_probe.py "
           f"{cid} {chi} 8 > argmax_{cid}_c{chi}.log 2>&1 < /dev/null & echo OK")
    i, o, e = c.exec_command(cmd, timeout=60)
    o.channel.recv_exit_status()
    out = o.read().decode().strip()
    if "OK" in out:
        n += 1
    print(f"launched {cid} chi={chi} -> {out[:40]}")

i, o, e = c.exec_command(
    "ps -eo comm,pcpu | grep -c python3", timeout=60)
o.channel.recv_exit_status()
print("python3 procs now:", o.read().decode().strip())
c.close()
