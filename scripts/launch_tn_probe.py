import os
import paramiko, sys

HOST, USER, PASS = os.environ.get("SRVPRO_PASS", ""), "basilrari", "iloveesl"
REMOTE = "/mnt/8tb_hdd2/basilrari/enigma-work/verify/tn_width_probe.py"
PY = "/mnt/8tb_hdd2/basilrari/enigma-work/l2venv/bin/python"
LOG = "/mnt/8tb_hdd2/basilrari/enigma-work/verify/tn_width.log"

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, username=USER, password=PASS, timeout=30)

cmd = (
    "cd /mnt/8tb_hdd2/basilrari/enigma-work/verify && "
    "export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 && "
    f"nohup {PY} -u {REMOTE} d0_s0_trivial d1_s1_4043cafb d2_s1_39b370e4 d3_s1_2674779a "
    f"> {LOG} 2>&1 & echo LAUNCHED $!"
)
_, o, e = c.exec_command(cmd, timeout=60)
print(o.read().decode().strip())
c.close()
