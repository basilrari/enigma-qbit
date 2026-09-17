import os
import paramiko, sys, time

HOST, USER, PASS = os.environ.get("SRVPRO_PASS", ""), "basilrari", "iloveesl"
LOCAL = "/home/basilsclaw/enigma-solve/scripts/tn_width_probe.py"
REMOTE = "/mnt/8tb_hdd2/basilrari/enigma-work/verify/tn_width_probe.py"
PY = "/mnt/8tb_hdd2/basilrari/enigma-work/l2venv/bin/python"

targets = sys.argv[1:] or ["d1_s1_4043cafb"]

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, username=USER, password=PASS, timeout=30)
sftp = c.open_sftp()
sftp.put(LOCAL, REMOTE)
sftp.close()

cmd = (
    "cd /mnt/8tb_hdd2/basilrari/enigma-work/verify && "
    "export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 && "
    f"{PY} -u {REMOTE} {' '.join(targets)} 2>&1"
)
_, o, e = c.exec_command(cmd, timeout=1800)
for line in o:
    print(line.rstrip(), flush=True)
err = e.read().decode(errors="replace").strip()
if err:
    print("[stderr]", err[:2000])
c.close()
