import os
import paramiko

HOST, USER, PASS = os.environ.get("SRVPRO_PASS", ""), "basilrari", "iloveesl"
VER = "/mnt/8tb_hdd2/basilrari/enigma-work/verify"
PY = "/mnt/8tb_hdd2/basilrari/enigma-work/l2venv/bin/python"
PEAK1 = "0001001101001111101001001110010001111010100000"

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, username=USER, password=PASS)
sf = c.open_sftp()
# upload the newly modified solver
sf.put("/home/basilsclaw/enigma-solve/hqp_solver.py", f"{VER}/hqp_solver.py")
sh = ["#!/bin/bash", f"cd {VER}",
      "export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1",
      f"setsid nohup {PY} hqp_solver.py --qasm d1_s1_4043cafb.qasm --chi 64 "
      f"--samples 20000 --hh-top 8 --restarts 4 --budget 2400 "
      f"--peak {PEAK1} --out {VER}/smoke_d1.json "
      f"> {VER}/smoke_d1.log 2>&1 < /dev/null &",
      "sleep 2; echo LAUNCHED"]
with sf.open(f"{VER}/_smoke.sh", "w") as f:
    f.write("\n".join(sh) + "\n")
sf.close()
i, o, e = c.exec_command(f"bash {VER}/_smoke.sh", timeout=60)
o.channel.recv_exit_status()
print(o.read().decode().strip(), e.read().decode()[-300:])
c.close()
