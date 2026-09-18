#!/usr/bin/env python3
"""Upload and run probe_mps_conv.py on the box."""
import os, sys
import paramiko

PW = open(os.path.expanduser("~/.srvpro_env")).read().split("'")[1]
W = "/mnt/8tb_hdd2/basilrari/enigma-work"
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("140.123.105.18", username="basilrari", password=PW)
s = c.open_sftp()
for f in ("l2_absorb.py", "probe_mps_conv.py"):
    s.put(f"/home/basilsclaw/enigma-solve/scripts/{f}", f"{W}/{f}")
s.close()
cmd = (f"cd {W}/verify && HQP_SABRE_TRIALS=1000 HQP_QUIET=1 timeout 280 "
       f"{W}/l2venv/bin/python {W}/probe_mps_conv.py 2>&1 | grep -vE 'FutureWarning|tensor_network_ag_compress' | tail -25")
i, o, e = c.exec_command(cmd, timeout=320)
o.channel.recv_exit_status()
print(o.read().decode() + e.read().decode())
c.close()
