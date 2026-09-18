#!/usr/bin/env python3
"""Read the tail of a remote log (arg: log path or name)."""
import os
import sys

import paramiko

W = "/mnt/8tb_hdd2/basilrari/enigma-work"
name = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 40

pw = open(os.path.expanduser("~/.srvpro_env")).read().split("'")[1]
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("140.123.105.18", username="basilrari", password=pw)
_i, o, e = c.exec_command(f"grep -avE 'Warning|warn' {W}/logs/{name} | tail -{n}", timeout=120)
o.channel.recv_exit_status()
print(o.read().decode() + e.read().decode())
c.close()
