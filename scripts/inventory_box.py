#!/usr/bin/env python3
"""Inventory the Enigma toolchain on srvpro.

The dossier says the L2/L3 pipeline (unswap absorb + extract) and the decisive
buried-vs-discarded probes live in ~/enigma-work/ -- "ready, not launched".
Find out exactly what exists, what state it is in, and whether the decisive
diagnostic has ever been run.
"""
import os

import paramiko

PW = open(os.path.expanduser("~/.srvpro_env")).read().split("'")[1]
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("140.123.105.18", username="basilrari", password=PW)


def run(x, t=300):
    i, o, e = c.exec_command(x, timeout=t)
    o.channel.recv_exit_status()
    return o.read().decode()


print("=== ~/enigma-work top level ===")
print(run("ls -la ~/enigma-work/ | head -40"))

print("=== probe / solver scripts ===")
print(run("find ~/enigma-work -maxdepth 2 -name '*.py' -printf '%TY-%Tm-%Td %10s %p\\n' "
          "2>/dev/null | sort | head -50"))

print("=== saved artefacts (npz/pt/json/mps) ===")
print(run("find ~/enigma-work -maxdepth 2 \\( -name '*.npz' -o -name '*.pt' -o -name '*.json' "
          "-o -name '*.log' \\) -printf '%TY-%Tm-%Td %10s %p\\n' 2>/dev/null | sort | tail -30"))

print("=== is a probe running? ===")
print(run("ps -eo etimes,pcpu,user,args --sort=-pcpu | grep -E 'enigma|probe|unswap|solver' "
          "| grep -v grep | head -25"))
print("=== load ===")
print(run("uptime; nproc"))
c.close()
