#!/usr/bin/env python3
"""Collect the decisive lines from every corner/model log on the box."""
import os
import sys

import paramiko

V = "/mnt/8tb_hdd2/basilrari/enigma-work/verify"
PW = open(os.path.expanduser("~/.srvpro_env")).read().split("'")[1]
PAT = sys.argv[1] if len(sys.argv) > 1 else (
    r"P\(|certified|EXACT|ANSWER|H=|ratio|uniform|peak P|best H|nodes=")

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("140.123.105.18", username="basilrari", password=PW)


def run(x, t=300):
    i, o, e = c.exec_command(x, timeout=t)
    o.channel.recv_exit_status()
    return o.read().decode()


logs = run(f"cd {V} && ls -t gap_*.log argmax_*c512*.log ab_old_128.log 2>/dev/null | head -20").split()
print("logs found:", " ".join(logs), "\n")
for lg in logs:
    body = run(f"cd {V} && grep -aE '{PAT}' {lg} | tail -12")
    tail = run(f"cd {V} && tail -2 {lg} | cut -c1-140")
    print(f"===== {lg} =====")
    print(body.strip() or "(no matching lines yet)")
    print("  ...tail:", tail.strip().replace("\n", " | ")[:200])
    print()
print("running:", run("ps -eo etime,args | grep 'hqp_solver.py --qasm' | grep hqp_solver "
                      "| grep -v grep | awk '{print $1}' | tr '\\n' ' '"))
c.close()
