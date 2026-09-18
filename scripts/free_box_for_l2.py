#!/usr/bin/env python3
"""Clear the box for the L2 pipeline: retire the superseded naive-MPS line.

The χ=512 ladder and the candidate-pool probes were measuring whether the naive
MPS line could ever work. The published evidence answers that: on Level 2 the
peak weight is ~0.1, while the naive MPS tops out near 5e-7 -- five orders worse.
Keeping them running would only starve the run that can actually finish.
"""
import os
import paramiko

PW = open(os.path.expanduser("~/.srvpro_env")).read().split("'")[1]
W = "/mnt/8tb_hdd2/basilrari/enigma-work"
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("140.123.105.18", username="basilrari", password=PW)


def run(x, t=300):
    _i, o, e = c.exec_command(x, timeout=t)
    o.channel.recv_exit_status()
    return o.read().decode() + e.read().decode()


print("=== before ===")
print(run("ps -C python -o pid,pcpu,etimes,args --no-headers | "
          "awk '{print $1, $2, $3, $4, $5, $6, $7}' | sed 's|/mnt.*enigma-work/||' | head -20"))
print("load:", run("cat /proc/loadavg"))

for pat in ["hqp_solver.py", "tn_width.py", "pool_"]:
    print(pat, "->", run(f"pkill -f '{pat}' && echo killed || echo none"))

print("=== after ===")
print(run("ps -C python -o pid,pcpu,etimes,args --no-headers | "
          "awk '{print $1, $2, $3, $4, $5, $6}' | sed 's|/mnt.*enigma-work/||' | head -20"))
print("load:", run("cat /proc/loadavg"))
c.close()
