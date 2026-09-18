#!/usr/bin/env python3
"""Kill orphaned loky/resource_tracker workers; check GPU availability."""
import os
import paramiko

PW = open(os.path.expanduser("~/.srvpro_env")).read().split("'")[1]
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("140.123.105.18", username="basilrari", password=PW)


def run(x, t=300):
    _i, o, e = c.exec_command(x, timeout=t)
    o.channel.recv_exit_status()
    return o.read().decode() + e.read().decode()


print("=== orphaned helpers of MY venv only ===")
print(run("ps -eo pid,pcpu,args --no-headers | grep 'l2venv/bin/python' | "
          "grep -E 'loky|resource_tracker|joblib' | grep -v grep"))
print("kill:", run("pkill -f 'l2venv/bin/python -m joblib' ; "
                   "pkill -f 'l2venv/bin/python -c from multiprocessing.resource_tracker' ; echo done"))
print("remaining mine:", run("ps -eo pid,args --no-headers | grep l2venv | grep -v grep | wc -l")[:20])

print("=== GPUs ===")
print(run("nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu "
          "--format=csv,noheader"))
print("=== who is on them ===")
print(run("nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader"))
print("=== load now ===")
print(run("cat /proc/loadavg"))
c.close()
