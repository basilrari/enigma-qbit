#!/usr/bin/env python3
"""Server-side reality check: load, GPU memory, and who owns it."""
import os

import paramiko

REMOTE = r'''
echo "=== $(hostname) $(date +%F\ %T) ==="
echo "--- load / cores ---"; uptime; nproc
echo "--- top CPU ---"; ps -eo pcpu,rss,etime,user,comm --sort=-pcpu | head -8
echo "--- GPU ---"; nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
echo "--- GPU procs ---"; nvidia-smi --query-compute-apps=pid,used_memory --format=csv | head -15
echo "--- my absorb procs ---"; pgrep -af "l2_absorb|l2venv" | head -10; echo "(none if blank)"
'''


def main():
    pw = open(os.path.expanduser("~/.srvpro_env")).read().split("'")[1]
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect("140.123.105.18", username="basilrari", password=pw)
    _, o, e = c.exec_command(REMOTE, timeout=90)
    o.channel.recv_exit_status()
    print(o.read().decode())
    err = e.read().decode()
    if err.strip():
        print("STDERR:", err[:300])
    c.close()


if __name__ == "__main__":
    main()
