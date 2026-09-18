#!/usr/bin/env python3
"""Is each absorb job advancing, or stuck inside one giant step?

Prints, per log: last INFO line, its timestamp, log mtime, and how long the
file has been silent.  Silence + rising CPU time means one long compression
step; silence + flat CPU time means a genuine stall.
"""
import os
import time

import paramiko

W = "/mnt/8tb_hdd2/basilrari/enigma-work"
REMOTE = r'''
cd %s || exit 1
date "+NOW %%Y-%%m-%%dT%%H:%%M:%%S"
echo "--- cpu ---"
for p in $(pgrep -f l2_absorb.py); do
  awk -v pid=$p '{print pid, $14, $15}' /proc/$p/stat 2>/dev/null
done | head -20
echo "--- logs ---"
for f in logs/l2_d*.log; do
  n=$(basename $f .log)
  last=$(grep -a "INFO" $f | tail -1 | cut -c1-95)
  mt=$(date -r $f "+%%H:%%M:%%S")
  echo "$n|$mt|$last"
done
''' % W


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
        print("STDERR:", err[:400])
    c.close()


if __name__ == "__main__":
    main()
