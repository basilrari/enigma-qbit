#!/usr/bin/env python3
"""Hard proof of WHICH machine is doing the work.

Prints, from inside srvpro: hostname, the full command line of every absorb
process (including the interpreter path -- if that path is the box's venv, the
compute is on the box), elapsed seconds, CPU seconds, and current load.
Nothing here is inferred; it is all read from the server itself.
"""
import os
import sys

import paramiko

pw = open(os.path.expanduser("~/.srvpro_env")).read().split("'")[1]

CMD = (
    'echo "host: $(hostname)   date: $(date +%F_%T)"; '
    'echo "cores: $(nproc)   load: $(cut -d\\  -f1-3 /proc/loadavg)"; '
    'echo "--- absorb processes (pid elapsed cpu% full cmd) ---"; '
    "ps -eo pid,etimes,pcpu,rss,args | grep '[l]2_absorb.py' | head -6; "
    'echo "--- cpu seconds burned per job ---"; '
    "for p in $(pgrep -f '[l]2_absorb.py'); do "
    "awk '{print \\$1, \\$14/100, \\$15/100}' /proc/$p/stat 2>/dev/null; done; "
    'echo "--- remote work dir ---"; '
    "ls -1 /mnt/8tb_hdd2/basilrari/enigma-work/verify/*.qasm 2>/dev/null | head -4"
)


def main():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect("140.123.105.18", username="basilrari", password=pw)
    _, o, e = c.exec_command(CMD, timeout=120)
    o.channel.recv_exit_status()
    out = o.read().decode().strip()
    err = e.read().decode().strip()
    print(out if out else "(no output)")
    if err:
        print("stderr:", err[:500])
    c.close()


if __name__ == "__main__":
    sys.exit(main())
