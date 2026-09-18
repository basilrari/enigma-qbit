#!/usr/bin/env python3
"""Tail the L2 absorb logs."""
import os
import sys
import paramiko

PW = open(os.path.expanduser("~/.srvpro_env")).read().split("'")[1]
W = "/mnt/8tb_hdd2/basilrari/enigma-work"
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("140.123.105.18", username="basilrari", password=PW)


def run(x, t=600):
    _i, o, e = c.exec_command(x, timeout=t)
    o.channel.recv_exit_status()
    return o.read().decode() + e.read().decode()


slept = sys.argv[1] if len(sys.argv) > 1 else "0"
if slept != "0":
    run(f"sleep {slept}", t=700)
print("=== procs ===")
print(run("ps -o pid,etime,pcpu,args -C python 2>/dev/null | grep -E 'l2_absorb' | "
          "sed 's|\\(/mnt[^ ]*\\)/[a-z0-9_]*\\.log|\\1/LOG|' | head"))
for inst in ["d1_s1_4043cafb", "d2_s1_39b370e4", "d3_s1_2674779a"]:
    print(f"\n===== {inst} =====")
    print(run(f"grep -aE 'circuit |absorb:|state:|###|leftover|Traceback|Error|error' "
              f"{W}/logs/l2_{inst}.log | grep -avE 'Pass:|start compressing' | tail -8"))
    print("  last absorb line:",
          run(f"grep -a '\\[1' {W}/logs/l2_{inst}.log | tail -1")[:190])
c.close()
