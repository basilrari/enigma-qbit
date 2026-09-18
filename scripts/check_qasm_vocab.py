#!/usr/bin/env python3
"""What gate vocabulary do the sample circuits actually use?

mpo_compress_unswap counts and decomposes 'unitary' gates, so the circuit must
carry them. Also compare the local samples/ copies against the box's QASMs.
"""
import os

import paramiko

PW = open(os.path.expanduser("~/.srvpro_env")).read().split("'")[1]
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("140.123.105.18", username="basilrari", password=PW)
V = "/mnt/8tb_hdd2/basilrari/enigma-work/verify"


def run(x, t=240):
    i, o, e = c.exec_command(x, timeout=t)
    o.channel.recv_exit_status()
    return o.read().decode() + e.read().decode()


print("=== box QASM files ===")
print(run(f"ls -la {V}/*.qasm 2>&1 | head"))
print("=== head of the real circuit ===")
print(run(f"head -6 {V}/d2_s1_39b370e4.qasm 2>&1"))
print("=== gate vocabulary (box) ===")
print(run(f"for f in d1_s1_4043cafb d2_s1_39b370e4 d3_s1_2674779a; do "
          f"echo -n \"$f: \"; awk '{{print $1}}' {V}/$f.qasm 2>/dev/null | sort | uniq -c | sort -rn | head -6 | tr '\\n' ' '; echo; done"))
print("=== is the file one long line / does it have newlines? ===")
print(run(f"wc -lc {V}/d2_s1_39b370e4.qasm; head -c 200 {V}/d2_s1_39b370e4.qasm | cat -v"))
print("=== local copies ===")
print(run("ls -la ~/enigma-work/verify/*.qasm 2>&1 | head -4"))
c.close()

import glob
for p in sorted(glob.glob("/home/basilsclaw/enigma-solve/samples/*.qasm")):
    b = open(p, "rb").read()
    print(os.path.basename(p), len(b), "bytes;", b[:90])
