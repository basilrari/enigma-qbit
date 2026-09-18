#!/usr/bin/env python3
"""Retire the single-cutoff runs; launch the documented ladder matrix.

d2 (Level 2 -- the published winner resolves this at peak 0.08-0.1):
   4 orderings at the documented final cutoff 1e-5  => consensus test
   1 ordering  at the tightened final cutoff 1e-7   => A/B the dossier's own
                                                       prescription on a KNOWN answer
d3 (Level 3 -- default config is documented to fail, w1~1e-7, fail-closed):
   3 orderings straight at the tightened setting (greedy 0.0005, final 1e-7)
d1 (Level 1 -- known answer): 1 control through the same ladder
"""
import os
import subprocess
import sys

import paramiko

W = "/mnt/8tb_hdd2/basilrari/enigma-work"

MATRIX = [
    # inst, seed, cutoff, cutoff_final, tag
    ("d2_s1_39b370e4", 123, 0.002, 1e-5, "s123"),
    ("d2_s1_39b370e4", 456, 0.002, 1e-5, "s456"),
    ("d2_s1_39b370e4", 789, 0.002, 1e-5, "s789"),
    ("d2_s1_39b370e4", 2026, 0.002, 1e-5, "s2026"),
    ("d2_s1_39b370e4", 123, 0.002, 1e-7, "tight1e7"),
    ("d3_s1_2674779a", 123, 0.0005, 1e-7, "tight1e7a"),
    ("d3_s1_2674779a", 456, 0.0005, 1e-7, "tight1e7b"),
    ("d3_s1_2674779a", 789, 0.0005, 1e-7, "tight1e7c"),
    ("d1_s1_4043cafb", 123, 0.002, 1e-5, "s123"),
]

pw = open(os.path.expanduser("~/.srvpro_env")).read().split("'")[1]
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("140.123.105.18", username="basilrari", password=pw)
_i, o, _e = c.exec_command("pkill -f l2_absorb.py; sleep 2; echo retired", timeout=60)
print("retire:", o.read().decode().strip())
c.close()

for inst, seed, cut, cutf, tag in MATRIX:
    cmd = [sys.executable, "/home/basilsclaw/enigma-solve/scripts/launch_l2_one.py", inst,
           "--seed", str(seed), "--cutoff", str(cut), "--cutoff-final", str(cutf),
           "--tag", tag, "--budget", "10800"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=150)
    print(((r.stdout or r.stderr).strip().splitlines() or ["?"])[-1], flush=True)
print(f"\nlaunched {len(MATRIX)} jobs")
