#!/usr/bin/env python3
"""Smoke-test the rebuilt L2 orchestrator on the trivial circuit.

d0 = x q[0]; x q[1]; x q[2];  ->  state |q4 q3 q2 q1 q0> = |00111>, so amp2 must
be 1.0 for the right convention and ~0 for the reversed one. Seconds to run,
and it proves: unitarize -> absorb -> leftover composition -> |0..0> projection
-> amp2 -> beam_search are all wired correctly before any real instance runs.
"""
import os

import paramiko

PW = open(os.path.expanduser("~/.srvpro_env")).read().split("'")[1]
W = "/mnt/8tb_hdd2/basilrari/enigma-work"
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("140.123.105.18", username="basilrari", password=PW)
sftp = c.open_sftp()
sftp.put("/home/basilsclaw/enigma-solve/scripts/l2_absorb.py", f"{W}/l2_absorb.py")
sftp.close()

cmd = (f"cd {W}/verify && HQP_SABRE_TRIALS=1000 HQP_QUIET=1 timeout 280 "
       f"{W}/l2venv/bin/python {W}/l2_absorb.py --qasm d0_s0_trivial.qasm "
       f"--truth 11100 --budget 200 --max-bond 256 --beam 32 --topk 4 2>&1 | tail -30")
i, o, e = c.exec_command(cmd, timeout=320)
rc = o.channel.recv_exit_status()
print("exit:", rc)
print(o.read().decode() + e.read().decode())
c.close()
