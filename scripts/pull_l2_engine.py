#!/usr/bin/env python3
"""Pull the L2 engine sources locally and hunt for any existing caller.

The calling convention for mpo_compress_unswap is the one thing I must not
guess -- if a previous orchestrator called it anywhere on the box, copy that
usage verbatim.
"""
import os
import pathlib

import paramiko

PW = open(os.path.expanduser("~/.srvpro_env")).read().split("'")[1]
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("140.123.105.18", username="basilrari", password=PW)
W = "/mnt/8tb_hdd2/basilrari/enigma-work"

loc = pathlib.Path("/home/basilsclaw/enigma-solve/l2win_ref")
loc.mkdir(exist_ok=True)
sftp = c.open_sftp()
for f in ["unswap_stallfix.py", "circuit_mpo.py", "extract.py", "utils.py", "Dockerfile"]:
    try:
        sftp.get(f"{W}/l2win/{f}", str(loc / f))
        print("pulled", f, (loc / f).stat().st_size)
    except Exception as ex:
        print("MISS", f, ex)
sftp.close()


def run(x, t=300):
    i, o, e = c.exec_command(x, timeout=t)
    o.channel.recv_exit_status()
    return o.read().decode()


print("\n=== any caller of the engine anywhere on the box? ===")
print(run(f"grep -rn 'mpo_compress_unswap\\|tebd_mps\\|topk(\\|beam_search(\\|mpo_to_mps' "
          f"{W} --include=*.py 2>/dev/null | grep -v 'l2win/unswap\\|l2win/extract' | head -25"))
print("=== shell history hints ===")
print(run("grep -aE 'unswap|tebd|topk|probe_diag|hqp_v3' ~/.bash_history 2>/dev/null | tail -25"))
print("=== any hqp_v3_solution dir? ===")
print(run("find / -maxdepth 6 -name 'hqp_v3_solution' -o -maxdepth 6 -name 'probe_full_v3.py' "
          "2>/dev/null | head"))
c.close()
print("\nlocal:", sorted(p.name for p in loc.iterdir()))
