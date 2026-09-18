#!/usr/bin/env python3
"""Establish the runnable state of the L2 (unswap absorb + extract) pipeline.

Two questions: (1) does the engine's dependency stack still exist on the box
after the Sep 14 reboot, and (2) does our own verify/probe.py already wire
unswap+extract together (so no orchestrator needs writing from scratch)?
"""
import os

import paramiko

PW = open(os.path.expanduser("~/.srvpro_env")).read().split("'")[1]
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("140.123.105.18", username="basilrari", password=PW)
W = "/mnt/8tb_hdd2/basilrari/enigma-work"


def run(x, t=300):
    i, o, e = c.exec_command(x, timeout=t)
    o.channel.recv_exit_status()
    return o.read().decode() + e.read().decode()


print("=== does verify/probe.py already wire the L2 engine? ===")
print(run(f"grep -n 'unswap\\|mpo_compress\\|beam_search\\|amp2\\|tebd_mps\\|mpo_to_mps\\|l2win' "
          f"{W}/verify/probe.py | head -30"))
print("--- probe.py surface ---")
print(run(f"grep -nE '^def |^class |__main__|add_argument' {W}/verify/probe.py | head -40"))

print("=== torch / qiskit_quimb availability per venv ===")
for v in ["l2venv", ".venv", "gpuenv"]:
    print(v, "->", run(f"cd {W} && timeout 120 {W}/{v}/bin/python -c "
                       f"'import torch,qiskit,quimb;print(\"torch\",torch.__version__,"
                       f"\"cuda\",torch.cuda.is_available(),\"qiskit\",qiskit.__version__)' 2>&1 | tail -2").strip())
print("qiskit_quimb ->", run(f"{W}/l2venv/bin/python -c 'import qiskit_quimb;print(qiskit_quimb.__version__)' 2>&1 | tail -1").strip())

print("=== recover tebd_mps.py surface from bytecode (show errors) ===")
print(run(f"/usr/bin/python3 /tmp/co_probe.py {W}/l2win/__pycache__/tebd_mps.cpython-312.pyc 2>&1 | head -40"))
print("=== sourceless import of tebd_mps in l2venv ===")
print(run(f"mkdir -p /tmp/sl && cp {W}/l2win/__pycache__/tebd_mps.cpython-312.pyc /tmp/sl/tebd_mps.pyc && "
          f"cd /tmp/sl && timeout 180 {W}/l2venv/bin/python -c "
          f"'import tebd_mps as m,inspect;print([n for n in dir(m) if not n.startswith(\"_\")]);"
          f"print([str(inspect.signature(getattr(m,n))) for n in dir(m) if callable(getattr(m,n)) and not n.startswith(\"_\")][:12])' 2>&1 | tail -6"))

print("=== what did the last successful pipeline run leave behind? ===")
print(run(f"ls -la {W}/l2win/__pycache__/ | head -12; echo ---; "
          f"find {W} -maxdepth 2 -name '*.log' -newermt '2026-08-08' ! -newermt '2026-08-30' "
          f"-printf '%TY-%Tm-%Td %10s %p\\n' 2>/dev/null | sort | tail -15"))
c.close()
