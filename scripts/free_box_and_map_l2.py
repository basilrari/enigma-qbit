#!/usr/bin/env python3
"""Free the box (self-inflicted load 54/32) and map the published L2 pipeline.

Kill: the six heuristic gap jobs (their evidence is established -- G widens --
and the dossier shows the naive-MPS line is the wrong attack anyway) and the
obsolete cotengra width probe.

Keep: the two chi=512 builds (the datum that retires the naive line) and the
two candidate-pool tests.
"""
import os

import paramiko

PW = open(os.path.expanduser("~/.srvpro_env")).read().split("'")[1]
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("140.123.105.18", username="basilrari", password=PW)


def run(x, t=300):
    i, o, e = c.exec_command(x, timeout=t)
    o.channel.recv_exit_status()
    return o.read().decode()


# The gap jobs are the only ones using --exact-top 3 --exact-cap 6000000.
print("killed (gap jobs):",
      run("pkill -f 'hqp_solver.py.*--exact-top 3 --exact-cap 6000000'; "
          "pkill -f 'tn_width.py'; sleep 3; echo done").strip())
print(run("ps -eo pcpu,args --sort=-pcpu | grep -c '[h]qp_solver.py'") + " solver procs left")
print("load now:", run("uptime").strip())

print("\n=== l2win/ (published L2 winner code) ===")
print(run("ls -la ~/enigma-work/l2win/ ~/enigma-work/l2win/*/ 2>&1 | head -40"))

print("=== repo/ (official enigma repo clone) ===")
print(run("ls -la ~/enigma-work/repo/ 2>&1 | head -20"))

print("=== every entrypoint / probe anywhere under enigma-work ===")
print(run("grep -rl --include=*.py -E \"__main__\" ~/enigma-work/ 2>/dev/null "
          "| grep -vE 'venv|gpuenv' | head -60"))
print("=== probe_* / tebd / hardening files anywhere ===")
print(run("find ~/ -maxdepth 4 -name 'probe_*.py' -o -maxdepth 4 -name 'tebd_mps.py' "
          "-o -maxdepth 4 -name 'hardening_quantum_proof*.py' 2>/dev/null | head -20"))
print("=== argparse surface of the L2 engine ===")
print(run("grep -nE 'add_argument|def main|__main__|^def ' ~/enigma-work/l2win/unswap_stallfix.py "
          "| head -30"))
print("=== extract.py API ===")
print(run("grep -nE 'def |__main__' ~/enigma-work/l2win/extract.py | head -20"))
c.close()
