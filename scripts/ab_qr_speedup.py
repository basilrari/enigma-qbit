#!/usr/bin/env python3
"""A/B the QR centre-move optimisation against the old SVD path.

Runs the SAME circuit at the SAME chi on the SAME box at the SAME time (both
jobs launched together) so machine load cannot confound the comparison, and
both finish with the certified exact argmax so we can also prove the physics
is unchanged, not just the clock.

Timing is read from the per-gate "dt=" log lines, so we compare like with like.
"""
import os
import subprocess
import sys
import time

import paramiko

ROOT = "/home/basilsclaw/enigma-solve"
V = "/mnt/8tb_hdd2/basilrari/enigma-work/verify"
PY = "/mnt/8tb_hdd2/basilrari/enigma-work/l2venv/bin/python"
PW = open(os.path.expanduser("~/.srvpro_env")).read().split("'")[1]
CIRC = sys.argv[1] if len(sys.argv) > 1 else "d1_s1_4043cafb.qasm"
CHI = int(sys.argv[2]) if len(sys.argv) > 2 else 128

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("140.123.105.18", username="basilrari", password=PW)


def run(x, t=400):
    i, o, e = c.exec_command(x, timeout=t)
    o.channel.recv_exit_status()
    return o.read().decode().strip()


def launch(script, tag):
    log = f"{V}/ab_{tag}.log"
    cmd = (f"cd {V} && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
           f"MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 setsid nohup {PY} "
           f"{script} --qasm {CIRC} --chi {CHI} --exact-top 3 "
           f"> {log} 2>&1 < /dev/null &")
    # Do NOT read the channel: the backgrounded job would hold it open and
    # block (that cost us the first attempt). Fire and forget, then poll the
    # log file.
    c.exec_command(cmd)
    return log


# Old engine = the version already committed (pre-QR), shipped side by side.
old_src = subprocess.run(["git", "-C", ROOT, "show", "HEAD:hqp_solver.py"],
                         capture_output=True, text=True).stdout
assert old_src and "_qr_center" not in old_src, "old source looks wrong"
open("/tmp/hqp_old.py", "w").write(old_src)

sftp = c.open_sftp()
sftp.put("/tmp/hqp_old.py", f"{V}/hqp_solver_old.py")
sftp.put(f"{ROOT}/hqp_solver.py", f"{V}/hqp_solver.py")
sftp.close()
print(f"uploaded old + new solver | circuit={CIRC} chi={CHI}")

# Launch both together so they see the same load.
print("old:", launch("hqp_solver_old.py", f"old_{CHI}"))
print("new:", launch("hqp_solver.py", f"new_{CHI}"))
for waited in range(0, 300, 60):
    time.sleep(60)
    a = run(f"grep -a 'gate 300/' {V}/ab_old_{CHI}.log | tail -1")
    b = run(f"grep -a 'gate 300/' {V}/ab_new_{CHI}.log | tail -1")
    if a and b:
        break

def secs(line):
    for tok in line.split():
        if tok.startswith("dt="):
            return float(tok[3:-1])
    return None

ta, tb = secs(a), secs(b)
print("\n=== timing at gate 300 (same load, same circuit) ===")
print("OLD:", a)
print("NEW:", b)
if ta and tb:
    print(f"speedup at gate 300: {ta/tb:.2f}x  (old {ta:.1f}s -> new {tb:.1f}s)")

print("\n=== jobs ===")
print(run(f"ps -eo etime,args | grep -E 'hqp_solver(_old)?\\.py --qasm {CIRC[:12]}'"
          f" | grep -v grep | cut -c1-70"))
c.close()
