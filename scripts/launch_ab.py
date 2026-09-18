#!/usr/bin/env python3
"""Clean A/B: same instance, same chi, only the SVD route differs.

Uploads the pre-patch solver (from git HEAD) and the patched one, then runs
them side by side with a control on the already-solved d1:

    ab_old_d2_128   old solver, d2 @ chi=128   (baseline: certified H=15)
    ab_new_d2_128   new solver, d2 @ chi=128   (the test)
    ab_new_d1_64    new solver, d1 @ chi=64    (control: must stay H=0)

One BLAS thread per job, non-blocking launch, absolute paths throughout.
"""
import os
import subprocess
import sys
import time

import paramiko

W = "/mnt/8tb_hdd2/basilrari/enigma-work"
V = f"{W}/verify"
PY = f"{W}/l2venv/bin/python"
ROOT = "/home/basilsclaw/enigma-solve"
ENV = ("OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "
       "NUMEXPR_NUM_THREADS=1")

JOBS = [
    # script on the box        qasm                chi  tag
    ("hqp_solver_old.py", "d2_s1_39b370e4.qasm", 128, "ab_old_d2_128"),
    ("hqp_solver_new.py", "d2_s1_39b370e4.qasm", 128, "ab_new_d2_128"),
    ("hqp_solver_new.py", "d1_s1_4043cafb.qasm", 64, "ab_new_d1_64"),
]


def main():
    pw = open(os.path.expanduser("~/.srvpro_env")).read().split("'")[1]
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect("140.123.105.18", username="basilrari", password=pw)

    # the pre-patch solver straight out of git HEAD
    old = subprocess.run(["git", "-C", ROOT, "show", "HEAD:hqp_solver.py"],
                         capture_output=True, text=True, check=True).stdout
    open("/tmp/hqp_old.py", "w").write(old)

    sftp = c.open_sftp()
    sftp.put("/tmp/hqp_old.py", f"{V}/hqp_solver_old.py")
    sftp.put(f"{ROOT}/hqp_solver.py", f"{V}/hqp_solver_new.py")
    print(f"# uploaded old ({len(old)} bytes) and new solver to {V}")

    _, o, _ = c.exec_command(f"mkdir -p {V}/logs", timeout=60)
    o.channel.recv_exit_status()

    for script, qasm, chi, tag in JOBS:
        out = f"{V}/{tag}.json"
        log = f"{V}/logs/{tag}.log"
        cmd = (f"cd {V} && {ENV} setsid nohup {PY} {V}/{script} "
               f"--qasm {qasm} --chi {chi} --exact-top 3 --out {out} "
               f"> {log} 2>&1 < /dev/null & echo started")
        _, o, e = c.exec_command(cmd, timeout=60)
        o.channel.recv_exit_status()
        time.sleep(1.5)
        print(f"# launched {tag:16s} {script} chi={chi}")

    time.sleep(20)
    _, o, _ = c.exec_command(
        "pgrep -fa 'hqp_solver_(old|new).py' | awk '{print $1, $NF}'", timeout=60)
    o.channel.recv_exit_status()
    print("\n=== running now ===")
    print(o.read().decode().strip() or "(none)")

    _, o, _ = c.exec_command(
        f"for f in {V}/logs/ab_*.log; do echo \"--- $f\"; tail -2 $f; done",
        timeout=90)
    o.channel.recv_exit_status()
    print("\n=== early log output ===")
    print(o.read().decode().strip()[:1200])
    sftp.close()
    c.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
