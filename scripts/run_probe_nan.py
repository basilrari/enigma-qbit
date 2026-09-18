#!/usr/bin/env python3
"""Upload probe_nan.py to the box and run it there, streaming output live.

Usage: run_probe_nan.py <qasm> [--layers N] [--sweep]
Wall-clock capped so a slow combination cannot hang the session.
"""
import os
import sys
import time

import paramiko

W = "/mnt/8tb_hdd2/basilrari/enigma-work"
V = f"{W}/verify"
PY = f"{W}/l2venv/bin/python"
HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    qasm = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else ""
    argv = sys.argv[2:] if qasm else sys.argv[1:]
    qpart = f"--qasm {qasm} " if qasm else ""
    script = "probe_nan.py"
    if "--script" in argv:                      # --script NAME selects a probe
        i = argv.index("--script")
        script = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    extra_env = []
    while "--setenv" in argv:                   # --setenv K=V  (repeatable)
        i = argv.index("--setenv")
        extra_env.append(argv[i + 1])
        argv = argv[:i] + argv[i + 2:]
    local = os.path.join(HERE, script)
    remote = f"{W}/{script}"
    # pass the caller's flags through verbatim -- rebuilding them scrambled order
    extra = argv
    cap = 900.0 if "--sweep" in extra else 420.0

    pw = open(os.path.expanduser("~/.srvpro_env")).read().split("'")[1]
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect("140.123.105.18", username="basilrari", password=pw)

    sftp = c.open_sftp()
    sftp.put(local, remote)
    print(f"# uploaded {os.path.getsize(local)} bytes -> {remote}")

    env = ("OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 "
           "NUMEXPR_NUM_THREADS=4 HQP_SABRE_TRIALS=1000 "
           + " ".join(extra_env))
    cmd = (f"cd {V} && {env} {PY} {remote} {qpart}"
           f"{' '.join(extra)} 2>&1 < /dev/null")
    print(f"# running: {cmd}")
    print("-" * 60)
    stdin, out, err = c.exec_command(cmd, timeout=cap + 60)
    t0 = time.time()
    f = out.channel.makefile("r")
    while True:
        line = f.readline()
        if not line:
            if out.channel.exit_status_ready() and not out.channel.recv_ready():
                break
            if time.time() - t0 > cap:
                print(f"# WALL CAP {cap:.0f}s hit -- killing remote probe")
                break
            time.sleep(0.5)
            continue
        sys.stdout.write(line.rstrip()[:300] + "\n")
        sys.stdout.flush()
    print("-" * 60)
    e = err.read().decode().strip()
    if e:
        print("# stderr:", e[:600])
    print(f"# wall {time.time() - t0:.1f}s")
    sftp.close()
    c.close()


if __name__ == "__main__":
    sys.exit(main())
