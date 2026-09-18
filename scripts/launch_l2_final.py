#!/usr/bin/env python3
"""Run the L2 absorb for real, with the final-reconstruction bond bounded.

Why this configuration:
  * the greedy absorb keeps the winners' documented 1024 / 0.002 -- that phase
    goes at ~1s/unitary once the unswap has driven the bond down (observed:
    128 -> 32), so a 40-qubit instance is ~an hour, not ten.
  * --max-bond-final is capped at 256 instead of the default 4096.  The default
    is where the crash lived: bond-4096 SVDs on tensors at ~5e5 entries blew up
    to inf/NaN.  The winners' own reported final_bond was 208, so capping at
    256 is at their operating point, not a shortcut.
  * d1 is included as a control: its answer is known, so a clean H=0 there
    proves the pipeline end to end.

Non-blocking launch; verifies from the box afterwards rather than assuming.
"""
import os
import sys
import time

import paramiko

W = "/mnt/8tb_hdd2/basilrari/enigma-work"
V = f"{W}/verify"
PY = f"{W}/l2venv/bin/python"

# Budget is a CLI arg: the measured throughput is ~4 units/min, so a 40-qubit
# instance needs ~11h. Anything less just burns cores for a partial answer.
BUDGET = sys.argv[1] if len(sys.argv) > 1 else "7200"
THREADS = sys.argv[2] if len(sys.argv) > 2 else "4"

ENV = (f"OMP_NUM_THREADS={THREADS} OPENBLAS_NUM_THREADS={THREADS} "
       f"MKL_NUM_THREADS={THREADS} NUMEXPR_NUM_THREADS={THREADS}")

TRUTH = {
    "d1_s1_4043cafb": "0001001101001111101001001110010001111010100000",
    "d2_s1_39b370e4": "1110101100010111001000000011101111001000",
}

JOBS = [
    ("d2_s1_39b370e4", "l2_d2_b256", 123),
    ("d1_s1_4043cafb", "l2_d1_b256", 123),
]


def main():
    pw = open(os.path.expanduser("~/.srvpro_env")).read().split("'")[1]
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect("140.123.105.18", username="basilrari", password=pw)

    _, o, _ = c.exec_command(f"mkdir -p {V}/logs", timeout=60)
    o.channel.recv_exit_status()

    for inst, tag, seed in JOBS:
        cmd = (f"cd {V} && {ENV} setsid nohup {PY} {W}/l2_absorb.py "
               f"--qasm {inst}.qasm --truth {TRUTH[inst]} "
               f"--max-bond 1024 --cutoff 0.002 "
               f"--max-bond-final 256 --cutoff-final 1e-5 "
               f"--seed {seed} --budget {BUDGET} "
               f"> {V}/logs/{tag}.log 2>&1 < /dev/null & echo launched")
        _, o, _ = c.exec_command(cmd, timeout=60)
        o.channel.recv_exit_status()
        time.sleep(1.0)
        print(f"# launched {tag:14s} ({inst})")

    time.sleep(25)
    _, o, _ = c.exec_command("pgrep -fa 'l2_absorb.py' | awk '{print $1}' | tr '\\n' ' '",
                             timeout=60)
    o.channel.recv_exit_status()
    print("\n=== PIDs on the box ===")
    print(o.read().decode().strip() or "(none)")

    _, o, _ = c.exec_command(
        f"for f in {V}/logs/l2_d*_b256.log; do echo \"--- $f\"; tail -3 $f; done",
        timeout=90)
    o.channel.recv_exit_status()
    print("\n=== first output ===")
    print(o.read().decode().strip()[:900])
    c.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
