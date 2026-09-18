#!/usr/bin/env python3
"""Knob grid: which stall setting actually lets the absorb make progress?

The measured trajectory on d2 is: ~150 unitaries absorbed in the first ~3
minutes while bonds are small, then STALL MODE, then zero net progress for
half an hour.  So the question is not "how fast" but "does it move at all".

Arms (each a short, capped run on the same instance and seed, so the only
difference is the knob):

  A  baseline                HQP_REWIRE_STALL_MIN_UNITS=15  HQP_STALL_LIMIT=2
  B  no soft-stall           REWIRE_STALL_MIN_UNITS=1000000
  C  never stall             both effectively disabled
  D  no soft-stall + eager unswap  as B, plus a lower unswap threshold

Metric is the last t_u:<n>/<total> in each log plus the observed bond range.
Progress is necessary; the final answer still needs a full run of the winner.
"""
import os
import re
import sys
import time

import paramiko

W = "/mnt/8tb_hdd2/basilrari/enigma-work"
V = f"{W}/verify"
PY = f"{W}/l2venv/bin/python"
INST = "d2_s1_39b370e4"
TRUTH = "1110101100010111001000000011101111001000"
ARMS = [
    ("A_base", {"HQP_REWIRE_STALL_MIN_UNITS": "15", "HQP_STALL_LIMIT": "2"}),
    ("B_nosoft", {"HQP_REWIRE_STALL_MIN_UNITS": "1000000", "HQP_STALL_LIMIT": "2"}),
    ("C_nostall", {"HQP_REWIRE_STALL_MIN_UNITS": "1000000", "HQP_STALL_LIMIT": "1000000"}),
    ("D_eager", {"HQP_REWIRE_STALL_MIN_UNITS": "1000000", "HQP_STALL_LIMIT": "1000000",
                 "HQP_UNSWAP_THRESHOLD": "200000"}),
]
ARM_SECONDS = int(sys.argv[1]) if len(sys.argv) > 1 else 780
THREADS = sys.argv[2] if len(sys.argv) > 2 else "4"


def connect():
    pw = open(os.path.expanduser("~/.srvpro_env")).read().split("'")[1]
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect("140.123.105.18", username="basilrari", password=pw)
    return c


def upload_sync(c):
    """Push the local absorber + engine only if the box copy differs."""
    import hashlib
    pairs = [("scripts/l2_absorb.py", f"{W}/l2_absorb.py"),
             ("l2win_ref/unswap_stallfix.py", f"{W}/l2win/unswap_stallfix.py")]
    sftp = c.open_sftp()
    for local, remote in pairs:
        if not os.path.exists(local):
            print(f"[sync] MISSING local {local} -- skipped")
            continue
        want = hashlib.sha256(open(local, "rb").read()).hexdigest()
        try:
            got = run(c, f"sha256sum {remote} | cut -c1-64").strip()
        except Exception:
            got = ""
        if want == got:
            print(f"[sync] {os.path.basename(local)} already current")
        else:
            sftp.put(local, remote)
            print(f"[sync] uploaded {os.path.basename(local)} ({want[:12]})")
    sftp.close()


def run(c, cmd, timeout=60):
    _, o, _ = c.exec_command(cmd, timeout=timeout)
    o.channel.recv_exit_status()
    return o.read().decode()


def main():
    c = connect()
    upload_sync(c)

    # clear the deck: the livelocked runs are not producing anything
    run(c, "pkill -f l2_absorb.py; sleep 2; pgrep -fc l2_absorb.py || true")
    print("=== knob grid on d2, " + str(ARM_SECONDS) + "s per arm ===", flush=True)

    results = []
    for tag, env in ARMS:
        full = f"{W}/logs/grid_{tag}.log"
        envs = (f"OMP_NUM_THREADS={THREADS} OPENBLAS_NUM_THREADS={THREADS} "
                f"MKL_NUM_THREADS={THREADS} NUMEXPR_NUM_THREADS={THREADS} " +
                " ".join(f"{k}={v}" for k, v in env.items()))
        cmd = (f"cd {V} && {envs} setsid nohup {PY} {W}/l2_absorb.py "
               f"--qasm {INST}.qasm --truth {TRUTH} --max-bond 1024 --cutoff 0.002 "
               f"--max-bond-final 256 --cutoff-final 1e-5 --seed 123 "
               f"--budget {ARM_SECONDS} > {full} 2>&1 < /dev/null & echo started")
        run(c, cmd, timeout=45)
        print(f"[{tag}] launched, waiting {ARM_SECONDS + 90}s ...", flush=True)
        time.sleep(ARM_SECONDS + 90)

        log = run(c, f"cat {full}")
        tus = re.findall(r"t_u: (\d+)/(\d+)", log)
        bonds = [int(x) for x in re.findall(r"'max_bond': (\d+)", log)]
        stalled = log.count("STALL MODE ON")
        last_u = tus[-1][0] if tus else "0"
        results.append((tag, int(last_u), max(bonds) if bonds else 0,
                        bonds[-1] if bonds else 0, stalled,
                        "VERDICT" in log and "H=" in log))
        print(f"[{tag}] t_u={last_u}  bond_max={results[-1][2]} "
              f"bond_last={results[-1][3]}  stall_ons={stalled}", flush=True)
        run(c, "pkill -f l2_absorb.py || true", timeout=30)
        time.sleep(5)

    print("\n=== summary (arms, 100s of unitaries of 2860) ===")
    for tag, last_u, bmax, blast, stalled, verdict in results:
        print(f"  {tag:10s} units={last_u:>5s}  bond_max={bmax:<5d} "
              f"bond_last={blast:<5d} stall_ons={stalled:<3d} extracted={verdict}")
    c.close()


if __name__ == "__main__":
    main()
