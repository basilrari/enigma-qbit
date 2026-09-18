#!/usr/bin/env python3
"""Relaunch the decisive absorb jobs on CPU -- correct paths this time.

The nine-job GPU batch died of out-of-memory (GPU 0 was already ~21 GB deep in
another user's processes).  This runs two jobs on CPU instead, 8 BLAS threads
each, documented ladder, 6 h budget, and passes --truth from the sample's meta
sidecar so the log carries the buried/discarded verdict.

Launched from the laptop but it only SSHes out; all compute is on srvpro.
"""
import json
import os
import sys

import paramiko

W = "/mnt/8tb_hdd2/basilrari/enigma-work"
V = f"{W}/verify"
PY = f"{W}/l2venv/bin/python"
LOCAL = "/home/basilsclaw/enigma-solve/samples"

JOBS = [
    ("d2_s1_39b370e4", 123, "cpu123"),
]
BUDGET = "21600"     # 6 h, against the ~63 min the dossier quotes per ordering
THREADS = "8"        # two jobs only -> 8 cores each, no oversubscription
MAXBOND = "1024"
CUT, CUTF = "0.002", "1e-5"

TMPL = (
    "cd {V} && mkdir -p {W}/mps {W}/logs && "
    "OMP_NUM_THREADS={T} OPENBLAS_NUM_THREADS={T} MKL_NUM_THREADS={T} "
    "NUMEXPR_NUM_THREADS={T} HQP_SABRE_TRIALS=1000 HQP_QUIET=1 "
    "setsid nohup {PY} {W}/l2_absorb.py --qasm {inst}.qasm --truth {truth} "
    "--cutoff {cut} --cutoff-final {cutf} --max-bond {mb} --max-bond-final {mb} "
    "--seed {seed} --early-stop 30 --budget {budget} --device cpu "
    "--save-mps {mps} > {log} 2>&1 < /dev/null & echo PID $!"
)


def truth_of(inst):
    d = json.load(open(f"{LOCAL}/{inst}_meta.json"))
    for k in ("truth", "answer", "bitstring"):
        if k in d:
            return d[k]
    for k, v in d.items():
        if isinstance(v, str) and set(v) <= {"0", "1"} and len(v) > 8:
            return v
    raise SystemExit(f"no truth in {inst}_meta.json: {list(d)[:8]}")


def main():
    pw = open(os.path.expanduser("~/.srvpro_env")).read().split("'")[1]
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect("140.123.105.18", username="basilrari", password=pw)

    pre = (f"echo load: $(cut -d' ' -f1-3 /proc/loadavg) cores: $(nproc); "
           f"ls -l {W}/l2_absorb.py {V}/d2_s1_39b370e4.qasm "
           f"{V}/d1_s1_4043cafb.qasm 2>&1 | tail -3; "
           f"echo absorb_running: $(pgrep -fc l2_absorb || true)")
    _, o, e = c.exec_command(pre, timeout=60)
    o.channel.recv_exit_status()
    print(o.read().decode().strip())

    for inst, seed, tag, in JOBS:
        truth = truth_of(inst)
        cmd = TMPL.format(V=V, W=W, T=THREADS, PY=PY, inst=inst, truth=truth,
                          cut=CUT, cutf=CUTF, mb=MAXBOND, seed=seed,
                          budget=BUDGET, log=f"{W}/logs/l2_{inst}_{tag}.log",
                          mps=f"{W}/mps/{inst}_{tag}.npy")
        _, o, e = c.exec_command(cmd, timeout=90)
        # do NOT block on the channel: the backgrounded job can hold stdout open
        o.channel.settimeout(5)
        try:
            print(f"{inst} [{tag}] truth={truth} -> {o.read().decode().strip()}")
        except Exception:
            print(f"{inst} [{tag}] truth={truth} -> launched (no ack read)")

    # verify they are actually alive and burning CPU before believing it
    import time
    time.sleep(20)
    chk = (f"pgrep -fa 'l2_absorb' | head -5; echo '---'; "
           f"ps -o pid,etimes,pcpu,args -C python 2>/dev/null | grep -c l2_absorb")
    _, o, e = c.exec_command(chk, timeout=60)
    o.channel.recv_exit_status()
    print("--- liveness ---")
    print(o.read().decode().strip())
    c.close()
    print(f"launched {len(JOBS)} CPU jobs, {THREADS} threads each, budget {BUDGET}s")


if __name__ == "__main__":
    sys.exit(main())
