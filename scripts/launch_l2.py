#!/usr/bin/env python3
"""Launch the real instances through the rebuilt L2 pipeline.

Truth strings are read from the box's *_meta.json (never from memory), and each
job saves its absorbed MPS so extraction experiments never need a re-absorb.
"""
import json
import os
import stat
import paramiko

PW = open(os.path.expanduser("~/.srvpro_env")).read().split("'")[1]
W = "/mnt/8tb_hdd2/basilrari/enigma-work"
V = f"{W}/verify"
PY = f"{W}/l2venv/bin/python"
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("140.123.105.18", username="basilrari", password=PW)
s = c.open_sftp()
s.put("/home/basilsclaw/enigma-solve/scripts/l2_absorb.py", f"{W}/l2_absorb.py")


def run(x, t=240):
    _i, o, e = c.exec_command(x, timeout=t)
    rc = o.channel.recv_exit_status()
    return rc, o.read().decode() + e.read().decode()


run(f"mkdir -p {W}/mps {W}/logs")
print(run(f"ls {V}/*_meta.json | head -20")[1])

jobs = []
for inst in ["d1_s1_4043cafb", "d2_s1_39b370e4", "d3_s1_2674779a"]:
    rc, txt = run(f"cat {V}/{inst}_meta.json")
    try:
        meta = json.loads(txt.strip())
    except Exception as e:
        print(f"!! no meta for {inst}: {e}: {txt[:120]}"); continue
    truth = meta.get("peaked_state") or meta.get("answer") or meta.get("solution")
    jobs.append((inst, truth, len(truth) if truth else 0))
    print(f"{inst}: truth={truth} ({len(truth) if truth else 0} bits)")

for inst, truth, nbits in jobs:
    log = f"{W}/logs/l2_{inst}.log"
    mpsf = f"{W}/mps/{inst}_absorbed.npz"
    cmd = (f"cd {V} && HQP_SABRE_TRIALS=1000 HQP_QUIET=1 setsid nohup {PY} {W}/l2_absorb.py "
           f"--qasm {inst}.qasm --truth {truth} --max-bond 8192 --cutoff 1e-3 "
           f"--budget 21600 --early-stop 30 --beam 512 --topk 8 "
           f"--save-mps {mpsf} > {log} 2>&1 < /dev/null & echo $!")
    rc, pid = run(cmd)
    print(f"launched {inst}: pid {pid.strip()} -> {log}")
c.close()
