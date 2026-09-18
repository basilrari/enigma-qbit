#!/usr/bin/env python3
"""Launch one L2 absorb job with the winners' documented config ladder.

Usage:
  launch_l2_one.py <inst> [--seed N] [--cutoff C] [--cutoff-final F]
                          [--max-bond B] [--max-bond-final BF] [--device D]

Log:  {W}/logs/l2_{inst}_s{seed}.log      MPS: {W}/mps/{inst}_s{seed}.npy

Config provenance (dossier enigma-sn63.md):
  greedy absorb   cutoff 0.002 loose, max_bond 1024
  final pass      cutoff 1e-5,      max_bond 4096
  L3 data: final_bond 208 => bond is NOT the limiter, cutoff fidelity is.
  => when the peak comes back ~1e-7, tighten FINAL_CUTOFF to 1e-6/1e-7.
"""
import argparse
import json
import os
import paramiko

W = "/mnt/8tb_hdd2/basilrari/enigma-work"
V = f"{W}/verify"
PY = f"{W}/l2venv/bin/python"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inst")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cutoff", type=float, default=0.002)
    ap.add_argument("--cutoff-final", type=float, default=1e-5)
    ap.add_argument("--max-bond", type=int, default=1024)
    ap.add_argument("--max-bond-final", type=int, default=4096)
    ap.add_argument("--early-stop", type=int, default=30)
    ap.add_argument("--budget", type=float, default=14400.0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--beam", type=int, default=512)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--tag", default=None, help="log suffix override")
    a = ap.parse_args()

    pw = open(os.path.expanduser("~/.srvpro_env")).read().split("'")[1]
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect("140.123.105.18", username="basilrari", password=pw)
    s = c.open_sftp()
    s.put("/home/basilsclaw/enigma-solve/scripts/l2_absorb.py", f"{W}/l2_absorb.py")

    tag = a.tag or f"s{a.seed}"
    try:
        txt = s.open(f"{V}/{a.inst}_meta.json").read().decode()
        truth = json.loads(txt.strip()).get("known_bitstring") or json.loads(txt.strip())["peaked_state"]
    except Exception as e:
        truth = None
        print(f"{a.inst}: NO META ({type(e).__name__})")
    s.close()
    print(f"{a.inst}: truth={truth}")

    log = f"{W}/logs/l2_{a.inst}_{tag}.log"
    mps = f"{W}/mps/{a.inst}_{tag}.npy"
    cmd = (f"cd {V} && mkdir -p {W}/mps {W}/logs && "
           f"HQP_SABRE_TRIALS=1000 setsid nohup {PY} {W}/l2_absorb.py "
           f"--qasm {a.inst}.qasm --truth {truth} "
           f"--cutoff {a.cutoff} --cutoff-final {a.cutoff_final} "
           f"--max-bond {a.max_bond} --max-bond-final {a.max_bond_final} "
           f"--seed {a.seed} --early-stop {a.early_stop} --budget {a.budget} "
           f"--device {a.device} --beam {a.beam} --topk {a.topk} "
           f"--save-mps {mps} > {log} 2>&1 < /dev/null & echo LAUNCHED $! log={log}")
    _i, o, _e = c.exec_command(cmd, timeout=60)
    print(o.read().decode().strip() or f"launched (log {log})")
    c.close()


if __name__ == "__main__":
    main()
