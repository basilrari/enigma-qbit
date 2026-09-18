#!/usr/bin/env python3
"""Kill orphaned joblib helpers by PID, then smoke-test the CUDA absorb path on d0."""
import os
import paramiko

PW = open(os.path.expanduser("~/.srvpro_env")).read().split("'")[1]
W = "/mnt/8tb_hdd2/basilrari/enigma-work"
V = f"{W}/verify"
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("140.123.105.18", username="basilrari", password=PW)


def run(x, t=300):
    _i, o, e = c.exec_command(x, timeout=t)
    o.channel.recv_exit_status()
    return o.read().decode() + e.read().decode()


pids = run("ps -eo pid,args --no-headers | grep 'l2venv/bin/python' | "
           "grep -E 'loky|resource_tracker' | grep -v grep | awk '{print $1}' | tr '\\n' ' '")
print("orphan pids:", pids.strip())
if pids.strip():
    print(run(f"kill -9 {pids.strip()} 2>/dev/null; sleep 1; echo killed"))
print("l2venv procs left:", run("ps -eo pid,args --no-headers | grep l2venv | grep -v grep | wc -l").strip())

print("\n=== CUDA smoke test on d0 ===")
cmd = (f"cd {V} && HQP_SABRE_TRIALS=1000 HQP_QUIET=1 timeout 280 "
       f"{W}/l2venv/bin/python {W}/l2_absorb.py --qasm d0_s0_trivial.qasm --truth 11100 "
       f"--budget 200 --max-bond 256 --beam 32 --topk 4 --device cuda:0 2>&1 | "
       f"grep -vE 'FutureWarning|tensor_network_ag_compress' | tail -18")
print(run(cmd, t=320))
c.close()
