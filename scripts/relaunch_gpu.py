#!/usr/bin/env python3
"""Relaunch the three real instances on the free GPU 0 (CUDA absorb).

The CPU runs are ~12 min in and the engine is built for the GPU: on the CUDA
path d0 absorbed in 6 s including startup. Verify the GPU wiring on d0 first,
then replace the CPU jobs.
"""
import os
import paramiko

PW = open(os.path.expanduser("~/.srvpro_env")).read().split("'")[1]
W = "/mnt/8tb_hdd2/basilrari/enigma-work"
V = f"{W}/verify"
PY = f"{W}/l2venv/bin/python"
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("140.123.105.18", username="basilrari", password=PW)


def run(x, t=600):
    _i, o, e = c.exec_command(x, timeout=t)
    o.channel.recv_exit_status()
    return o.read().decode() + e.read().decode()


s = c.open_sftp()
s.put("/home/basilsclaw/enigma-solve/scripts/l2_absorb.py", f"{W}/l2_absorb.py")
s.close()

print("=== GPU d0 verification (must absorb 100%, bond 1, amp2=1.0) ===")
print(run(f"cd {V} && HQP_SABRE_TRIALS=1000 HQP_QUIET=1 timeout 280 {PY} {W}/l2_absorb.py "
          f"--qasm d0_s0_trivial.qasm --truth 11100 --budget 200 --max-bond 256 "
          f"--beam 32 --topk 4 --device cuda:0 2>&1 | "
          f"grep -vE 'FutureWarning|tensor_network_ag_compress' | tail -16", t=320))

print("=== retiring the CPU runs ===")
print(run("pkill -f 'l2_absorb.py' ; sleep 2; ps -eo pid,args --no-headers | "
          "grep -c '[l]2_absorb.py' || echo 0"))

print("=== relaunch on cuda:0 ===")
for inst in ["d1_s1_4043cafb", "d2_s1_39b370e4", "d3_s1_2674779a"]:
    print(inst, "->", run(f"cd /home/basilsclaw/enigma-solve && /usr/bin/python3 "
                          f"scripts/launch_l2_one.py {inst} --device cuda:0 2>/dev/null | tail -1", t=120))
c.close()
