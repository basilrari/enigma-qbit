#!/usr/bin/env python3
"""Measure the GAP between the model's certified optimum and the true answer.

Why: the whole "climb chi" plan rests on an assumption nobody has tested --
that raising chi lifts the TRUE peak's probability faster than it lifts the
spurious optima that currently outrank it.  This measures that directly.

For each chi we run the solver with --peak <known answer> and --exact-top N.
Each run yields two numbers on the same model:
  * P(truth)/uniform   -- the signal we are chasing
  * the CERTIFIED exact argmax of that model, and how far it is from truth
Their ratio G(chi) = P(argmax)/P(truth) is the decisive quantity.  The model
can only answer the challenge when G(chi) < 1 (the truth becomes what the
model itself says is most likely).  If G(chi) never crosses 1, no amount of
chi will solve it and the ladder must be abandoned for a new method.

This is measurement, not solution: --peak feeds in the answer only to SCORE
the model, exactly as a diagnostic flag.  The deliverable never uses it.
"""
import os
import paramiko

V = "/mnt/8tb_hdd2/basilrari/enigma-work/verify"
PY = "/mnt/8tb_hdd2/basilrari/enigma-work/l2venv/bin/python"
PW = open(os.path.expanduser("~/.srvpro_env")).read().split("'")[1]

# Known answers, used purely to score the models (already public in samples/_meta.json).
KNOWN = {
    "d2_s1_39b370e4": "1110101100010111001000000011101111001000",
    "d3_s1_2674779a": "011100110011000101111001110011100000101101100100",
}
LADDER = {
    "d2_s1_39b370e4": [64, 96, 192, 256, 384],
    "d3_s1_2674779a": [96, 192, 256, 384],
}
CAP = 6000000

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("140.123.105.18", username="basilrari", password=PW)


def run(x, t=300):
    i, o, e = c.exec_command(x, timeout=t)
    o.channel.recv_exit_status()
    return o.read().decode()


n = 0
for circ, chis in LADDER.items():
    for chi in chis:
        log = f"{V}/gap_{circ.split('_')[0]}_c{chi}.log"
        cmd = (f"cd {V} && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
               f"MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 setsid nohup {PY} "
               f"hqp_solver.py --qasm {circ}.qasm --chi {chi} "
               f"--peak {KNOWN[circ]} --exact-top 3 --exact-cap {CAP} "
               f"> {log} 2>&1 < /dev/null &")
        c.exec_command(cmd)
        n += 1
        print(f"launched {circ} chi={chi} -> {log}")

print(f"\n{n} gap-measurement jobs launched")
print(run("sleep 5; ps -eo etime,args | grep 'hqp_solver.py --qasm' | grep -v grep "
          "| grep -c gap | sed 's/^/gap jobs alive: /'"))
c.close()
