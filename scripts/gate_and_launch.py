#!/usr/bin/env python3
"""Gate on the d1 control, then launch d2 and d3 through the same engine.

d1 is already solved, so a HEALTHY pipeline must land near H=0. Only if the
control shows a real signal do we spend 3.5h x2 on the unsolved tiers.
"""
import os
import re
import time

import paramiko

W = '/mnt/8tb_hdd2/basilrari/enigma-work'
PY = f'{W}/l2venv/bin/python'
LOG = f'{W}/logs/l2win_d1_v3.log'
TRUTH = {
    'd2': '1110101100010111001000000011101111001000',
    'd3': '011100110011000101111001110011000000101101100100',
}
BUDGET = 12600


def sh(cmd, read_t=20):
    pw = open(os.path.expanduser('~/.srvpro_env')).read().split("'")[1]
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect('140.123.105.18', username='basilrari', password=pw, timeout=20)
    _i, o, _e = c.exec_command(cmd, timeout=60)
    o.channel.settimeout(read_t)
    try:
        out = o.read().decode(errors='replace')
    except Exception as ex:
        out = f'[read timeout {type(ex).__name__}]'
    c.close()
    return out


txt = ''
for i in range(90):
    txt = sh(f'grep -E "^\\[mps\\]|^\\[diag\\]|^    |^\\[marginal\\]|^  #|^\\[conf\\]" {LOG} | tail -40')
    if 'amp2(truth) per convention' in txt:
        break
    if sh("pgrep -f 'l2win_gpu_ru[n]' | wc -l").strip() == '0':
        txt = sh(f'tail -30 {LOG}')
        break
    time.sleep(60)

print('=== d1 control diagnostic ===', flush=True)
print(txt, flush=True)

a2 = [float(m) for m in re.findall(r'^\s+\S+\s+([0-9.]+e-?\d+)', txt, re.M)]
best_a2 = max(a2) if a2 else 0.0
hs = [int(m) for m in re.findall(r'H=(\d+)/46', txt)]
best_h = min(hs) if hs else 999
print(f'\n[gate] best amp2={best_a2:.3e}  best H={best_h}/46', flush=True)

# A worked absorb puts amp2 at 1e-3..1e-1; a finding pipeline lands near H=0.
if best_a2 >= 1e-3 or best_h <= 8:
    print('[gate] CONTROL PASSES -> launching d2 and d3', flush=True)
    for tag, qasm in (('d2', 'd2_s1_39b370e4'), ('d3', 'd3_s1_2674779a')):
        cmd = (f'( cd {W} && setsid env CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 '
               f'{PY} {W}/l2win_gpu_run.py {W}/verify/{qasm}.qasm {BUDGET} 1024 0.002 0 '
               f'{TRUTH[tag]} > {W}/logs/l2win_{tag}_v1.log 2>&1 < /dev/null & ) ; echo started')
        print(f'[launch] {tag}: {sh(cmd, read_t=8).strip()}', flush=True)
        time.sleep(20)
else:
    print('[gate] CONTROL FAILED the gate -- NOT launching. Diagnose the convention first.', flush=True)
