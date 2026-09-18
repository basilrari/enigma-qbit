#!/usr/bin/env python3
"""Wait for the d1 control run. If the winner's engine reproduces H=0 on the
already-solved d1, launch d2 and d3 through the SAME engine at bond 1024.
"""
import json
import os
import re
import time
import paramiko

W = '/mnt/8tb_hdd2/basilrari/enigma-work'
PY = f'{W}/l2venv/bin/python'
TRUTH = {
    'd1': '0001001101001111101001001110010001111010100000',
    'd2': '1110101100010111001000000011101111001000',
    'd3': '011100110011000101111001110011000000101101100100',
}
QASM = {'d2': f'{W}/verify/d2_s1_39b370e4.qasm', 'd3': f'{W}/verify/d3_s1_2674779a.qasm'}

pw = open(os.path.expanduser('~/.srvpro_env')).read().split("'")[1]
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('140.123.105.18', username='basilrari', password=pw, timeout=20)


def run(cmd, t=90, read_t=25):
    _i, o, _e = c.exec_command(cmd, timeout=t)
    o.channel.settimeout(read_t)
    try:
        return o.read().decode(errors='replace')
    except Exception as ex:
        return f'[read timeout {type(ex).__name__}]'


# 1. wait for the d1 control
for i in range(60):
    log = run(f'cat {W}/logs/l2win_d1_smoke.log 2>/dev/null')
    if 'RESULT JSON' in log:
        break
    if run("pgrep -fc 'l2win_gpu_ru[n]' || echo 0").strip() == '0' and i > 2:
        print('[watch] d1 process gone without RESULT', flush=True)
        break
    print(f'[watch] {time.strftime("%H:%M")} waiting on d1 control', flush=True)
    time.sleep(60)

log = run(f'cat {W}/logs/l2win_d1_smoke.log')
print('=== d1 control tail ===', flush=True)
print('\n'.join(log.strip().splitlines()[-14:]), flush=True)

# 2. parse the verdict
verdict = None
m = re.search(r'\[marginal\] H=(\d+)/(\d+)', log)
if m:
    verdict = int(m.group(1))
print(f'[gate] d1 marginal H={verdict}', flush=True)

# 3. if the engine reproduces d1 exactly, use it on the real targets
if verdict == 0:
    for tag in ('d2', 'd3'):
        cmd = (f'( cd {W} && setsid env CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 '
               f'{PY} {W}/l2win_gpu_run.py {QASM[tag]} 12600 1024 0.002 0 {TRUTH[tag]} '
               f'> {W}/logs/l2win_{tag}_b1024.log 2>&1 < /dev/null & ) ; echo ok')
        run(cmd, read_t=8)
        print(f'[launch] {tag} via winner engine, bond 1024, budget 12600s', flush=True)
        time.sleep(5)
else:
    print('[gate] d1 did NOT reproduce H=0 -- not launching d2/d3. Investigate first.', flush=True)

c.close()
print('DONE', flush=True)
