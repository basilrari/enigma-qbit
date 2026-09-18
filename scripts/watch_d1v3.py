#!/usr/bin/env python3
"""Poll the d1 v3 diagnostic until it prints, then dump it."""
import os
import time

import paramiko

W = '/mnt/8tb_hdd2/basilrari/enigma-work'
LOG = f'{W}/logs/l2win_d1_v3.log'
MARKER = 'amp2(truth) per convention'


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


for i in range(90):
    txt = sh(f'grep -E "^\\[mps\\]|^\\[diag\\]|^    |^\\[marginal\\]|^\\[beam\\]|^  #|^\\[conf\\]|RESULT|Traceback|Error" {LOG} | tail -30')
    if MARKER in txt:
        print(txt, flush=True)
        print('--- tail ---', flush=True)
        print(sh(f'tail -3 {LOG}'), flush=True)
        print('DIAGNOSTIC CAPTURED', flush=True)
        break
    alive = sh("pgrep -f 'l2win_gpu_ru[n]' | wc -l").strip()
    if alive == '0':
        print('PROCESS GONE without diagnostics', flush=True)
        print(sh(f'tail -25 {LOG}'), flush=True)
        break
    if i % 5 == 0:
        print(f'[{i}] alive={alive} tail: ' + sh(f'tail -1 {LOG}').strip()[:120], flush=True)
    time.sleep(60)
else:
    print('TIMEOUT after 90 min', flush=True)
