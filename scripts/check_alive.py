#!/usr/bin/env python3
"""Liveness check: is each absorb job actually burning CPU, or stalled?"""
import os
import time

import paramiko

W = "/mnt/8tb_hdd2/basilrari/enigma-work"
REMOTE = r"""
import glob, os, time
rows = []
for p in glob.glob('/proc/[0-9]*'):
    try:
        cl = open(p + '/cmdline', 'rb').read().decode(errors='ignore').split('\x00')
    except Exception:
        continue
    if not any('l2_absorb.py' in x for x in cl):
        continue
    try:
        st = open(p + '/stat').read().split()
        cpu = (int(st[13]) + int(st[14])) / os.sysconf('SC_CLK_TCK')
    except Exception:
        cpu = -1
    tag = ''
    for k, x in enumerate(cl):
        if x == '--tag' and k + 1 < len(cl):
            tag = cl[k + 1]
    if not tag:
        for x in cl:
            if x.startswith('l2_'):
                tag = os.path.basename(x)[:-2]
    rows.append((tag, int(p.rsplit('/', 1)[1]), round(cpu, 1)))
print('CPU_TIME_SAMPLE_1')
for t, pid, cpu in sorted(rows):
    print(f'  {t:28s} pid={pid:8d} cpu_s={cpu}')
print('n_jobs', len(rows))
"""


def main():
    pw = open(os.path.expanduser("~/.srvpro_env")).read().split("'")[1]
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect("140.123.105.18", username="basilrari", password=pw)
    s = c.open_sftp()
    with s.open(W + "/alive_probe.py", "w") as fh:
        fh.write(REMOTE)
    s.close()
    for i in (1, 2):
        _, o, _ = c.exec_command(f"cd {W} && ./l2venv/bin/python alive_probe.py && uptime", timeout=90)
        o.channel.recv_exit_status()
        print(f"--- sample {i} (t={time.strftime('%H:%M:%S')})")
        print(o.read().decode())
        if i == 1:
            time.sleep(120)
    c.close()


if __name__ == "__main__":
    main()
