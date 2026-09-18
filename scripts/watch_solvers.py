#!/usr/bin/env python3
"""Watch the d2/d3 solver runs on srvpro and report the moment they finish.

The runs are long (an hour or more), and the point of watching is that the
finish must not be missed -- the whole standing goal is d2/d3 at Hamming 0.
Polls until no hqp_solver process remains, then prints the verdicts.
"""
import json
import os
import re
import time

import paramiko

W = '/mnt/8tb_hdd2/basilrari/enigma-work'
V = W + '/verify'
TRUTH = {
    'd2': '1110101100010111001000000011101111001000',      # 40 bits
    'd3': '011100110011000101111001110011000000101101100100',  # 48 bits
}
TAGS = ('d2_384', 'd3_256', 'd2_256')


def hamming(a, b):
    return sum(x != y for x, y in zip(a, b))


def connect():
    pw = open(os.path.expanduser('~/.srvpro_env')).read().split("'")[1]
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect('140.123.105.18', username='basilrari', password=pw)
    return c


def R(c, cmd, t=120):
    _, o, _ = c.exec_command(cmd, timeout=t)
    o.channel.settimeout(25)
    try:
        return o.read().decode().strip()
    except Exception:
        return '(channel timeout)'


def main():
    deadline = time.time() + 9000
    while True:
        try:
            c = connect()
        except Exception as e:
            print(f'[watch] ssh failed ({e}); retrying in 5 min', flush=True)
            time.sleep(300)
            continue
        n = R(c, "pgrep -fc 'hqp_solve[r].py' || echo 0")
        clock = R(c, 'date +%H:%M')
        prog = []
        for tag in TAGS:
            g = re.findall(r'gate (\d+)/(\d+) .*?dt=([\d.]+)s',
                           R(c, f'cat {V}/solve_{tag}.log'))
            if g:
                prog.append(f'{tag} {g[-1][0]}/{g[-1][1]}')
        print(f'[watch] {clock} solvers={n} {" ".join(prog)}', flush=True)
        if n in ('0', '') or time.time() > deadline:
            print('\n=== FINAL ===', flush=True)
            for tag in TAGS:
                key = tag.split('_')[0]
                raw = R(c, f'cat {V}/{tag}.json 2>/dev/null')
                tail = [l for l in R(c, f'tail -6 {V}/solve_{tag}.log').splitlines()
                        if l.strip()][-3:]
                print(f'\n--- {tag} ---')
                if raw and raw != '(channel timeout)':
                    try:
                        d = json.loads(raw)
                        best = None
                        for k in ('peaked_state', 'bits', 'answer',
                                  'bitstring', 'state', 'best'):
                            if isinstance(d, dict) and d.get(k):
                                best = d[k]
                                break
                        if best:
                            print(f'  state={best}')
                            print(f'  H={hamming(best, TRUTH[key])}/{len(TRUTH[key])}'
                                  f' {"*** MATCH ***" if hamming(best, TRUTH[key]) == 0 else ""}')
                        else:
                            print('  json keys:', list(d)[:12] if isinstance(d, dict) else type(d))
                            print('  raw:', raw[:300])
                    except Exception as e:
                        print('  json parse failed:', e, '| raw:', raw[:200])
                else:
                    print('  no output file')
                print('  log tail:', ' | '.join(tail))
            c.close()
            break
        c.close()
        time.sleep(240)


if __name__ == '__main__':
    main()
