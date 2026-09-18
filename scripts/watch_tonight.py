#!/usr/bin/env python3
"""Watch tonight's three srvpro jobs; report when each finishes."""
import os, re, sys, time, json
import paramiko

W = '/mnt/8tb_hdd2/basilrari/enigma-work'
TRUTH = {
    'd2': '1110101100010111001000000011101111001000',
    'd3': '011100110011000101111001110011000000101101100100',
}
JOBS = [('d2_512', 'd2', f'{W}/verify/d2_512.json', f'{W}/logs/d2_512.log'),
        ('d3_512', 'd3', f'{W}/verify/d3_512.json', f'{W}/logs/d3_512.log'),
        ('pristine_d2_s0', 'd2', None, f'{W}/logs/pristine_d2_s0.log')]


def conn():
    pw = open(os.path.expanduser('~/.srvpro_env')).read().split("'")[1]
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect('140.123.105.18', username='basilrari', password=pw, timeout=25)
    return c


def run(c, cmd, read_t=25):
    _i, o, e = c.exec_command(cmd, timeout=120)
    o.channel.settimeout(read_t)
    try:
        return o.read().decode(errors='replace')
    except Exception as ex:
        return f'[timeout {type(ex).__name__}]'


def hamm(a, b):
    return sum(x != y for x, y in zip(a, b)) if len(a) == len(b) else None


def main():
    max_hours = 6.0
    t0 = time.time()
    while time.time() - t0 < max_hours * 3600:
        try:
            c = conn()
            n = run(c, "pgrep -fc 'hqp_solv[e]r|l2_absorb_pristin[e]' || echo 0").strip()
            line = [f'[{time.strftime("%H:%M")}] procs={n}']
            for tag, ref, out, log in JOBS:
                tail = run(c, f'tail -1 {log} 2>/dev/null')
                m = re.search(r'gate (\d+)/(\d+)', tail)
                if m:
                    line.append(f'{tag} {m.group(1)}/{m.group(2)}')
                elif 'end unswap' in tail or 'absorb' in tail.lower():
                    mm = re.search(r'(\d+)/(\d+)', tail)
                    line.append(f'{tag} ' + (f'{mm.group(1)}/{mm.group(2)}' if mm else 'active'))
            c.close()
            print(' '.join(line), flush=True)
            if n.strip() in ('0', ''):
                break
        except Exception as ex:
            print(f'[watch] error: {type(ex).__name__}: {ex}', flush=True)
        time.sleep(240)

    print('\n=== FINAL ===', flush=True)
    try:
        c = conn()
        for tag, ref, out, log in JOBS:
            print(f'\n--- {tag} ---', flush=True)
            if out:
                got = run(c, f'cat {out} 2>/dev/null || echo NOFILE').strip()
                print(' ' + got[:600], flush=True)
                try:
                    st = json.loads(got).get('peaked_state') or json.loads(got).get('state')
                    if st:
                        print(f'  H={hamm(st, TRUTH[ref])}/{len(TRUTH[ref])}', flush=True)
                except Exception:
                    pass
            else:
                print(' ' + run(c, f'tail -6 {log} 2>/dev/null')[:600], flush=True)
        c.close()
    except Exception as ex:
        print(f'final error: {ex}', flush=True)
    print('\nDONE', flush=True)


if __name__ == '__main__':
    main()
