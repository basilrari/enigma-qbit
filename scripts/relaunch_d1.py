#!/usr/bin/env python3
"""Upload the fixed l2win runner and relaunch the d1 control detached."""
import os

import paramiko

W = '/mnt/8tb_hdd2/basilrari/enigma-work'
PY = f'{W}/l2venv/bin/python'
D1 = '0001001101001111101001001110010001111010100000'


def main():
    pw = open(os.path.expanduser('~/.srvpro_env')).read().split("'")[1]
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect('140.123.105.18', username='basilrari', password=pw, timeout=20)

    def run(cmd, t=90, read_t=20):
        _i, o, _e = c.exec_command(cmd, timeout=t)
        o.channel.settimeout(read_t)
        try:
            return o.read().decode(errors='replace')
        except Exception as ex:
            return f'[read timeout {type(ex).__name__}]'

    print('=== other jobs still running? ===')
    print(run("ps -eo etime,cmd | grep -E 'l2_absorb_pristin[e]|hqp_solv[e]r' | cut -c1-58"))
    print('=== recent logs ===')
    print(run(f'ls -t {W}/logs/ | head -6'))

    c.open_sftp().put('/home/basilsclaw/enigma-solve/scripts/l2win_gpu_run.py',
                      f'{W}/l2win_gpu_run.py')
    print('=== uploaded fixed runner ===')

    job = (f'cd {W} && exec env CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 {PY} '
           f'{W}/l2win_gpu_run.py {W}/verify/d1_s1_4043cafb.qasm 900 128 0.002 0 {D1} '
           f'> {W}/logs/l2win_d1_v3.log 2>&1')
    transport = c.get_transport()
    chan = transport.open_session()
    chan.exec_command(f'cd {W} && ({job}) & echo started')
    chan.settimeout(10)
    try:
        print('[launch]', chan.recv(200).decode(errors='replace').strip())
    except Exception as ex:
        print('[launch read timeout]', type(ex).__name__)
    c.close()


if __name__ == '__main__':
    main()
