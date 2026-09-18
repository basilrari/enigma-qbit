#!/usr/bin/env python3
"""Check save API + poll all L2 runs."""
import os

import paramiko

W = "/mnt/8tb_hdd2/basilrari/enigma-work"

CHK = '''import numpy as np, pickle, quimb.tensor as qtn
tn = qtn.MatrixProductState([np.ones((1,2,2)), np.ones((2,2,1))], site_ind_id="k{}")
print("has save_to_disk:", hasattr(tn, "save_to_disk"), "| has to_disk:", hasattr(tn, "to_disk"))
print("pickle rounds:", len(pickle.dumps(tn)))
try:
    tn.save_to_disk("/tmp/_t.h5"); print("save_to_disk CALL OK")
except Exception as e:
    print("save_to_disk FAILS:", type(e).__name__, str(e)[:120])
'''

pw = open(os.path.expanduser("~/.srvpro_env")).read().split("'")[1]
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("140.123.105.18", username="basilrari", password=pw)
s = c.open_sftp()
with s.open("/tmp/chk_save.py", "w") as fh:
    fh.write(CHK)
s.close()


def run(x, t=180):
    _i, o, e = c.exec_command(x, timeout=t)
    o.channel.recv_exit_status()
    return (o.read().decode() + e.read().decode()).strip()


print("=== save API ===")
print(run(f"cd /tmp && {W}/l2venv/bin/python /tmp/chk_save.py 2>&1 | tail -6"))

print("\n=== progress ===")
print(run(
    f'for f in {W}/logs/l2_d2*_s123.log {W}/logs/l2_d2*tight1e7.log '
    f'{W}/logs/l2_d3*tight1e7a.log {W}/logs/l2_d1*_s123.log; do '
    f'echo "-- $(basename $f)"; grep -ac "INFO" $f; '
    f'grep -a "start unswap" $f | wc -l; tail -1 $f | cut -c1-100; done'))
c.close()
