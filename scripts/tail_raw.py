#!/usr/bin/env python3
"""Raw tails of the absorb logs -- how did each run actually finish?"""
import os
import sys

import paramiko

W = "/mnt/8tb_hdd2/basilrari/enigma-work"
names = sys.argv[1:] or [
    "l2_d1_s1_4043cafb_s123",
    "l2_d2_s1_39b370e4_s123",
    "l2_d3_s1_2674779a_tight1e7a",
]
REMOTE = "cd %s && for n in %s; do echo \"===== $n\"; tail -22 logs/$n.log; done" % (
    W,
    " ".join(names),
)


def main():
    pw = open(os.path.expanduser("~/.srvpro_env")).read().split("'")[1]
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect("140.123.105.18", username="basilrari", password=pw)
    _, o, e = c.exec_command(REMOTE, timeout=120)
    o.channel.recv_exit_status()
    print(o.read().decode())
    err = e.read().decode()
    if err.strip():
        print("STDERR:", err[:300])
    c.close()


if __name__ == "__main__":
    main()
