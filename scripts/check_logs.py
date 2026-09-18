#!/usr/bin/env python3
"""Tail the freshly launched job logs -- they hold the reason for any instant exit."""
import os
import sys

import paramiko

pw = open(os.path.expanduser("~/.srvpro_env")).read().split("'")[1]
W = "/mnt/8tb_hdd2/basilrari/enigma-work"
names = sys.argv[1:] or ["l2_d1_s1_4043cafb_cpuctl.log", "l2_d2_s1_39b370e4_cpu123.log"]


def main():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect("140.123.105.18", username="basilrari", password=pw)
    for n in names:
        for cmd in (f"wc -l {W}/logs/{n} 2>&1", f"tail -25 {W}/logs/{n} 2>&1"):
            _, o, e = c.exec_command(cmd, timeout=90)
            o.channel.recv_exit_status()
            out = o.read().decode().strip()
            err = e.read().decode().strip()
            print(f"$ {cmd}")
            print((out or "(empty)")[:2500])
            if err:
                print("  stderr:", err[:300])
            print()
    c.close()


if __name__ == "__main__":
    sys.exit(main())
