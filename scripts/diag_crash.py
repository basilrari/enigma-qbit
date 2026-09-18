#!/usr/bin/env python3
"""Diagnose the NaN crash in the absorb: how far did each job get, and where.

Prints per log: line count, the last progress markers before the crash, the
exception type, whether a verdict was ever emitted, and any norm/bond warnings
that would explain the divergence.
"""
import os
import sys

import paramiko

W = "/mnt/8tb_hdd2/basilrari/enigma-work"
LOGS = sys.argv[1:] or ["l2_d1_s1_4043cafb_cpuctl.log",
                        "l2_d2_s1_39b370e4_cpu123.log"]

CMDS = [
    ("lines", "wc -l {p}"),
    ("verdict hits", "grep -c -iE 'VERDICT|BURIED|DISCARD' {p} || true"),
    ("exception", "grep -nE '^[A-Za-z]*(Error|Exception|Warning):' {p} | tail -3"),
    ("progress tail", "grep -nE 't_u|t_u_l|t_u_r|swap|absorb|bond|layer|norm|chi' {p} | tail -12"),
    ("raw tail", "tail -6 {p}"),
]


def main():
    pw = open(os.path.expanduser("~/.srvpro_env")).read().split("'")[1]
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect("140.123.105.18", username="basilrari", password=pw)
    for name in LOGS:
        print(f"\n######## {name} ########")
        p = f"{W}/logs/{name}"
        for label, tmpl in CMDS:
            cmd = tmpl.format(p=p)
            _, o, e = c.exec_command(cmd, timeout=60)
            o.channel.recv_exit_status()
            out = o.read().decode().strip()
            err = e.read().decode().strip()
            print(f"-- {label}:")
            print((out or "(none)")[:1800])
            if err and label == "lines":
                print("   err:", err[:200])
    c.close()


if __name__ == "__main__":
    sys.exit(main())
