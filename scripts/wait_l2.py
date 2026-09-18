#!/usr/bin/env python3
"""Watch the two L2 absorb jobs and print their verdicts when they finish.

Polls the box every 3 minutes; exits as soon as no absorb process remains, or
after MAX_HOURS.  Prints the verdict-bearing lines from both logs.
"""
import os
import sys
import time

import paramiko

W = "/mnt/8tb_hdd2/basilrari/enigma-work"
V = f"{W}/verify"
TAGS = ["l2_d2_b256", "l2_d1_b256"]
MAX_HOURS = 3.0


def main():
    pw = open(os.path.expanduser("~/.srvpro_env")).read().split("'")[1]
    t0 = time.time()
    while time.time() - t0 < MAX_HOURS * 3600:
        try:
            c = paramiko.SSHClient()
            c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            c.connect("140.123.105.18", username="basilrari", password=pw, timeout=30)
            _, o, _ = c.exec_command("pgrep -fc 'l2_absorb.py'", timeout=60)
            o.channel.recv_exit_status()
            n = o.read().decode().strip()
            _, o, _ = c.exec_command(
                f"tail -1 {V}/logs/l2_d2_b256.log | cut -c1-120", timeout=60)
            o.channel.recv_exit_status()
            last = o.read().decode().strip()
            print(f"[{time.strftime('%H:%M:%S')}] running={n} | {last}", flush=True)
            if n == "0":
                break
            c.close()
        except Exception as exc:
            print(f"[{time.strftime('%H:%M:%S')}] poll error: {type(exc).__name__}",
                  flush=True)
        time.sleep(180)

    print("\n" + "=" * 60)
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect("140.123.105.18", username="basilrari", password=pw)
    for tag in TAGS:
        _, o, _ = c.exec_command(
            f"grep -aE 'VERDICT|ANSWER|absorb:|H=|peak|buried|BURIED|Traceback|Error' "
            f"{V}/logs/{tag}.log | tail -14", timeout=90)
        o.channel.recv_exit_status()
        print(f"\n===== {tag} =====")
        print(o.read().decode().strip()[:1500])
    c.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
