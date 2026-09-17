#!/usr/bin/env python3
"""Final collection: exactness validation, gap curve, chi=512 certified optima."""
import os
import time

import paramiko

V = "/mnt/8tb_hdd2/basilrari/enigma-work/verify"
PW = open(os.path.expanduser("~/.srvpro_env")).read().split("'")[1]

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("140.123.105.18", username="basilrari", password=PW)


def run(x, t=540):
    i, o, e = c.exec_command(x, timeout=t)
    o.channel.recv_exit_status()
    return o.read().decode()


PAT = r"P\(known\)|ex#|certified|ratio_vs_peak|H=|uniform|VERDICT|ANSWER|complete"

print("waiting for the d1_s1 chi=128 validation run ...")
for _ in range(9):
    done = run(f"cd {V} && grep -ac 'VERDICT' abnew_c128.log")
    if done.strip() not in ("0", ""):
        break
    time.sleep(60)

print("=== EXACTNESS GATE: new optimised engine, d1_s1 chi=128 ===")
print("(old engine on the same command gave: ex#1 P=2.516320e-03 H=0)")
print(run(f"cd {V} && grep -aE \"{PAT}\" abnew_c128.log | tail -10").strip())
print("build wall-clock:", run(
    f"cd {V} && grep -a 'build done' abnew_c128.log | tail -1").strip())

print("\n=== CHI=512 CERTIFIED OPTIMA (the decisive point) ===")
for lg in ["argmax_d2_s1_39b370e4_c512", "argmax_d3_s1_2674779a_c512"]:
    print(f"[{lg}]")
    print(run(f"cd {V} && grep -aE \"{PAT}|gate [0-9]+/\" {lg}.log "
              f"| tail -6").strip() or "(running)")
    print(run(f"cd {V} && tail -1 {lg}.log | cut -c1-120").strip())

print("\n=== GAP CURVE (heuristic best vs known peak, same model) ===")
for lg in ["gap_d2_c64", "gap_d2_c96", "gap_d2_c192", "gap_d2_c256",
           "gap_d2_c384", "gap_d3_c96", "gap_d3_c192", "gap_d3_c256"]:
    g = run(f"cd {V} && grep -aoE 'gate [0-9]+/[0-9]+' {lg}.log | tail -1")
    r = run(f"cd {V} && grep -a 'ratio_vs_peak' {lg}.log | tail -1")
    h = run(f"cd {V} && grep -aE 'after polish|VERDICT' {lg}.log | tail -1")
    print(f"{lg:<14} {g.strip():<18} {r.strip()[:60]:<60} {h.strip()[:44]}")
c.close()
