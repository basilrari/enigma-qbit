#!/usr/bin/env python3
"""Pool test: is the true answer even present in a low-chi model's top-200?

If the truth appears among the certified top-K optima of the cheap chi=64
model, then an ensemble over chi (pool the candidates, re-rank by how their
probability behaves across the ladder) can find it.  If it does not appear,
that method is dead before we build it and we must invent again.
"""
import os

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


print("=== POOL TEST: top-200 certified optima at chi=64 ===")
for lg in ["pool_d2_c64_top200", "pool_d3_c64_top200"]:
    print(f"\n--- {lg} ---")
    print(run(f"cd {V} && tail -1 {lg}.log | cut -c1-110").strip())
    n_ex = run(f"cd {V} && grep -ac 'ex#' {lg}.log").strip()
    per = run(f"cd {V} && grep -a 'ex#' {lg}.log | head -3").strip()
    print(f"exact optima enumerated: {n_ex}")
    print(per)
    # Does the truth (H=0) appear anywhere in the enumerated optima?
    h0 = run(f"cd {V} && grep -a 'ex#' {lg}.log | grep -c 'H=0 '")
    md = run(f"cd {V} && grep -a 'ex#' {lg}.log | grep -aoE 'H=[0-9]+' "
             f"| sort | uniq -c | sort -k2 -t= -n | head -8")
    print(f"entries with H=0 (the true answer): {h0.strip()}")
    print("Hamming-distance histogram of the pool:")
    print(md.strip())

print("\n=== GAP CURVE (heuristic best / known peak, same model) ===")
for lg in ["gap_d2_c192", "gap_d2_c256", "gap_d2_c384", "gap_d3_c192",
           "gap_d3_c256", "gap_d3_c384"]:
    g = run(f"cd {V} && grep -aoE 'gate [0-9]+/[0-9]+' {lg}.log | tail -1")
    r = run(f"cd {V} && grep -a 'ratio_vs_peak' {lg}.log | tail -1")
    print(f"{lg:<14} {g.strip():<18} {r.strip()[:55]}")

print("\n=== CHI=512 BUILDS ===")
for lg in ["argmax_d2_s1_39b370e4_c512", "argmax_d3_s1_2674779a_c512"]:
    print(f"{lg:<34}", run(f"cd {V} && tail -1 {lg}.log | cut -c1-100").strip())
c.close()
