#!/usr/bin/env python3
"""Aggregate every L2 ordering into a verdict per instance.

Reads all {W}/logs/l2_<inst>_*.log, extracts from each:
    ### amp2(truth) = <a2>   (reversed site order: <a2r>)
    ### VERDICT: BURIED|DISCARDED
    ### beam_search candidates ...  #1 w=... bits=... H=<h>/<n> (reversed-truth <hr>)
    # total <s>s

Confidence rule (dossier enigma-sn63.md):
    HIGH  : >=4 independent orderings agree on the same top-1 bitstring
            OR one ordering with margin w1/w2 >= 5
    else  : fail-closed -> report best effort, do not claim a solve
"""
import os
import re

import paramiko

W = "/mnt/8tb_hdd2/basilrari/enigma-work"
INSTS = ["d1_s1_4043cafb", "d2_s1_39b370e4", "d2_s2_1efabaf4", "d3_s1_2674779a", "d3_s2_c09ba537"]

RE_AMP = re.compile(r"### amp2\(truth\) = ([\d.eE+-]+)\s+\(reversed site order: ([\d.eE+-]+)\)")
RE_RAT = re.compile(r"truth is ([\d.eE+-]+)x uniform")
RE_VER = re.compile(r"### VERDICT: (\w+)")
RE_B1 = re.compile(r"#1 w=([\d.eE+-]+) bits=(\S+)\s+H=(-?\d+)/(\d+)")
RE_B2 = re.compile(r"#2 w=([\d.eE+-]+)")
RE_TOT = re.compile(r"# total (\d+)s")

pw = open(os.path.expanduser("~/.srvpro_env")).read().split("'")[1]
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("140.123.105.18", username="basilrari", password=pw)


def read(path):
    _i, o, e = c.exec_command(f"cat {path}", timeout=120)
    o.channel.recv_exit_status()
    return o.read().decode()


_i, o, _e = c.exec_command(f"ls {W}/logs/", timeout=60)
o.channel.recv_exit_status()
files = [f for f in o.read().decode().split() if f.startswith("l2_")]

rows = {}
for f in sorted(files):
    txt = read(f"{W}/logs/{f}")
    m1 = RE_AMP.search(txt)
    if not m1:
        rows.setdefault(f, {"state": "running/absorbing"})
        continue
    d = {
        "state": "done",
        "amp2": float(m1.group(1)),
        "amp2_rev": float(m1.group(2)),
        "ratio": float(RE_RAT.search(txt).group(1)) if RE_RAT.search(txt) else float("nan"),
        "verdict": RE_VER.search(txt).group(1) if RE_VER.search(txt) else "?",
        "total_s": int(RE_TOT.search(txt).group(1)) if RE_TOT.search(txt) else -1,
    }
    b1 = RE_B1.search(txt)
    if b1:
        d["w1"] = float(b1.group(1))
        d["bits"] = b1.group(2)
        d["h"] = int(b1.group(3))
        d["n"] = int(b1.group(4))
        b2 = RE_B2.search(txt)
        d["margin"] = d["w1"] / float(b2.group(1)) if b2 else float("inf")
    rows[f] = d

print(f"{'log':46s} {'state':10s} {'amp2(truth)':>12s} {'xUnif':>10s} {'verdict':>10s} {'w1':>10s} {'marg':>7s} H")
for f, d in rows.items():
    if d["state"] != "done":
        print(f"{f:46s} {d['state']}")
        continue
    print(f"{f:46s} {'done':10s} {d['amp2']:12.3e} {d['ratio']:10.2e} {d['verdict']:>10s} "
          f"{d.get('w1', float('nan')):10.3e} {d.get('margin', float('nan')):7.2f} "
          f"{d.get('h', '?')}/{d.get('n', '?')}")

# --- consensus ------------------------------------------------------------
print("\n=== consensus ===")
for inst in INSTS:
    got = {f: d for f, d in rows.items() if f.startswith(f"l2_{inst}") and d["state"] == "done"}
    if not got:
        continue
    votes = {}
    for f, d in got.items():
        if "bits" in d:
            votes.setdefault(d["bits"], []).append(f)
    top = max(votes.items(), key=lambda kv: len(kv[1])) if votes else (None, [])
    best_margin = max((d.get("margin", 0) for d in got.values()), default=0)
    hs = sorted(d.get("h", 99) for d in got.values())
    conf = "HIGH" if (len(top[1]) >= 4 or best_margin >= 5) else "LOW (fail-closed)"
    print(f"{inst}: orderings={len(got)} done | best H={hs[0] if hs else '?'} | "
          f"agree={len(top[1])} on {top[0]} | max margin={best_margin:.2f} | {conf}")
    for f, d in got.items():
        if "bits" in d:
            print(f"    {f.split('l2_')[1]:42s} amp2={d['amp2']:.2e} {d['verdict']:9s} "
                  f"H={d['h']}/{d['n']} margin={d.get('margin', float('nan')):.2f} {d['total_s']}s")
c.close()
