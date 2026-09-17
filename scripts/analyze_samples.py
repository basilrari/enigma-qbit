#!/usr/bin/env python3
"""Pull the sampling results off the box and report the heavy-hitter table.

Key question: does SAMPLING (heavy-hitter search) find the planted peak where
our old hill-climbing search did not?
"""
import os
import paramiko, json, sys, os

HOST, USER, PASS = os.environ.get("SRVPRO_PASS", ""), "basilrari", "iloveesl"
VER = "/mnt/8tb_hdd2/basilrari/enigma-work/verify"
LOCAL = "/home/basilsclaw/enigma-solve/sampling_results"
os.makedirs(LOCAL, exist_ok=True)

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, username=USER, password=PASS)
i, o, e = c.exec_command(f"ls {VER}/sample_*_c*.json 2>/dev/null")
o.channel.recv_exit_status()
files = [l.strip() for l in o.read().decode().split() if l.strip()]
sf = c.open_sftp()
rows = []
for f in files:
    loc = os.path.join(LOCAL, os.path.basename(f))
    try:
        sf.get(f, loc)
        rows.append(json.load(open(loc)))
    except Exception as ex:
        print("skip", f, ex)
sf.close()
c.close()

if not rows:
    print("no result JSONs yet")
    sys.exit()

print(f"{'circuit':<18} {'chi':>4} {'variant':>10} {'maxbond':>7} {'hits':>7} "
      f"{'true_rank':>9} {'modeH':>5} {'bestH':>5} {'distinct':>8}  "
      f"{'P(known)':>11}  {'P(known)/uniform':>17}")
print("-" * 148)
for r in sorted(rows, key=lambda x: (x["cid"], x["chi"], x.get("tag", ""))):
    top = r["top"]
    modeH = top[0]["h"] if top else -1
    bestH = min(t["h"] for t in top) if top else -1
    tr = r.get("true_rank", "-")
    uni = 2.0 ** (-r["qubits"])
    pk = r["mps_P_known"]
    print(f"{r['cid']:<18} {r['chi']:>4} {r.get('tag',''):>10} {r['maxbond']:>7} "
          f"{r['true_hits']:>7} {str(tr):>9} {modeH:>5} {bestH:>5} {r['distinct']:>8}  "
          f"{pk:>11.3e}  {pk/uni:>17.3e}")

print("\n--- detail ---")
for r in sorted(rows, key=lambda x: (x["cid"], x["chi"])):
    print(f"\n{r['cid']}  chi={r['chi']}  nsamples={r['nsamples']}  "
          f"qubits={r['qubits']}  maxbond={r['maxbond']}")
    print(f"  known       : {r['known']}")
    print(f"  MPS P(known): {r['mps_P_known']:.6e}")
    print(f"  TRUE ANSWER sampled {r['true_hits']} times "
          f"(freq {r['true_freq']:.6f}), rank {r.get('true_rank','not found')}")
    print(f"  mode        : {r['top'][0]['bits']}  (H={r['top'][0]['h']}, "
          f"freq {r['top'][0]['count']/r['nsamples']:.5f})")
