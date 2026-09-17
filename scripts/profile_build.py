#!/usr/bin/env python3
"""Where does the MPS build actually spend its time?

Why this matters: the ladder's ceiling is set by build cost, and build cost
scales ~chi^3.  A 10x build speedup is worth a ~2.2x larger chi -- the
difference between chi=512 and chi=1000, i.e. between "peak barely emerging"
and "peak found".  The measured SVD cost at chi=256 is ~8 ms, but the real
build runs ~1 s per gate, so ~99% of the wall time is NOT the SVD.  This
profiles the build to find where it actually goes.
"""
import cProfile
import io
import pstats
import sys
import time

sys.path.insert(0, "/mnt/8tb_hdd2/basilrari/enigma-work/verify")
import numpy as np  # noqa: E402
import hqp_solver as HS  # noqa: E402

CHI = int(sys.argv[2]) if len(sys.argv) > 2 else 192
cid = sys.argv[1] if len(sys.argv) > 1 else "d1_s1_4043cafb"
VER = "/mnt/8tb_hdd2/basilrari/enigma-work/verify"

circ = HS.load_circuit(f"{VER}/{cid}.qasm")
nq = HS.nq(circ) if hasattr(HS, "nq") else None
print(f"profiling build {cid} chi={CHI} qubits={nq}", flush=True)

pr = cProfile.Profile()
t0 = time.time()
pr.enable()
try:
    tensors, order, q_at = HS.build_mps(circ, CHI, verbose=False)
except Exception as ex:
    pr.disable()
    print(f"build failed: {type(ex).__name__}: {ex}")
    raise
pr.disable()
dt = time.time() - t0
print(f"BUILD took {dt:.1f}s  ({(dt/4400)*1000:.0f} ms per gate at 4400 gates)")

s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats("tottime")
ps.print_stats(18)
out = s.getvalue()
print("---- top by TOTTIME ----")
for line in out.split("\n"):
    print(line)
