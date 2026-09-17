#!/usr/bin/env python3
"""Contraction width of a SINGLE amplitude for each Enigma circuit.

Builds the space-time tensor network of one output amplitude (all output
legs fixed) and asks cotengra for a contraction path, reporting:
   width   = log2(largest intermediate)   <- MEMORY exponent
   flops   = log10(total scalar mults)    <- TIME exponent

Decision rule:
   width <= 31  -> exact contraction fits in 96 GB with slicing: NEW WEAPON
   width >= 36  -> memory-infeasible, MPS is the only route

usage: tn_width.py <circuit_id> [max_time_seconds]
"""
import sys, os, re, json, time, math
import cotengra as ctg

SC = "/mnt/8tb_hdd2/basilrari/enigma-work/verify"
CIDS = ["d1_s1_4043cafb", "d2_s1_39b370e4", "d3_s1_2674779a"]


def build_tn(path):
    """einsum (inputs, output, size_dict) for <x|C|0>, all legs contracted."""
    txt = open(path).read()
    m = re.search(r"qreg\s+\w+\[(\d+)\]", txt) or re.search(r"qubit\[(\d+)\]\s+\w+", txt)
    n = int(m.group(1))
    size = {}
    nxt = [0]

    def fresh():
        i = nxt[0]
        nxt[0] += 1
        size[i] = 2
        return i

    wires = [None] * n
    inputs = []
    for q in range(n):
        w = fresh()
        wires[q] = w
        inputs.append((w,))                     # |0> boundary vector
    for raw in txt.splitlines():
        line = raw.strip()
        if not line or line.startswith(("//", "OPENQASM", "include", "qreg", "creg",
                                        "qubit", "bit", "output", "input")):
            continue
        mu = re.match(r"u\(([^)]*)\)\s+q\[(\d+)\]", line)
        if mu:
            q = int(mu.group(2))
            wi = wires[q]
            wo = fresh()
            wires[q] = wo
            inputs.append((wi, wo))             # rank-2 single-qubit gate
            continue
        mc = re.match(r"cz\s+q\[(\d+)\]\s*,\s*q\[(\d+)\]", line)
        if mc:
            a, b = int(mc.group(1)), int(mc.group(2))
            ai, bi = wires[a], wires[b]
            ao, bo = fresh(), fresh()
            wires[a], wires[b] = ao, bo
            inputs.append((ai, ao, bi, bo))     # rank-4 CZ
            continue
    for q in range(n):
        inputs.append((wires[q],))              # <x| boundary vector
    return inputs, (), size, n


def report(cid, maxt):
    t0 = time.time()
    inputs, out, size, n = build_tn(f"{SC}/{cid}.qasm")
    n2 = sum(1 for k in inputs if len(k) == 4)
    print(f"\n=== {cid}  n={n}  tensors={len(inputs)}  cz={n2}  "
          f"indices={len(size)}  build={time.time()-t0:.1f}s", flush=True)
    opt = ctg.HyperOptimizer(max_time=maxt, progbar=False,
                             minimize="flops", parallel="auto")
    tree = opt.search(inputs, out, size)
    li = tree.largest_intermediate()
    fl = tree.total_flops()
    width = math.log2(li)
    print(f"  largest intermediate = {li:.4g} entries "
          f"-> width = {width:.2f}  (mem {li*16/2**30:.4g} GiB @ complex128)", flush=True)
    print(f"  total flops = {fl:.4g}  -> log10 = {math.log10(fl):.2f}", flush=True)
    try:
        import numpy as np
        print(f"  tree.contraction_width() = {tree.contraction_width()}", flush=True)
    except Exception as ex:
        print(f"  (contraction_width() unavailable: {ex})", flush=True)
    res = {"cid": cid, "n": n, "tensors": len(inputs), "cz": n2,
           "width": width, "largest_intermediate": li,
           "log10_flops": math.log10(fl), "seconds": time.time() - t0}
    with open(f"{SC}/tnwidth_{cid}.json", "w") as f:
        json.dump(res, f, indent=1)
    verdict = ("EXACT CONTRACTION VIABLE (with slicing)" if width <= 31 else
               "MEMORY-INFEASIBLE" if width >= 36 else "BORDERLINE")
    print(f"  VERDICT: {verdict}", flush=True)


if __name__ == "__main__":
    maxt = float(sys.argv[1]) if len(sys.argv) > 1 else 240.0
    for cid in CIDS:
        try:
            report(cid, maxt)
        except Exception as ex:
            import traceback
            print(f"*** {cid} FAILED: {type(ex).__name__}: {ex}", flush=True)
            traceback.print_exc()
