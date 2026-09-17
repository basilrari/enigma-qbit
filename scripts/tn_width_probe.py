#!/usr/bin/env python3
"""DECISIVE MEASUREMENT: what is the true tensor-network contraction width
for a SINGLE amplitude of each Enigma circuit?

If width <= ~31, exact contraction (with slicing to fit memory) becomes
affordable -> we get truncation-free amplitudes = a real oracle.
If width >= ~40, exact methods are permanently dead and only MPS remains.

Builds the amplitude TN directly from the QASM (1q gate = rank2, cz = rank4,
boundary = rank1) and asks cotengra for an optimised contraction tree.
"""
import re, sys, json, time
import cotengra as ctg

SC = "/mnt/8tb_hdd2/basilrari/enigma-work/verify"


def build_tn(path):
    txt = open(path).read()
    m = re.search(r"qreg\s+\w+\[(\d+)\]", txt)
    n = int(m.group(1))
    size_dict = {}
    nxt = [0]

    def newwire():
        i = nxt[0]
        nxt[0] += 1
        size_dict[i] = 2
        return i

    inputs = []
    wires = [None] * n
    for q in range(n):
        w = newwire()
        wires[q] = w
        inputs.append((w,))  # <0| boundary

    for line in txt.splitlines():
        line = line.strip()
        mu = re.match(r"u\([^)]*\)\s+q\[(\d+)\]", line)
        if mu:
            q = int(mu.group(1))
            wi = wires[q]
            wo = newwire()
            wires[q] = wo
            inputs.append((wi, wo))
            continue
        mc = re.match(r"cz\s+q\[(\d+)\]\s*,\s*q\[(\d+)\]", line)
        if mc:
            a, b = int(mc.group(1)), int(mc.group(2))
            ai, bi = wires[a], wires[b]
            ao, bo = newwire(), newwire()
            wires[a], wires[b] = ao, bo
            inputs.append((ai, ao, bi, bo))
            continue

    for q in range(n):
        inputs.append((wires[q],))  # <x| boundary (bitstring fixed => vector)
    return inputs, (), size_dict, n


def report(tag, path):
    t0 = time.time()
    inputs, output, size_dict, n = build_tn(path)
    ng = len(inputs)
    print(f"[{tag}] n_qubits={n} n_tensors={ng} n_indices={len(size_dict)}", flush=True)

    res = {"tag": tag, "n_qubits": n, "n_tensors": ng}
    try:
        opt = ctg.HyperOptimizer(
            max_time=180, max_repeats=32, progbar=False,
            minimize="flops", slicing_opts=None,
        )
        tree = opt.search(inputs, output, size_dict)
        li = tree.largest_intermediate()
        fl = tree.total_flops()
        import math
        res["largest_intermediate_entries"] = li
        res["log2_largest_intermediate"] = round(math.log2(max(li, 1)), 2)
        res["total_flops"] = str(fl)
        res["log10_total_flops"] = round(math.log10(max(float(fl), 1.0)), 2)
        try:
            res["contraction_width"] = float(tree.contraction_width())
        except Exception as e:
            res["contraction_width_err"] = str(e)[:80]
        try:
            res["max_size"] = int(tree.max_size())
        except Exception:
            pass
        # memory of the biggest intermediate in GB (complex128 = 16 B)
        res["peak_intermediate_GB"] = round(li * 16 / 1e9, 3)
        # sliced variant: reduce biggest intermediate under a 40 GB budget
        try:
            import math
            need = math.ceil(math.log2(li * 16 / 40e9))
            if need > 0:
                res["slices_needed_for_40GB"] = 2 ** need
                res["sliced_flops_est"] = fl * (2 ** need)
                res["log10_sliced_flops"] = round(math.log10(fl * (2 ** need)), 2)
        except Exception:
            pass
    except Exception as e:
        res["error"] = f"{type(e).__name__}: {e}"[:300]

    res["seconds"] = round(time.time() - t0, 1)
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    for cid in sys.argv[1:]:
        report(cid, f"{SC}/{cid}.qasm")
