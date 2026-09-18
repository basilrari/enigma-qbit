#!/usr/bin/env python3
"""Extract the peak by combinatorial search with an EXACT amplitude oracle.

This is not state simulation.  Every |amplitude|^2 here is exact -- no
truncation, no chi, no bond.  So an improvement is a real improvement; there is
no estimator noise to chase, which is what every previous approach suffered
from.

Search in log|amp|^2 space: a peaked instance has amplitudes ~1e-1 while a
random string sits at ~2^-n, so raw differences make a useless temperature.

Moves, per step:
  * with probability P1F: steepest ascent over ALL n single flips, each
    evaluated exactly.  That is a true gradient on a noise-free landscape.
  * otherwise: a random 2-flip, to escape 1-flip local optima.
Acceptance is simulated annealing with geometric cooling.

Parallelism: the contraction *tree* and the built tensor network are inherited
by workers through fork, so the expensive path search happens exactly once.

Validation: d1's answer is known, so --truth prints the Hamming distance.  A
search that walks from a random start to d1 would prove the method works at
all, before spending anything on d2/d3.
"""
import argparse
import math
import multiprocessing as mp
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from amp_oracle import ReusableOracle, parse_qasm  # noqa: E402

LOG_FLOOR = -300.0  # ~1e-130; below any amplitude this hardware will ever see
_ORACLE = None      # inherited by fork()
_N = 0


def _flip(b, q):
    return b[:q] + ("1" if b[q] == "0" else "0") + b[q + 1:]


def _logp(bits):
    amp, _ = _ORACLE.amplitude(bits)
    p = abs(amp) ** 2
    return (math.log(p) if p > 0.0 else LOG_FLOOR), p


def _run(job):
    start, steps, t_hi, t_lo, seed, p1f = job
    rng = np.random.default_rng(seed)
    b = start
    cur, curp = _logp(b)
    best, bestp = b, curp
    evals = 1
    traj = []
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        T = t_hi * (t_lo / t_hi) ** frac
        if rng.random() < p1f:
            # exact gradient: the whole 1-flip neighbourhood
            cand = [_logp(_flip(b, q)) + (q,) for q in range(_N)]
            evals += _N
            lp, pp, q = max(cand)
            prop = _flip(b, q)
        else:
            qs = rng.choice(_N, size=2, replace=False)
            prop = _flip(_flip(b, int(qs[0])), int(qs[1]))
            lp, pp = _logp(prop)
            evals += 1
        if lp >= cur or rng.random() < math.exp(min((lp - cur) / max(T, 1e-12), 60.0)):
            b, cur, curp = prop, lp, pp
        if curp > bestp:
            best, bestp = b, curp
            traj.append((s, curp, b))
        if s % max(steps // 8, 1) == 0:
            print(f"    step {s}/{steps} T={T:.3f} best={bestp:.4e} "
                  f"bits={best}", flush=True)
    return {"best": best, "p": bestp, "evals": evals, "traj": traj}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qasm", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--restarts", type=int, default=0,
                    help="extra random starts (default: workers-1)")
    ap.add_argument("--seeds", default=None, help="comma-separated start strings")
    ap.add_argument("--bits-file", default=None, help="one start string per line")
    ap.add_argument("--truth", default=None, help="known answer, for Hamming check")
    ap.add_argument("--slice", type=int, default=24, dest="slice_bits")
    ap.add_argument("--reps", type=int, default=16)
    ap.add_argument("--p1f", type=float, default=0.75,
                    help="probability of a full 1-flip gradient step")
    ap.add_argument("--t-hi", type=float, default=3.0)
    ap.add_argument("--t-lo", type=float, default=0.02)
    ap.add_argument("--random-start", action="store_true",
                    help="ignore --seeds/--bits-file, start from random strings")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    n, gates, _ = parse_qasm(args.qasm)
    if not gates:
        print("[skip] no u/cz gates in this file")
        return
    print(f"[search] {n} qubits, {len(gates)} gates, {args.workers} workers, "
          f"{args.steps} steps each", flush=True)

    global _ORACLE, _N
    _ORACLE = ReusableOracle(n, gates, args.slice_bits, reps=args.reps)
    _N = n

    starts = []
    if not args.random_start:
        if args.bits_file:
            starts += [l.strip() for l in open(args.bits_file) if l.strip()]
        if args.seeds:
            starts += [s.strip() for s in args.seeds.split(",") if s.strip()]
    starts = [s for s in starts if len(s) == n]
    rng = np.random.default_rng(0)
    nrand = args.restarts if args.restarts else max(args.workers - len(starts), 0)
    for _ in range(nrand):
        starts.append("".join(rng.integers(0, 2, n).astype(str)))
    if not starts:
        starts = ["0" * n]
    print(f"[starts] {len(starts)}", flush=True)
    for s in starts:
        lp, p = _logp(s)
        print(f"    start |amp|^2={p:.4e} bits={s}", flush=True)

    jobs = [(s, args.steps, args.t_hi, args.t_lo, 1000 + i, args.p1f)
            for i, s in enumerate(starts)]
    t0 = time.time()
    if args.workers > 1 and len(jobs) > 1:
        with mp.Pool(processes=min(args.workers, len(jobs))) as pool:
            results = pool.map(_run, jobs)
    else:
        results = [_run(j) for j in jobs]
    dt = time.time() - t0

    results.sort(key=lambda r: -r["p"])
    tot = sum(r["evals"] for r in results)
    print(f"\n[done] {dt:.1f}s, {tot} exact amplitudes, "
          f"{tot/max(dt,1e-9):.1f}/s", flush=True)
    for r in results[:5]:
        print(f"    |amp|^2={r['p']:.6e} evals={r['evals']} bits={r['best']}")
    win = results[0]
    print(f"\n[PEAK] |amp|^2={win['p']:.6e}\n       bits={win['best']}")
    if args.truth and len(args.truth) == n:
        h = sum(a != b for a, b in zip(win["best"], args.truth))
        print(f"[verify] Hamming to known answer = {h}/{n}")
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(win["best"] + "\n")
        print(f"[out] {args.out}")


if __name__ == "__main__":
    main()
