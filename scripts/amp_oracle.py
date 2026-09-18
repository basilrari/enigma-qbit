#!/usr/bin/env python3
"""Exact single-amplitude oracle: <b|U|0> for one output bitstring.

Fundamentals.  Every approach so far has tried to represent the *whole state*
(MPS at bond chi, absorb, L2 unswap) and every wall has been there.  But the
challenge only ever asks for the peak -- one bitstring.  For a FIXED bitstring
the amplitude is a tensor-network contraction with both boundaries pinned,
which is the standard single-amplitude problem: exact, no truncation, and a
completely different cost profile from state simulation.

With an exact, cheap amplitude oracle, "find the peak" stops being state
simulation and becomes combinatorial search -- hill-climbing or annealing over
bitstrings where every evaluation is exact, and the truncation error that
buried the peak simply does not exist.

Conventions (Qiskit): string index i is qubit i; gates are u(theta,phi,lambda).
Gate matrices are imported from recover_perm so they match the solver exactly.

Usage:
  amp_oracle.py --qasm samples/d1_s1_4043cafb.qasm --bits 0001001101... \
                [--random 5] [--optimize greedy] [--width]
"""
import argparse
import math
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from recover_perm import _num, u_matrix  # same convention as the solver
    HAVE_UMATRIX = True
except Exception:  # pragma: no cover - fall back to the standard ZYZ form
    HAVE_UMATRIX = False

    def u_matrix(theta, phi, lam):
        ct, st = np.cos(theta / 2), np.sin(theta / 2)
        return np.array([
            [np.exp(-1j * (phi + lam) / 2) * ct,
             -np.exp(-1j * (phi - lam) / 2) * st],
            [np.exp(1j * (phi - lam) / 2) * st,
             np.exp(1j * (phi + lam) / 2) * ct],
        ], dtype=complex)

    def _num(s):
        """QASM writes parameters symbolically: 'pi', '-pi', 'pi/2'."""
        s = s.strip()
        try:
            return float(s)
        except ValueError:
            pass
        return float(eval(s.replace("pi", "PI"), {"__builtins__": {}},
                          {"PI": math.pi}))


UG = re.compile(r"^u\(([^)]*)\)\s+q\[(\d+)\];")
CZ = re.compile(r"^cz\s+q\[(\d+)\],q\[(\d+)\];")


def parse_qasm(path):
    """Return (n_qubits, gates).  gates are ('u', q, [params]) or ('cz', a, b)."""
    gates, unmatched, n = [], [], 0
    for raw in open(path):
        line = raw.split("//")[0].strip()
        if not line or line.startswith(("OPENQASM", "include", "barrier",
                                        "measure", "creg")):
            continue
        if line.startswith("qreg"):
            digits = "".join(c for c in line if c.isdigit())
            n = max(n, int(digits)) if digits else n
            continue
        m = UG.match(line)
        if m:
            gates.append(("u", int(m.group(2)),
                          [_num(v) for v in m.group(1).split(",")]))
            continue
        m = CZ.match(line)
        if m:
            gates.append(("cz", int(m.group(1)), int(m.group(2))))
            continue
        unmatched.append(line)
    return n, gates, unmatched


def build_tn(n, gates, bits):
    """Circuit TN with the input pinned to |0> and the output pinned to <bits|."""
    import quimb.tensor as qtn

    tensors = []
    frontier = [f"in{q}" for q in range(n)]
    # Monotonic counter, NOT len(tensors): reusing a small integer collides with
    # the initial wire names (e.g. the first gate's output index would be named
    # w39_0, identical to qubit 39's input wire, giving an index 3 uses).
    cid = 0

    for g in gates:
        if g[0] == "u":
            _, q, prm = g
            cid += 1
            out = f"w{q}_{cid}"
            # TRANSPOSE: this tensor's axis 0 is the *incoming* wire, but
            # u_matrix[i, j] has i as the output (row).  Attaching the incoming
            # wire to axis 0 without transposing applies U^T per gate -- which
            # silently corrupts every amplitude (found by amp_oracle_test.py).
            m = np.asarray(u_matrix(*prm), dtype=complex).T.reshape(2, 2)
            tensors.append(qtn.Tensor(m, (frontier[q], out), tags=[f"u{q}"]))
            frontier[q] = out
        else:
            _, a, b = g
            cid += 1
            oa, ob = f"w{a}_{cid}", f"w{b}_{cid}"
            m = np.zeros((2, 2, 2, 2), dtype=complex)
            for i in range(2):
                for j in range(2):
                    m[i, j, i, j] = 1.0 if not (i and j) else -1.0
            tensors.append(qtn.Tensor(m, (frontier[a], frontier[b], oa, ob),
                                      tags=[f"cz{a},{b}"]))
            frontier[a], frontier[b] = oa, ob

    # boundaries: |0> on each input wire, <bits| on each output wire
    ket0 = np.array([1.0, 0.0], dtype=complex)
    for q in range(n):
        tensors.append(qtn.Tensor(ket0, (f"in{q}",), tags=[f"in{q}"]))
        bra = np.zeros(2, dtype=complex)
        bra[int(bits[q])] = 1.0
        tensors.append(qtn.Tensor(bra, (frontier[q],), tags=[f"out{q}"]))

    return qtn.TensorNetwork(tensors)


def optimizer_for(slice_bits, minimize="flops"):
    """cotengra optimizer that slices large indices to bound memory.

    Without slicing the greedy path tried a 4 TiB intermediate on d1.  Slicing
    turns the single amplitude into 2**slice_bits small contractions summed
    afterwards -- the standard way a single amplitude is computed classically.
    """
    import cotengra as ctg
    return ctg.HyperOptimizer(
        max_repeats=32,
        progbar=False,
        minimize=minimize,
        slicing_opts={"target_size": 2 ** slice_bits},
    )


def amplitude(n, gates, bits, optimize="greedy", slice_bits=24, verbose=False):
    tn = build_tn(n, gates, bits)
    if verbose:
        try:
            print(f"    [tn] tensors={tn.num_tensors} "
                  f"width={tn.contraction_width():.1f} "
                  f"log2cost={tn.contraction_cost():.1f}")
        except Exception:
            pass
    opt = optimizer_for(slice_bits) if slice_bits else optimize
    t0 = time.time()
    # No progbar kwarg: quimb forwards extra kwargs into cotengra, and this
    # cotengra version rejects progbar with TypeError.
    val = tn.contract(optimize=opt)
    return complex(val), time.time() - t0


class ReusableOracle:
    """Pick the contraction path ONCE, then evaluate many bitstrings cheaply.

    The path depends only on the network *structure*, which is identical for
    every bitstring -- only the 2n boundary vectors change.  Re-searching per
    string (what amplitude() does) is pure waste: it spent minutes on a single
    d1 amplitude.

    This is the enabling measurement.  If one amplitude costs minutes, no
    search is possible and the method is dead.  If it costs milliseconds,
    finding the peak becomes a combinatorial problem we can actually attack.
    """

    def __init__(self, n, gates, slice_bits=24, reps=32, verbose=True):
        import cotengra as ctg
        self.n, self.gates, self.ctg = n, gates, ctg
        # dummy bits: only the boundary *values* get overwritten later
        self.tn = build_tn(n, gates, "0" * n)
        self.tensors = list(self.tn.tensors)
        self.inds = [list(map(str, t.inds)) for t in self.tensors]
        arrays = [t.data for t in self.tensors]

        opt = (ctg.HyperOptimizer(max_repeats=reps, progbar=False,
                                  slicing_opts={"target_size": 2 ** slice_bits})
               if slice_bits else "greedy")
        # cotengra wants (inputs, output, size_dict) -- NOT the arrays.  Passing
        # arrays where the einsum inputs belong makes every trial fail with
        # KeyError: 'tree'.  Every index here is a qubit wire, so size 2.
        size_dict = {}
        for t in self.tensors:
            for i in t.inds:
                size_dict[str(i)] = 2
        t0 = time.time()
        try:
            self.tree = opt.search(self.inds, [], size_dict)
        except Exception as exc:
            err = None
            try:
                err = opt.best.get("error")
            except Exception:
                pass
            raise RuntimeError(f"path search failed: {err or exc}") from exc
        self.search_s = time.time() - t0

        try:
            self.cost = float(self.tree.contraction_cost())
            self.width = float(self.tree.contraction_width())
        except Exception:
            self.cost = self.width = float("nan")
        self.sliced = tuple(str(i) for i in getattr(self.tree, "sliced_inds", ()))
        self.nslices = 2 ** len(self.sliced)

        # Probe which contraction call this cotengra build actually supports.
        self.mode, self._fn = None, None
        for name, fn in (
            ("tree.contract", lambda a: self.tree.contract(a)),
            ("ctg.array_contract", lambda a: ctg.array_contract(
                a, self.inds, [], optimize=self.tree, progbar=False)),
            ("tn+path", lambda a: self.tn.contract(
                optimize=self.tree.get_path())),
        ):
            try:
                fn(arrays)
                self.mode, self._fn = name, fn
                break
            except Exception:
                continue
        if self.mode is None:
            raise RuntimeError("no working contraction call for this cotengra")
        if verbose:
            print(f"[path] search {self.search_s:.1f}s via {self.mode} "
                  f"log2flops={self.cost:.1f} log2width={self.width:.1f} "
                  f"slices=2**{len(self.sliced)}", flush=True)

    def _vector(self, bit):
        v = np.zeros(2, dtype=complex)
        v[int(bit)] = 1.0
        return v

    def amplitude(self, bits):
        """Exact <bits|U|0>, reusing the cached path."""
        for q in range(self.n):
            self.tn[f"out{q}"].modify(data=self._vector(bits[q]))
        arrays = [t.data for t in self.tensors]
        t0 = time.time()
        val = self._fn(arrays)
        return complex(np.asarray(val).reshape(-1)[0]), time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qasm", required=True)
    ap.add_argument("--bits", default=None, help="output bitstring, index i = qubit i")
    ap.add_argument("--random", type=int, default=0, help="also time N random strings")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--optimize", default="greedy")
    ap.add_argument("--slice", type=int, default=24, dest="slice_bits",
                    help="slice large indices to bound memory (0 = no slicing)")
    ap.add_argument("--width", action="store_true", help="report path stats only")
    ap.add_argument("--stats", action="store_true",
                    help="search the path ONCE then time many amplitudes")
    ap.add_argument("--bits-file", default=None,
                    help="file of bitstrings, one per line (search mode)")
    ap.add_argument("--reps", type=int, default=32,
                    help="cotengra HyperOptimizer repeats")
    args = ap.parse_args()

    n, gates, unmatched = parse_qasm(args.qasm)
    print(f"[circuit] qubits={n} gates={len(gates)} unmatched={len(unmatched)} "
          f"u_matrix_from_solver={HAVE_UMATRIX}")
    if not gates:
        # d0-style files use a different gate set (h/x/cx/rz/...).  Say so
        # instead of contracting an empty network.
        print("[skip] no u/cz gates (different gate set); "
              f"first unmatched line: {unmatched[0] if unmatched else '(none)'}")
        return

    if args.stats:
        if args.slice_bits < 1:
            print("[error] --stats needs slicing (--slice >= 1) so the path "
                  "search returns a tree we can reuse")
            return
        rng = np.random.default_rng(args.seed)
        oracle = ReusableOracle(n, gates, args.slice_bits, reps=args.reps)
        strings = []
        if args.bits_file:
            strings += [l.strip() for l in open(args.bits_file) if l.strip()]
        elif args.bits:
            strings.append(args.bits)
        for k in range(args.random if strings else max(args.random, 3)):
            strings.append("".join(rng.integers(0, 2, n).astype(str)))
        print(f"[eval] {len(strings)} strings with the cached path", flush=True)
        best = (None, -1.0)
        t_all = time.time()
        for i, b in enumerate(strings):
            amp, dt = oracle.amplitude(b)
            p = abs(amp) ** 2
            if p > best[1]:
                best = (b, p)
            print(f"  [{i}] |amp|^2={p:.6e} {dt*1000:.1f}ms bits={b}", flush=True)
        tot = time.time() - t_all
        print(f"[summary] total {tot:.1f}s for {len(strings)} strings "
              f"=> {tot/max(len(strings),1)*1000:.1f}ms each")
        print(f"[peak] best |amp|^2={best[1]:.6e} bits={best[0]}")
        return

    if args.width:
        bits = args.bits or "0" * n
        amplitude(n, gates, bits, args.optimize, args.slice_bits, verbose=True)
        return

    rng = np.random.default_rng(args.seed)
    todo = []
    if args.bits:
        todo.append(("given", args.bits))
    for k in range(args.random):
        todo.append((f"rand{k}", "".join(rng.integers(0, 2, n).astype(str))))

    for tag, b in todo:
        if len(b) != n:
            print(f"[{tag}] length {len(b)} != {n} qubits -- skipped")
            continue
        amp, dt = amplitude(n, gates, b, args.optimize, args.slice_bits)
        print(f"[{tag}] |amp|^2 = {abs(amp)**2:.6e}  amp = {amp:.6e}  "
              f"{dt:.1f}s  bits={b}")


if __name__ == "__main__":
    main()
