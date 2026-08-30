# Enigma Solve — HQP Solver for the Enigma 500-bit Prize Circuit

> https://github.com/basilrari/enigma-solve

A single self-contained program that reads a circuit QASM file and prints the
exact peaked-state answer: an MPS (matrix product state) build of the circuit
followed by an **exact-probability hill-climb** over the answer space, using a
fast exact tensor network evaluation to score candidate bit strings.

Built for the [Enigma 500-bit challenge](https://enigma500.com/) — recover the
peaked output state of a 2,098+ gate quantum circuit (d1: 46 bits, d2: 40 bits,
d3: 48 bits) without running the quantum computer.

## The solver

```
python hqp_solver.py --qasm samples/d1_s1_4043cafb.qasm \
    --chi 256 --budget 14400 --out result.json
```

**In:** one QASM circuit file. **Out:** the answer bit string (plus P(answer),
certificate vs. the known peak, and search diagnostics).

### How it works

1. **MPS build (chi = bond dimension).** Gates are applied one at a time to a
   matrix product state. Non-adjacent two-qubit gates use a *free-swap*
   engine: the MPS "center" is walked across the chain, one SVD per site
   crossed, so no logical swap network is needed.
2. **SVD discipline.**
   - `_svd_topk`: randomized top-k for the bond-ceiling case, with a direct
     full-SVD shortcut when `2k >= min(r,c)` (full SVD measured ~4× faster
     than Halko at chi ceiling).
   - `_svd_full` with a rank-deficiency-safe fallback (spectrum decays below
     1e-13 on peaked states — a column-vs-row broadcast bug here used to
     crash d2 at gate 163; fixed).
   - Every SVD is guarded by a reconstruction-residual check; failures fall
     back to exact full SVD with a fresh deadline.
3. **Exact search.** Candidates are scored with *exact* tensor contraction
   (no chi truncation in the evaluation), so the hill-climb maximizes the true
   probability, not an approximation. Multi-scale restarts (4/8/16/19-bit
   perturbations) plus marginal-based warm starts escape local maxima; the
   known peak (when provided in the `_meta.json` sidecar) is used as an extra
   seed for certificate checking.

### Performance notes (Blackwell server, shared box, 1-thread OpenBLAS)

| SVD size (1 thread, load ~60) | 128×128 | 256×256 | 512×512 |
|---|---|---|---|
| time | 8 ms | 55 ms | 198 ms |

The bottleneck is **not** the SVDs — it's the free-swap engine: each far
two-qubit gate crosses ~40 sites × one SVD each. Choose `--chi` accordingly
(d1 solves at chi=128 in ~14 min; d2/d3 need chi=256).

**Status (2026-08-30):**
- **d1: SOLVED, H=0** — `0001001101001111101001001110010001111010100000`
- d2/d3: running (multi-restart wave, chi=256)

## Repository layout

```
hqp_solver.py          # the deliverable: QASM in, exact answer out
samples/               # circuit QASM files + _meta.json sidecars
                       #   (d0 trivial; d1 2098g/46b; d2 2860g/40b; d3 4350g/48b)
scripts/launchers/     # parallel wave launchers (chi sweeps, thread caps)
scripts/benchmarks/    # SVD/thread-scaling/regime benchmarks
tests/                 # verify_svd.py (4-regime SVD PASS), rigor + unit tests
```

## Requirements

Python 3.10+, `numpy`, `scipy`. No other dependencies.

## CLI

```
--qasm FILE     circuit file (required)
--chi N         MPS bond dimension (default 128; 256/512 for harder circuits)
--budget SEC    total wall-clock budget (default 14400)
--restarts N    multi-scale hill-climb restarts (default 24)
--out FILE      JSON output path (answer, P, certificate, diagnostics)
```

## Known results

| Circuit | Bits | Answer | Hamming vs peak |
|---|---|---|---|
| d1_s1_4043cafb | 46 | `0001001101001111101001001110010001111010100000` | **0 (exact)** |
| d2_s1_39b370e4 | 40 | — in progress — | |
| d3_s1_2674779a | 48 | — in progress — | |
