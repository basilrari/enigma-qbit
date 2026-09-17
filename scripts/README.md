# scripts/

Helper scripts for driving the solver on the remote GPU/CPU box.

## Credentials

These scripts read the SSH password from the environment. **Never hardcode it.**

```bash
export SRVPRO_PASS='<ssh password for basilrari@140.123.105.18>'
python3 scripts/ladder_512.py
```

They fail loudly rather than silently connecting as the wrong user if it is
unset.

## The interesting ones

- `argmax_probe.py` — **exact global argmax of an MPS by branch-and-bound.**
  Validated against brute-force enumeration. This is the search method that
  replaced heuristic hill-climbing.
- `peak_read.py` / `load_imbalance.py` — tests whether the answer is encoded in
  the circuit's gate structure (it is not; both hypotheses died with data).
- `profile_build.py` — where the MPS build actually spends its time.
- `analyze_samples.py` — summarises `sample_*.json` across bond dimensions.
- `swap_budget.py`, `gadget_hunt.py`, `angle_dist.py`, `tn_width*.py` —
  falsification probes for reordering, planted Clifford gadgets, and exact
  contraction width.
- `launch_*.py`, `ladder_*.py`, `sync_and_launch.py` — remote launchers.

## History of this repo's exposure

The SSH password was once hardcoded in these scripts and pushed to this (public)
repo in commit `d37f6f5`. The working tree has been scrubbed, but the value
remains in git history — rotate the credential if it is still live.
