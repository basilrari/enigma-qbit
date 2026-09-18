#!/usr/bin/env python3
"""End-to-end check of the submission protocol against the official code.

Emits through our solver's own emitter, then runs it through the challenge's
real extract_artifacts + validate_hqp_solution.  Uses d1, whose answer we
already know, so a pass proves the whole chain -- separator framing, zip
layout, result.json key names, the `status` field -- is genuinely correct
rather than merely plausible.  Regression case 3 is the bug we just fixed: a
correct bitstring with no `status` must FAIL.
"""
import io
import json
import os
import sys
import tempfile
import importlib.util

sys.path.insert(0, '/home/basilsclaw/enigma-solve')

# Load ONLY the two challenge modules we actually test.  Importing the
# qbittensor package would drag in bittensor -> pydantic -> the whole validator
# stack, none of which is involved in the output protocol.  solution_output.py
# is pure stdlib; hardening_quantum_proof.py does `from . import Serde`, so it
# gets a synthetic parent package holding a minimal Serde (the validator only
# constructs Solution/Verif -- Serde's own (de)serialisation is never reached).
import types  # noqa: E402
import importlib.util  # noqa: E402

_CDIR = '/tmp/enigma-pub/qbittensor/challenges'

_pkg = types.ModuleType('qbt')
_pkg.__path__ = [_CDIR]
_pkg.Serde = type('Serde', (), {})
sys.modules['qbt'] = _pkg


def _load_mod(modname):
    spec = importlib.util.spec_from_file_location(
        f'qbt.{modname}', os.path.join(_CDIR, f'{modname}.py'))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f'qbt.{modname}'] = mod
    spec.loader.exec_module(mod)
    return mod


_so = _load_mod('solution_output')
_hq = _load_mod('hardening_quantum_proof')

RESULT_JSON_FILENAME = _so.RESULT_JSON_FILENAME
extract_artifacts = _so.extract_artifacts
Solution = _hq.Solution
Verif = _hq.Verif
validate_hqp_solution = _hq.validate_hqp_solution

spec = importlib.util.spec_from_file_location(
    'hqp_solver', '/home/basilsclaw/enigma-solve/hqp_solver.py')
hqp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hqp)

D1 = '0001001101001111101001001110010001111010100000'
ver = Verif(peaked_state=D1)


class Cap:
    """Stand-in stdout exposing only the .buffer our emitter writes through."""
    def __init__(self, b):
        self.buffer = b

    def flush(self):
        pass

    def write(self, s):
        pass


def roundtrip(submission, label):
    buf = io.BytesIO()
    real = sys.stdout
    sys.stdout = Cap(buf)
    try:
        hqp.emit_solution_output(submission)
    finally:
        sys.stdout = real
    raw = buf.getvalue()
    d = tempfile.mkdtemp()
    ok, err = extract_artifacts(raw, d)
    print(f'[{label:12s}] official extract: ok={ok} err={err}')
    if not ok:
        return None
    path = os.path.join(d, RESULT_JSON_FILENAME)
    if not os.path.isfile(path):
        print(f'[{label:12s}] MISSING {RESULT_JSON_FILENAME}')
        return None
    return json.load(open(path))


def check(submission, label, expect):
    got = roundtrip(submission, label)
    if got is None:
        print(f'[{label:12s}] FAIL: could not extract')
        return False
    sol = Solution(status=got.get('status'), peaked_state=got.get('peaked_state'))
    ok, reason = validate_hqp_solution(sol, ver)
    verdict = 'PASS' if ok == expect else 'FAIL'
    print(f'[{label:12s}] official validator: ok={ok} expected={expect} '
          f'-> {verdict}   {reason or ""}')
    return ok == expect


results = []
results.append(check({'status': 'success', 'peaked_state': D1}, 'correct', True))
results.append(check({'status': 'success', 'peaked_state': D1[::-1]}, 'reversed', True))
results.append(check({'peaked_state': D1}, 'no-status', False))
results.append(check({'status': 'success', 'peaked_state': '0' * len(D1)}, 'wrong', False))

print()
print(f'PROTOCOL TEST: {"ALL PASS" if all(results) else "FAILURES"} ({sum(results)}/{len(results)})')
sys.exit(0 if all(results) else 1)
