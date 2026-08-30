"""Rigor: MPS oracle vs exact statevector. The MPS free-swap engine PERMUTES
qubits: qo[k] = original qubit at MPS position k. The oracle's flip index k is
an MPS POSITION, so flipping position k means flipping original qubit qo[k].
A bitstring s is ORIGINAL order: s[q] = bit of qubit q. To express the state
with MPS position k having bit x, set s[qo[k]] = x.
qiskit Statevector: qubit q is bit 2^q (q0 = LSB), idx = sum s[q]*2^q."""
import numpy as np
src = open('hqp_solver.py').read()
ns = {}
exec(src[:src.index('def main(')], ns)
MPS = ns['MPS']; build_mps = ns['build_mps']
from qiskit import QuantumCircuit
from qiskit.circuit.library import UGate
from qiskit.quantum_info import Statevector

rng = np.random.default_rng(12345)
worst = 0.0
def sidx(s, n): return sum(int(s[q]) * (2 ** q) for q in range(n))

for trial in range(8):
    n = 10
    qc = QuantumCircuit(n)
    for _ in range(12):
        qc.append(UGate(*rng.uniform(0, 2*np.pi, 3)), [int(rng.integers(0, n))])
    for _ in range(8):
        a, b = rng.choice(n, 2, replace=False)
        qc.cz(int(a), int(b))
    sv = Statevector(qc); probs = np.abs(sv) ** 2
    T, order, qo = build_mps(qc, 16, verbose=False)
    m = MPS(T, qo, None, f"t{trial}")
    qo = list(qo)

    def mk_state(seed):
        # seed: bit per MPS position (list len n) -> original-order string
        s = ['0']*n
        for k in range(n): s[qo[k]] = str(seed[k])
        return ''.join(s)

    # exact P for random MPS-position seeds
    for r in range(8):
        s = mk_state(list(rng.integers(0, 2, n)))
        exact = float(probs[sidx(s, n)])
        got = m.P(s)
        worst = max(worst, abs(exact-got))
        assert abs(exact-got) < 1e-8, f"P t{trial}: {exact} vs {got}"

    # single-flip-all (oracle index = MPS position k -> flips qubit qo[k])
    seed0 = list(rng.integers(0, 2, n))
    s0 = mk_state(seed0)
    fl = m.P_single_flip_all(s0)
    for k in range(n):
        fs = list(seed0); fs[k] = 1 - fs[k]
        t = mk_state(fs)
        ex = float(probs[sidx(t, n)])
        worst = max(worst, abs(ex - fl[k]))
        assert abs(ex - fl[k]) < 1e-8, f"flip t{trial} k{k}: {ex} vs {fl[k]}"

    # double-flip best over all MPS-position pairs
    best_exp = -1.0
    for i in range(n):
        for j in range(i+1, n):
            fs = list(seed0); fs[i] = 1-fs[i]; fs[j] = 1-fs[j]
            best_exp = max(best_exp, float(probs[sidx(mk_state(fs), n)]))
    best_got, _ = m.P_double_flip_all(s0)
    worst = max(worst, abs(best_exp - best_got))
    assert abs(best_exp - best_got) < 1e-8, f"dbl t{trial}: {best_exp} vs {best_got}"

    # marginal P(1) per MPS position k -> original qubit qo[k]
    p1 = m.marginal_p1()
    for k in range(n):
        q = qo[k]
        exact1 = float(sum(probs[x] for x in range(2**n) if (x >> q) & 1))
        worst = max(worst, abs(exact1 - p1[k]))
        assert abs(exact1 - p1[k]) < 1e-8, f"marg t{trial} k{k}: {exact1} vs {p1[k]}"

    print(f"trial {trial}: OK  qo={qo}  worst={worst:.2e}")

print(f"\nFINAL worst abs error = {worst:.3e}")
print("ALL ORACLES CORRECT" if worst < 1e-8 else "FAILURES")
