#!/usr/bin/env python3
"""One-shot structural invention probe. No simulation. Prints a verdict."""
import collections
import math
import re
from pathlib import Path

import numpy as np

UG = re.compile(r"^u\(([^)]*)\)\s+q\[(\d+)\];")
CZ = re.compile(r"^cz\s+q\[(\d+)\],q\[(\d+)\];")


def num(tok):
    tok = tok.strip()
    try:
        return float(tok)
    except ValueError:
        return float(eval(tok.replace("pi", repr(math.pi)), {"__builtins__": {}}, {}))


def u_mat(t, p, l):
    c, s = math.cos(t / 2.0), math.sin(t / 2.0)
    return np.array(
        [[c, -np.exp(1j * l) * s], [np.exp(1j * p) * s, np.exp(1j * (p + l)) * c]],
        dtype=complex,
    )


def canon(M):
    A = np.asarray(M, dtype=complex)
    k = np.unravel_index(np.argmax(np.abs(A)), A.shape)
    if abs(A[k]) < 1e-12:
        return ("zero",)
    B = A / A[k]
    return tuple(round(float(v), 5) for v in np.concatenate([B.real.ravel(), B.imag.ravel()]))


def parse(path):
    gates = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        m = UG.match(line)
        if m:
            ang = [num(v) for v in m.group(1).split(",")]
            gates.append(("u", int(m.group(2)), ang))
            continue
        m = CZ.match(line)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            gates.append(("cz", (a, b) if a < b else (b, a), None))
    n = 0
    for g in gates:
        if g[0] == "u":
            n = max(n, g[1] + 1)
        else:
            n = max(n, g[1][0] + 1, g[1][1] + 1)
    return gates, n


def inverse_pairs(gates):
    us = [(i, g[1], g[2]) for i, g in enumerate(gates) if g[0] == "u"]
    fwd = collections.defaultdict(list)
    mats = {}
    for i, q, ang in us:
        M = u_mat(*ang)
        mats[i] = M
        fwd[canon(M)].append(i)
    pairs = []
    for i, q, ang in us:
        key = canon(mats[i].conj().T)
        for j in fwd.get(key, ())[:8]:
            if j != i:
                pairs.append((i, j, q, gates[j][1]))
    return pairs


def split_profile(gates, pairs, n_bins=40):
    N = len(gates)
    # each unordered pair {i,j} counted once
    seen = set()
    uniq = []
    for i, j, qa, qb in pairs:
        a, b = (i, j) if i < j else (j, i)
        if (a, b) in seen:
            continue
        seen.add((a, b))
        uniq.append((a, b))
    crosses = []
    for s in range(1, n_bins):
        cut = int(N * s / n_bins)
        cross = sum(1 for a, b in uniq if a < cut <= b)
        crosses.append((cut, cross))
    return uniq, crosses


def vote_margins(gates, pairs, n):
    V = np.zeros((n, n))
    for i, j, qa, qb in pairs:
        if qa != qb:
            V[qa, qb] += 1
    margins = []
    for a in range(n):
        row = V[a] + V[:, a]
        tot = row.sum()
        mx = row.max() if tot else 0
        partner = int(row.argmax()) if tot else -1
        margins.append((a, mx / tot if tot else 0.0, tot, partner))
    return margins


def last_u(gates, n):
    last = [None] * n
    deg = [0] * n
    for g in gates:
        if g[0] == "u":
            last[g[1]] = g[2]
        else:
            deg[g[1][0]] += 1
            deg[g[1][1]] += 1
    return last, deg


def phased_perm_scan(gates, max_q=3, widths=(8, 16, 24)):
    """Slide windows; if support <= max_q, test whether the unitary is a phased permutation."""
    # Only feasible exactly for support<=3 if we simulate the small unitary.
    # We do a statevector on the window's qubits (2^q <= 8).
    hits = {w: 0 for w in widths}
    tested = {w: 0 for w in widths}
    id_hits = {w: 0 for w in widths}
    N = len(gates)
    for w in widths:
        for start in range(0, N - w, 2):
            window = gates[start : start + w]
            qs = []
            for g in window:
                if g[0] == "u":
                    qs.append(g[1])
                else:
                    qs.extend(g[1])
            supp = sorted(set(qs))
            if not supp or len(supp) > max_q:
                continue
            tested[w] += 1
            q = len(supp)
            idx = {qubit: k for k, qubit in enumerate(supp)}
            dim = 1 << q
            # apply window to each basis state; record images
            images = []
            mono = True
            ident = True
            for basis in range(dim):
                st = np.zeros(dim, dtype=complex)
                st[basis] = 1
                for g in window:
                    if g[0] == "u":
                        M = u_mat(*g[2])
                        b = idx[g[1]]
                        st = apply_1q(st, M, b, q)
                    else:
                        a, b = idx[g[1][0]], idx[g[1][1]]
                        st = apply_cz(st, a, b, q)
                nz = np.flatnonzero(np.abs(st) > 0.15)
                if len(nz) != 1 or abs(st[nz[0]]) < 0.9:
                    mono = False
                    break
                images.append(int(nz[0]))
                if int(nz[0]) != basis or abs(st[nz[0]] - 1) > 0.05:
                    ident = False
            if mono and len(set(images)) == dim:
                hits[w] += 1
                if ident:
                    id_hits[w] += 1
    return tested, hits, id_hits


def apply_1q(st, M, b, q):
    dim = st.shape[0]
    out = np.zeros_like(st)
    mask = 1 << b
    for i in range(dim):
        bit = (i & mask) >> b
        base = i & ~mask
        for nb in (0, 1):
            j = base | (nb << b)
            out[j] += M[nb, bit] * st[i]
    return out


def apply_cz(st, a, b, q):
    out = st.copy()
    ma, mb = 1 << a, 1 << b
    for i in range(st.shape[0]):
        if (i & ma) and (i & mb):
            out[i] *= -1
    return out


def hamming(a, b):
    return [i for i, (x, y) in enumerate(zip(a, b)) if x != y]


def main():
    truth = "1110101100010111001000000011101111001000"
    # chi=384 reported answer (H=4). Confirmed against the session log.
    near = "1110100000010111001000000111001111001000"
    bad = hamming(truth, near)
    print("d2 error bits", bad, "H", len(bad))

    path = Path(__file__).resolve().parents[1] / "samples" / "d2_s1_39b370e4.qasm"
    gates, n = parse(path)
    print(f"gates {len(gates)} qubits {n} cz {sum(g[0]=='cz' for g in gates)}")
    pairs = inverse_pairs(gates)
    uniq, crosses = split_profile(gates, pairs, 40)
    print(f"unique inverse pairs {len(uniq)}")
    crosses_sorted = sorted(crosses, key=lambda x: -x[1])
    print("top split cuts (gate_index, cross_pairs):")
    for cut, c in crosses_sorted[:6]:
        print(f"  {cut:5d}  {c:4d}  ({100*cut/len(gates):.0f}% depth)")
    vals = [c for _, c in crosses]
    print(f"cross min/median/max {min(vals)}/{sorted(vals)[len(vals)//2]}/{max(vals)}")
    # sharpness: max vs mean
    print(f"max/mean {max(vals)/ (sum(vals)/len(vals)):.2f}")
    best_cut = crosses_sorted[0][0]
    crossing = [(a, b) for a, b in uniq if a < best_cut <= b]
    offs = sorted(b - a for a, b in crossing)
    med = offs[len(offs) // 2]
    print(
        f"best-cut crossing {len(crossing)} offset min/median/max "
        f"{offs[0]}/{med}/{offs[-1]} within5 {sum(d <= 5 for d in offs)}"
    )
    lefts = [a for a, _b in crossing]
    print(
        f"left-member depth span "
        f"{100 * min(lefts) / len(gates):.0f}-{100 * max(lefts) / len(gates):.0f}%"
    )

    margins = vote_margins(gates, pairs, n)
    last, deg = last_u(gates, n)
    print("\nerror-bit features (margin, votes, partner, cz_deg, last_theta):")
    for a, frac, tot, partner in margins:
        if a in bad:
            th = last[a][0] if last[a] else None
            print(f"  q{a}: margin={frac:.2f} votes={tot:.0f} partner={partner} cz={deg[a]} last_th={th}")
    # rank qubits by LOW margin (suspect) and by HIGH cz
    by_margin = sorted(margins, key=lambda x: (x[1], -x[2]))
    print("lowest-margin qubits", [(a, round(f, 2), int(t)) for a, f, t, _ in by_margin[:8]])
    print("error bits' margin ranks", [i for i, (a, *_) in enumerate(by_margin) if a in bad])
    by_cz = sorted(range(n), key=lambda a: -deg[a])
    print("highest-cz qubits", [(a, deg[a]) for a in by_cz[:8]])
    print("error bits' cz ranks", [by_cz.index(a) for a in bad])

    # are error bits partners of each other?
    partner = {a: p for a, f, t, p in margins}
    print("error-bit partners", {a: partner[a] for a in bad})

    print("\nphased-permutation window scan (support<=3)...")
    tested, hits, ids = phased_perm_scan(gates)
    for w in tested:
        print(f"  width {w}: tested {tested[w]}  phased-perm {hits[w]}  of which identity {ids[w]}")


if __name__ == "__main__":
    main()
