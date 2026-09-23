# d2 split probe (2026-09-23)

Question: is there a new structural crack for `d2_s1`, or was the old "no mirror" conclusion too strict?

Probe: `python scripts/hqp_invent.py`. No full-circuit simulation. Sample: `samples/d2_s1_39b370e4.qasm`.

## Measured

- 40 qubits, 2860 gates, 940 CZ, 161 unique matrix-inverse pairs.
- Best split is a band, not a point. Crossing pairs: 149 at 45% depth (gate 1287), 149 at 47%, 148 at 50%. Other cuts are mostly 0. Peak / mean = 4.19.
- At the 45% cut those 149 pairs are smeared: offset min / median / max = 117 / 617 / 1145 gates. None sit within 5 gates. Their early members span 27–45% depth.
- χ=384 still misses bits 6, 7, 25, 28. All four have inverse-pair vote margin 1.00. Partners are 2, 16, 22, 3 — not each other. They are not the lowest-margin wires (those are 23, 0, 26) and not a CZ-degree cluster (ranks 30, 9, 1, 26).
- Sliding windows that are only a wire-swap: 5 of 42 at width 8, none of them identity. Width 16 and 24 had no window small enough to test.

## Answer

A strict gate-i vs gate-(n-i) mirror test is the wrong test. The two halves really are inverses of each other, just smeared by hundreds of gates. That is the sandwich the published middle-MPO method needs. It is not a gadget you can delete: the early members are interleaved across 27–45% of the circuit.

The four wrong bits are not marked in the wiring. Do not hunt them there.

## Recommendation

Do not build a new simulator. Score the existing engine (runner bugs already fixed) before adding knobs. If it peels the disguise before the 27–52% band is inside the simulation, the one new switch is: absorb through that band first, then unswap. `equal=True` exists in the engine and has not been turned on.

Solver behaviour is unchanged by this note.
