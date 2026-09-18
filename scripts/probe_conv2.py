import sys

import numpy as np
import quimb.tensor as qtn

sys.path.insert(0, "/mnt/8tb_hdd2/basilrari/enigma-work/l2win")
sys.path.insert(0, "/mnt/8tb_hdd2/basilrari/enigma-work")
import extract as EX  # noqa: E402

n = 6
qc = qtn.Circuit(n)  # |0...0>
tn = qc.psi
print("Circuit.psi type       :", type(tn).__name__)

mps = qtn.MatrixProductState.from_TN(tn, site_ind_id="k{}")
print("from_TN type           :", type(mps).__name__)
for m in ("copy", "site_ind", "normalize", "isel", "bond_size", "norm", "to_dense"):
    print(f"   has {m:10s}: {hasattr(mps, m)}")

try:
    print("amp2 from_TN           :", EX.amp2(mps, "0" * n))
except Exception as e:
    print("amp2 from_TN FAILS     :", type(e).__name__, str(e)[:90])

mps2 = mps.apply_to_arrays(lambda x: np.asarray(x))
print("apply_to_arrays type   :", type(mps2).__name__)
for m in ("copy", "site_ind", "normalize"):
    print(f"   has {m:10s}: {hasattr(mps2, m)}")
try:
    print("amp2 after a2a         :", EX.amp2(mps2, "0" * n))
except Exception as e:
    print("amp2 after a2a FAILS   :", type(e).__name__, str(e)[:90])

# the alternative: move arrays to host in place, keep the MPS object
t3 = mps.copy()
for t in t3.tensors:
    t.modify(data=np.asarray(t.data))
print("in-place modify type   :", type(t3).__name__,
      "| copy:", hasattr(t3, "copy"), "| site_ind:", hasattr(t3, "site_ind"))
try:
    print("amp2 in-place          :", EX.amp2(t3, "0" * n))
except Exception as e:
    print("amp2 in-place FAILS    :", type(e).__name__, str(e)[:90])
