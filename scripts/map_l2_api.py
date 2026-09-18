#!/usr/bin/env python3
"""Map the published L2 pipeline's API so an orchestrator can be rebuilt.

tebd_mps.py is missing (only its .pyc survives), so recover its surface by
marshalling the code object -- no torch needed for that.
"""
import os

import paramiko

PW = open(os.path.expanduser("~/.srvpro_env")).read().split("'")[1]
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("140.123.105.18", username="basilrari", password=PW)
L = "/mnt/8tb_hdd2/basilrari/enigma-work/l2win"


def run(x, t=240):
    i, o, e = c.exec_command(x, timeout=t)
    o.channel.recv_exit_status()
    return o.read().decode()


for f in ["circuit_mpo.py", "utils.py", "extract.py", "unswap_stallfix.py"]:
    print(f"=== {f} top-level defs ===")
    print(run(f"grep -nE '^def |^class |^[A-Z_]+ *=' {L}/{f} | head -25"))

print("=== imports (which modules each needs) ===")
print(run(f"cd {L} && grep -hE '^ *(import|from) ' *.py | sort -u | head -30"))

print("=== tebd_mps.py surface, recovered from its bytecode ===")
probe = r'''
import marshal, importlib.util, sys
p = sys.argv[1]
d = open(p, "rb").read()
co = marshal.loads(d[16:])
def walk(c, ind=0):
    for k in c.co_consts:
        if hasattr(k, "co_name"):
            doc = (k.co_consts[0] or "") if k.co_consts else ""
            if isinstance(doc, str):
                doc = doc.strip().splitlines()[0][:70] if doc.strip() else ""
            print(" " * ind + k.co_name + "(" + ",".join(k.co_varnames[:k.co_argcount]) + ")" +
                  ("  # " + doc if doc else ""))
            walk(k, ind + 2)
print("MODULE consts:", [k for k in co.co_consts if isinstance(k, str)][:12])
walk(co)
'''
run(f"cat > /tmp/co_probe.py <<'EOF'\n{probe}\nEOF")
print(run(f"/usr/bin/python3 /tmp/co_probe.py {L}/__pycache__/tebd_mps.cpython-312.pyc"))

print("=== is torch present in gpuenv / l2venv? ===")
print(run("~/enigma-work/gpuenv/bin/python -c 'import torch;print(\"gpuenv torch\",torch.__version__,torch.cuda.is_available())' 2>&1 | tail -2"))
print(run("~/enigma-work/l2venv/bin/python -c 'import quimb;print(\"l2venv quimb\",quimb.__version__)' 2>&1 | tail -1")) 
print("=== nvidia-smi ===")
print(run("nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu "
          "--format=csv,noheader 2>&1 | head -4"))
c.close()
