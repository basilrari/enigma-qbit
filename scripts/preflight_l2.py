#!/usr/bin/env python3
"""Pre-flight each QASM: can it load, unitarize, and how many layers?"""
import os
import paramiko

PW = open(os.path.expanduser("~/.srvpro_env")).read().split("'")[1]
W = "/mnt/8tb_hdd2/basilrari/enigma-work"
V = f"{W}/verify"
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("140.123.105.18", username="basilrari", password=PW)
s = c.open_sftp()
s.put("/home/basilsclaw/enigma-solve/scripts/l2_absorb.py", f"{W}/l2_absorb.py")
s.close()

code = f'''
import sys
sys.path.insert(0, "{W}")
sys.path.insert(0, "{W}/l2win")
import l2_absorb as LA, utils as UT
for inst in ["d0_s0_trivial", "d1_s1_4043cafb", "d2_s1_39b370e4", "d3_s1_2674779a"]:
    try:
        qc = LA.load_qasm("{V}/" + inst + ".qasm")
        u = LA.unitarize(qc)
        n = len(list(UT.iter_layers(u)))
        sizes = sorted({{len(lay.data) for lay in UT.iter_layers(u)}}, reverse=True)[:5]
        print(f"OK {{inst}}: {{qc.num_qubits}}q {{len(qc.data)}} gates -> "
              f"{{u.count_ops().get('unitary',0)}} unitaries, {{n}} layers, biggest layers {{sizes}}")
    except Exception as e:
        print(f"FAIL {{inst}}: {{type(e).__name__}}: {{repr(e)[:150]}}")
'''
with open("/tmp/preflight_l2.py", "w") as fh:
    fh.write(code)
s = c.open_sftp(); s.put("/tmp/preflight_l2.py", f"{W}/preflight_l2.py"); s.close()
_i, o, e = c.exec_command(f"cd {V} && HQP_QUIET=1 {W}/l2venv/bin/python {W}/preflight_l2.py 2>&1 | "
                          f"grep -vE 'FutureWarning|tensor_network_ag_compress'", timeout=420)
o.channel.recv_exit_status()
print(o.read().decode() + e.read().decode())
c.close()
