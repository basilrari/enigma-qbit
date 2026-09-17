import os
import paramiko

HOST, USER, PASS = os.environ.get("SRVPRO_PASS", ""), "basilrari", "iloveesl"
VER = "/mnt/8tb_hdd2/basilrari/enigma-work/verify"
PY = "/mnt/8tb_hdd2/basilrari/enigma-work/l2venv/bin/python"

# (circuit, chi, nsamples) -- 1 BLAS thread each
JOBS = [
    ("d1_s1_4043cafb", 128, 200000),   # CONTROL: solved instance -> pipeline check
    ("d2_s1_39b370e4", 128, 200000),
    ("d2_s1_39b370e4", 256, 200000),
    ("d3_s1_2674779a", 128, 200000),
    ("d3_s1_2674779a", 256, 200000),
]

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, username=USER, password=PASS)
for cid, chi, ns in JOBS:
    tag = f"{cid}_c{chi}"
    cmd = (f"cd {VER} && export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
           f"MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 && "
           f"nohup {PY} sample_probe.py {cid} {chi} {ns} 4000 "
           f"> {VER}/sample_{tag}.log 2>&1 & echo STARTED $!")
    i, o, e = c.exec_command(cmd)
    print(tag, o.read().decode().strip(), e.read().decode().strip())
c.close()
