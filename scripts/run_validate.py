import paramiko, sys, os

HOST, USER, PASS = "140.123.105.18", "basilrari", "iloveesl"
VER = "/mnt/8tb_hdd2/basilrari/enigma-work/verify"
PY = "/mnt/8tb_hdd2/basilrari/enigma-work/l2venv/bin/python"
LOCAL = "/home/basilsclaw/enigma-solve/scripts"

files = ["sample_probe.py", "validate_sampler.py"]
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, username=USER, password=PASS)
sf = c.open_sftp()
for f in files:
    sf.put(f"{LOCAL}/{f}", f"{VER}/{f}")
    print("uploaded", f)
sf.close()

cmd = (f"cd {VER} && export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "
       f"NUMEXPR_NUM_THREADS=1 && {PY} validate_sampler.py 2>&1")
i, o, e = c.exec_command(cmd)
out = o.read().decode()
err = e.read().decode()
print(out)
if err.strip():
    print("STDERR:", err[-2000:])
c.close()
