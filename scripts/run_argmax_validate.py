import os
import json

import paramiko

HOST, USER, PASS = os.environ.get("SRVPRO_PASS", ""), "basilrari", "iloveesl"
VER = "/mnt/8tb_hdd2/basilrari/enigma-work/verify"
PY = "/mnt/8tb_hdd2/basilrari/enigma-work/l2venv/bin/python"

PEAKS = {
    "d1_s1_4043cafb": "0001001101001111101001001110010001111010100000",
    "d1_s2_adeddcf3": "101110000101001110100111011000011101110111000111",
    "d2_s1_39b370e4": "1110101100010111001000000011101111001000",
    "d2_s2_1efabaf4": "01110101101101000111110010101000011010110100",
    "d3_s1_2674779a": "011100110011000101111001110011100000101101100100",
    "d3_s2_c09ba537": "110101001000011010110111110011110010111010100100",
}

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, username=USER, password=PASS)
sf = c.open_sftp()
sf.put("/home/basilsclaw/enigma-solve/scripts/argmax_probe.py",
       f"{VER}/argmax_probe.py")
with sf.open(f"{VER}/_peaks.json", "w") as f:
    f.write(json.dumps(PEAKS, indent=1))
sf.close()
cmd = (f"cd {VER} && export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
       f"MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 && timeout 600 "
       f"{PY} argmax_probe.py --validate")
i, o, e = c.exec_command(cmd, timeout=650)
o.channel.recv_exit_status()
print(o.read().decode())
err = e.read().decode()
if err.strip():
    print("STDERR:", err[-2000:])
c.close()
