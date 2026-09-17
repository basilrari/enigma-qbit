import paramiko

HOST, USER, PASS = "140.123.105.18", "basilrari", "iloveesl"
VER = "/mnt/8tb_hdd2/basilrari/enigma-work/verify"
PY = "/mnt/8tb_hdd2/basilrari/enigma-work/l2venv/bin/python"

# Test: is the randomized top-k SVD suppressing the peak at large chi?
# Compare the SAME chi with the fast path on (already have chi=128) vs off.
JOBS = [
    # (circuit, chi, nsamples, env, tag_suffix)
    ("d2_s1_39b370e4", 128, 20000, "HQP_EXACT_SVD=1", "exactsvd"),
    ("d2_s1_39b370e4", 256, 20000, "HQP_EXACT_SVD=1", "exactsvd"),
    ("d3_s1_2674779a", 128, 20000, "HQP_EXACT_SVD=1", "exactsvd"),
    # and the same thing with MORE power iterations in the fast path
    ("d2_s1_39b370e4", 128, 20000, "HQP_POWER_STEPS=32", "p32"),
]

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, username=USER, password=PASS)
sh = ["#!/bin/bash", f"cd {VER}",
      "export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1"]
for cid, chi, ns, env, suf in JOBS:
    tag = f"{cid}_c{chi}{'_' + suf if suf else ''}"
    sh.append(f"setsid nohup env {env} {PY} sample_probe.py {cid} {chi} {ns} 4000 "
              f"\"{'_' + suf if suf else ''}\" "
              f"> {VER}/sample_{tag}.log 2>&1 < /dev/null &")
sh.append("sleep 3; ps aux | grep [s]ample_probe | wc -l")
sf = c.open_sftp()
sf.put("/home/basilsclaw/enigma-solve/scripts/sample_probe.py", f"{VER}/sample_probe.py")
with sf.open(f"{VER}/_launch_svdtest.sh", "w") as f:
    f.write("\n".join(sh) + "\n")
sf.close()
i, o, e = c.exec_command(f"bash {VER}/_launch_svdtest.sh", timeout=60)
o.channel.recv_exit_status()
print("live sample_probe procs now:", o.read().decode().strip())
c.close()
