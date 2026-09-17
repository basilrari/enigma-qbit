import paramiko, sys

HOST, USER, PASS = "140.123.105.18", "basilrari", "iloveesl"
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, username=USER, password=PASS, timeout=30)

cmds = [
 ("ls enigma-work/verify | head -30", "verify dir"),
 ("ls enigma-work/ 2>/dev/null | head", "work dir"),
 ("for p in cotengra opt_einsum quimb numpy; do "
  "/mnt/8tb_hdd2/basilrari/enigma-work/l2venv/bin/python -c "
  "\"import $p,sys;print('$p', getattr($p,'__version__','?'))\" 2>&1 | tail -1; done", "l2venv packages"),
 ("nproc; uptime", "load"),
 ("nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader", "gpu"),
]
for cmd, label in cmds:
    _, o, e = c.exec_command(cmd, timeout=120)
    out = o.read().decode(errors="replace").strip()
    err = e.read().decode(errors="replace").strip()
    print(f"===== {label} =====")
    print(out if out else "(no stdout)")
    if err: print("[stderr]", err[:400])
c.close()
