import paramiko

HOST, USER, PASS = "140.123.105.18", "basilrari", "iloveesl"
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, username=USER, password=PASS, timeout=30)

cmds = [
 ("ps -eo user,pid,etime,pcpu,args --sort=-pcpu | grep -iE 'hqp|solver|python' | grep -v grep | head -20", "RUNNING PROCS"),
 ("nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader", "GPU procs"),
 ("echo '--- wave6 ---'; cat /mnt/8tb_hdd2/basilrari/enigma-work/verify/launch_wave6.sh", "wave6 launcher"),
 ("echo '--- wave5 json ---'; for f in /tmp/d1_c256.json /tmp/d2_c256.json /tmp/d3_c256.json; do echo \"== $f\"; ls -la $f 2>/dev/null || echo MISSING; done", "wave5 artifacts"),
 ("echo '--- wave logs ---'; ls -la /tmp/*.log 2>/dev/null | tail -20", "logs"),
 ("echo '--- qasm_peak_correlate head ---'; head -50 /mnt/8tb_hdd2/basilrari/enigma-work/qasm_peak_correlate.py", "prior forensic script"),
]
for cmd, label in cmds:
    _, o, e = c.exec_command(cmd, timeout=180)
    print(f"===== {label} =====")
    print(o.read().decode(errors="replace").strip() or "(none)")
c.close()
