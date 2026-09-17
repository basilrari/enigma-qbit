import os
import paramiko, sys

HOST, USER, PASS = os.environ.get("SRVPRO_PASS", ""), "basilrari", "iloveesl"
VER = "/mnt/8tb_hdd2/basilrari/enigma-work/verify"
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, username=USER, password=PASS)

def sh(cmd):
    i, o, e = c.exec_command(cmd)
    o.channel.recv_exit_status()
    return o.read().decode() + e.read().decode()

print("### live jobs:", sh("ps aux | grep -c [s]ample_probe").strip())
print(sh("ps aux | grep [s]ample_probe | awk '{print $12, $13, $14}'"))

print("\n### TN WIDTH PROBE ###")
print(sh(f"grep -E '===|width|flops|VERDICT|FAILED' {VER}/tnwidth.log | tail -20"))

print("\n### SAMPLING RUNS ###")
for tag in ["d1_s1_4043cafb_c128", "d2_s1_39b370e4_c128", "d2_s1_39b370e4_c256",
            "d3_s1_2674779a_c128", "d3_s1_2674779a_c256"]:
    out = sh(f"tail -3 {VER}/sample_{tag}.log 2>/dev/null")
    key = sh(f"grep -E 'TRUE ANSWER|sampled |rank of|MPS P\\(|build done' "
             f"{VER}/sample_{tag}.log 2>/dev/null | tail -4")
    print(f"\n--- {tag}")
    print("   " + key.strip().replace("\n", "\n   "))
c.close()
