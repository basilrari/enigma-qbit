#!/usr/bin/env python3
import sys, time
import numpy as np
sys.path.insert(0, "/home/basilsclaw/enigma-solve")
import hqp_solver as H

path = "/home/basilsclaw/enigma-solve/samples/d1_s1_4043cafb.qasm"
chi = 128
circ = H.load_circuit(path)
n = circ.num_qubits
tensors = [np.zeros((1,2,1), dtype=complex) for _ in range(n)]
for k in range(n): tensors[k][0,0,0]=1.0
ops, order = H.reorder_qubits(circ)
pos_of = list(range(n)); q_at = list(range(n)); center = n-1

timings = {}
t0 = time.time()
ng = 0
for idx,(name,qs,params) in enumerate(ops):
    t_g = time.time()
    if name == 'u':
        t,p,lam = params[0]/2, params[1], params[2]
        c,s=np.cos(t),np.sin(t)
        U=np.array([[c,-np.exp(1j*lam)*s],[np.exp(1j*p)*s,np.exp(1j*(p+lam))*c]],dtype=complex)
        i=pos_of[qs[0]]
        tensors[i]=np.einsum('ab,lbr->lar',U,tensors[i])
        timings['u'] = timings.get('u',0)+(time.time()-t_g)
    else:
        a,b=qs
        mat=np.diag([1,1,1,-1]).astype(complex)
        i,j=pos_of[a],pos_of[b]
        if i>j: i,j=j,i; a,b=b,a
        while pos_of[b] > pos_of[a]+1:
            p2=pos_of[b]
            while center>p2-1:
                H._move_center_left(tensors,center); center-=1
                timings['moveL']=timings.get('moveL',0)+0
            while center<p2-1:
                H._move_center_right(tensors,center); center+=1
            ts=time.time()
            center=H._apply_swap(tensors,p2-1,chi)
            timings['swap']=timings.get('swap',0)+(time.time()-ts)
            q_at[p2-1],q_at[p2]=q_at[p2],q_at[p2-1]
            pos_of[q_at[p2-1]],pos_of[q_at[p2]]=p2-1,p2
        i,j=pos_of[a],pos_of[b]
        while center>i: H._move_center_left(tensors,center); center-=1
        while center<i: H._move_center_right(tensors,center); center+=1
        ts=time.time()
        center=H._apply_2site(tensors,i,mat,chi)
        timings['cz2s']=timings.get('cz2s',0)+(time.time()-ts)
    ng+=1
    if ng in (50,100,150,200):
        print(f"gate {ng}: {time.time()-t0:7.1f}s  | " + "  ".join(f"{k}={v:6.1f}s" for k,v in sorted(timings.items())), flush=True)
    if time.time()-t0 > 240:
        print("hitting 240s wall at gate", ng, flush=True)
        break
print("TOTAL:", flush=True)
for k,v in sorted(timings.items(), key=lambda x:-x[1]):
    print(f"  {k:6s} {v:8.1f}s")
print(f"  wall   {time.time()-t0:8.1f}s  gates={ng}")
# one isolated SVD timing at full bond
m = np.random.randn(2*chi, 2*chi)+1j*np.random.randn(2*chi, 2*chi)
ts=time.time(); np.linalg.svd(m, full_matrices=False); print(f"single {2*chi}x{2*chi} SVD: {time.time()-ts:.3f}s")
