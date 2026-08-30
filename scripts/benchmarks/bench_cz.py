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

# How many swaps does each cz need?
swap_counts = []
for idx,(name,qs,params) in enumerate(ops):
    if name!='cz': continue
    a,b = qs
    i,j = pos_of[a],pos_of[b]
    if i>j: i,j=j,i
    dist = abs(pos_of[b]-pos_of[a]) - 1
    swap_counts.append(dist)
import statistics
sc = sorted(swap_counts)
print(f"cz count={len(sc)}  swaps-per-cz: min={sc[0]} med={sc[len(sc)//2]} max={sc[-1]} total={sum(sc)}")
print(f"top 15 swaps-per-cz: {sc[-15:]}")

# Now time the first 25 cz gates individually
t0=time.time()
ng=0
for idx,(name,qs,params) in enumerate(ops):
    if name=='u':
        t,p,lam = params[0]/2, params[1], params[2]
        c,s=np.cos(t),np.sin(t)
        U=np.array([[c,-np.exp(1j*lam)*s],[np.exp(1j*p)*s,np.exp(1j*(p+lam))*c]],dtype=complex)
        i=pos_of[qs[0]]
        tensors[i]=np.einsum('ab,lbr->lar',U,tensors[i])
        ng+=1
        continue
    a,b=qs
    mat=np.diag([1,1,1,-1]).astype(complex)
    i,j=pos_of[a],pos_of[b]
    if i>j: i,j=j,i; a,b=b,a
    t_gate=time.time()
    while pos_of[b] > pos_of[a]+1:
        p2=pos_of[b]
        while center>p2-1: H._move_center_left(tensors,center); center-=1
        while center<p2-1: H._move_center_right(tensors,center); center+=1
        center=H._apply_swap(tensors,p2-1,chi)
        q_at[p2-1],q_at[p2]=q_at[p2],q_at[p2-1]
        pos_of[q_at[p2-1]],pos_of[q_at[p2]]=p2-1,p2
    i,j=pos_of[a],pos_of[b]
    while center>i: H._move_center_left(tensors,center); center-=1
    while center<i: H._move_center_right(tensors,center); center+=1
    center=H._apply_2site(tensors,i,mat,chi)
    dt=time.time()-t_gate
    bond=max(t.shape[2] for t in tensors)
    print(f"  cz #{idx}: {dt:6.2f}s  bond_max={bond}  (cum {time.time()-t0:.0f}s)",flush=True)
    ng+=1
    if ng>=25: break
print(f"first 25 cz done in {time.time()-t0:.1f}s")
