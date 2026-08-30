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

t0 = time.time()
for idx,(name,qs,params) in enumerate(ops):
    if time.time()-t0 > 180:
        print(f"!! 180s wall at gate {idx} ({name})", flush=True); break
    t_g = time.time()
    if name == 'u':
        t,p,lam = params[0]/2, params[1], params[2]
        c,s=np.cos(t),np.sin(t)
        U=np.array([[c,-np.exp(1j*lam)*s],[np.exp(1j*p)*s,np.exp(1j*(p+lam))*c]],dtype=complex)
        i=pos_of[qs[0]]
        tensors[i]=np.einsum('ab,lbr->lar',U,tensors[i])
    else:
        a,b=qs
        mat=np.diag([1,1,1,-1]).astype(complex)
        i,j=pos_of[a],pos_of[b]
        if i>j: i,j=j,i; a,b=b,a
        dist = pos_of[b]-pos_of[a]-1
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
        dt = time.time()-t_g
        bond = max(t.shape[2] for t in tensors)
        print(f"cz g{idx:4d} q{qs} dist={dist:2d} bond={bond:3d} dt={dt:7.2f}s wall={time.time()-t0:7.1f}s", flush=True)
    if idx % 25 == 0:
        bond = max(t.shape[2] for t in tensors)
        print(f"  ...g{idx:4d} ({name}) bond={bond:3d} wall={time.time()-t0:7.1f}s", flush=True)
