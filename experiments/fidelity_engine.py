"""Exact channel contraction for separately compiled overlap blocks.

Compute Tr[E_ref^*(readout_zero) rho_query]. This is mathematically identical
to simulating the concatenated, non-boundary-optimized overlap circuit.
It does NOT approximate the noise channel or omit inverse-preparation noise.
"""
from functools import lru_cache
from collections import deque
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix,Kraus,Operator
from src.pipeline import synthesize_structured
from src.topology import map_circuit,backend_coupling,check_noise_coverage,noiseless_gates

def canonical(v,b,layout):
 c=map_circuit(synthesize_structured(v,710)[0],b,layout,710);q=len(layout)
 final=c.layout.final_index_layout();at=np.argsort(final).tolist();fix=QuantumCircuit(q)
 pos={p:i for i,p in enumerate(layout)};adj={i:set() for i in range(q)}
 for a,z in backend_coupling(b):
  if a in pos and z in pos:adj[pos[a]].add(pos[z]);adj[pos[z]].add(pos[a])
 for i in range(q):
  if at[i]==i:continue
  j=at.index(i);queue=deque([[i]])
  while queue:
   path=queue.popleft()
   if path[-1]==j:break
   for nxt in sorted(adj[path[-1]]):
    if nxt not in path:queue.append(path+[nxt])
  edges=list(zip(path[:-1],path[1:]))
  for a,z in edges+edges[-2::-1]:fix.swap(a,z)
  at[i],at[j]=at[j],at[i]
 if fix.size():
  fix=map_circuit(fix,b,layout,710);assert fix.layout.final_index_layout()==list(range(q));c=c.compose(fix)
 assert at==list(range(q))
 return c

def inverse_block(c,b,layout):
 inv=map_circuit(c.inverse(),b,layout,710)
 assert inv.layout.final_index_layout()==list(range(len(layout)))
 return inv

def effect(c,nm,R):
 """Adjoint of every gate+noise channel, applied to the readout POVM."""
 M=np.diag(R[:,0]).astype(complex);cache={}
 for inst in reversed(c.data):
  name=inst.operation.name;qs=tuple(c.find_bit(q).index for q in inst.qubits)
  err=nm._local_quantum_errors.get(name,{}).get(qs,nm._default_quantum_errors.get(name))
  if err is not None:
   key=(name,qs)
   if key not in cache:cache[key]=Kraus([k.conj().T for k in Kraus(err.to_quantumchannel()).data])
   M=DensityMatrix(M).evolve(cache[key],qargs=list(qs)).data
  U=Operator(inst.operation).data
  M=DensityMatrix(M).evolve(Operator(U.conj().T),qargs=list(qs)).data
 assert np.allclose(M,M.conj().T,atol=1e-10)
 return M

def density(cs,sim):
 saved=[]
 for c in cs:
  qc=c.copy();qc.save_density_matrix(label='rho');saved.append(qc)
 r=sim.run(saved,shots=1).result()
 return np.array([np.asarray(r.data(i)['rho']) for i in range(len(cs))])

def contract(rhos,effects):
 p=np.einsum('nij,mji->nm',rhos,effects).real
 assert p.min()>-1e-9 and p.max()<1+1e-9
 return np.clip(p,0,1)
