"""Joint outcome probabilities for shallow/projected encodings, with audits."""
import sys,json,time,os
from pathlib import Path
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor,as_completed
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from src.pipeline import synthesize_structured
from src.cirillo import load_device
from src.topology import select_layout,restrict_noise_model,map_circuit,check_noise_coverage,noiseless_gates,audit_circuit,backend_coupling
from src.noise import make_simulator
from experiments.prepare import ROOT,DATASETS,CONFIGS

@lru_cache(None)
def context(device,q):
 b=load_device('nairobi' if device=='ideal' else device);layout=select_layout(b,q)['layout']
 nm=None if device=='ideal' else restrict_noise_model(b,layout,thermal_relaxation=True)
 sim=make_simulator(nm,max_parallel_threads=1)
 R=np.ones((1,1))
 for j in reversed(range(q)):
  r=np.eye(2) if nm is None else np.asarray(nm._local_readout_errors[(j,)].probabilities)
  R=np.kron(R,r)
 return b,layout,nm,sim,R

def state_circuit(v,config,b,layout):
 q=len(layout)
 if config in ('dense','adaptive16','adaptive8'):return synthesize_structured(v,710)[0]
 layers=int(config[-1]);c=QuantumCircuit(q)
 pos={p:i for i,p in enumerate(layout)}
 # Choose a deterministic physical spanning tree: no routing SWAPs for angle encoders.
 edges=sorted({tuple(sorted((pos[u],pos[w]))) for u,w in backend_coupling(b) if u in pos and w in pos})
 visited={0};tree=[]
 while len(visited)<q:
  edge=next((u,w) for u,w in edges if (u in visited)!=(w in visited));tree.append(edge);visited.update(edge)
 for layer in range(layers):
  for j in range(q):
   k=2*(layer*q+j);c.ry(float(v[k]) if k<len(v) else 0.,j);c.rz(float(v[k+1]) if k+1<len(v) else 0.,j)
  if not config.startswith('product'):
   for u,w in tree:c.cx(u,w)
 return c

def probabilities(circuits,nm,sim,R):
 if nm is None:return np.array([Statevector.from_instruction(c).probabilities() for c in circuits])
 saved=[]
 for c in circuits:
  c=c.copy();c.save_probabilities(label='p');saved.append(c)
 result=sim.run(saved,shots=1).result()
 p=np.array([result.data(i)['p'] for i in range(len(saved))])@R
 p=np.maximum(p,0);p/=p.sum(axis=1,keepdims=True)
 return p

def job(task):
 ds,config,device,part,start=task
 out=ROOT/ds/'probabilities'/config/device/part;out.mkdir(parents=True,exist_ok=True);dest=out/f'{start:04d}.npz'
 if dest.exists():return
 q=3 if config=='angle3x2' else 5;b,layout,nm,sim,R=context(device,q)
 data=np.load(ROOT/ds/'data.npz');key=config if config in ('dense','adaptive16','adaptive8') else 'angle';v=data[key+'_'+part][start:start+50]
 cs=[];cost=[];perms=[]
 for x in v:
  state=state_circuit(x,config,b,layout)
  for basis in ('Z','X'):
   c=state.copy()
   if basis=='X':
    for j in range(q):c.h(j)
   c=map_circuit(c,b,layout,710)
   audit=audit_circuit(c,b,layout);assert audit['off_edge']==0
   if nm is not None:check_noise_coverage([c],nm,allow=noiseless_gates(b,layout))
   final=c.layout.final_index_layout() if c.layout is not None else list(range(q))
   # Index physical output by logical bit string; routing can differ per state.
   perms.append([sum(((k>>j)&1)<<final[j] for j in range(q)) for k in range(2**q)])
   cost.append([audit['cx'],audit['depth']]);cs.append(c)
 p=probabilities(cs,nm,sim,R).reshape(len(v),2,2**q)
 tmp=dest.with_suffix(f'.{os.getpid()}.tmp')
 with tmp.open('wb') as stream:np.savez_compressed(stream,p=p,perms=np.array(perms).reshape(len(v),2,2**q),cost=np.array(cost).reshape(len(v),2,2),layout=layout,readout=R)
 tmp.replace(dest)

def run(parts,workers=20,configs=CONFIGS):
 tasks=[]
 for ds in DATASETS:
  for config in configs:
   for device in ('ideal','nairobi','lagos','perth'):
    for part in parts:
     n={'train':2000,'val':500,'test':1000}[part]
     for start in range(0,n,50):
      if not(ROOT/ds/'probabilities'/config/device/part/f'{start:04d}.npz').exists():tasks.append((ds,config,device,part,start))
 print('probability jobs',len(tasks),'parts',parts,flush=True);t=time.time()
 with ProcessPoolExecutor(max_workers=workers) as pool:
  for i,f in enumerate(as_completed([pool.submit(job,t) for t in tasks]),1):
   f.result()
   if i%50==0:print('probabilities',i,'/',len(tasks),'elapsed',round(time.time()-t),flush=True)

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--test',action='store_true');p.add_argument('--workers',type=int,default=20);a=p.parse_args()
 run(('test',) if a.test else ('train','val'),a.workers)
