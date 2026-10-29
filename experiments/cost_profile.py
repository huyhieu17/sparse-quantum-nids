"""Gate-only ASAP duration estimate on eight fixed inputs per configuration.

No pulse alignment, readout, reset, queues or idle-noise scheduling is modeled;
this is not a measured QPU latency. Full-test CX/depth remain in metrics.csv.
"""
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from experiments.prepare import ROOT,DATASETS,CONFIGS
from experiments.learn import DEVICES
from experiments.quantum import context,state_circuit
from src.topology import map_circuit

def job(task):
 ds,cfg,dev=task;q=3 if cfg=='angle3x2' else 5;b,layout,*_=context(dev,q);props=b.properties();a=np.load(ROOT/ds/'data.npz');key=cfg if cfg in ('dense','adaptive16','adaptive8') else 'angle';rows=[]
 for i,v in enumerate(a[key+'_test'][:8]):
  state=state_circuit(v,cfg,b,layout)
  for basis in ('Z','X'):
   c=state.copy()
   if basis=='X':
    for j in range(q):c.h(j)
   c=map_circuit(c,b,layout,710);ends=np.zeros(q);total=0.
   for inst in c.data:
    qs=[c.find_bit(x).index for x in inst.qubits];physical=[layout[j] for j in qs]
    duration=float(props.gate_length(inst.operation.name,physical))*1e9
    finish=max(ends[qs])+duration;ends[qs]=finish;total+=duration
   rows.append(dict(dataset=ds,config=cfg,device=dev,input_index=i,basis=basis,cx=c.count_ops().get('cx',0),depth=c.depth(),asap_gate_only_ns=max(ends),sum_gate_duration_ns=total))
 return rows

if __name__=='__main__':
 rows=[]
 with ProcessPoolExecutor(max_workers=4) as pool:
  for f in as_completed([pool.submit(job,(ds,c,d)) for ds in DATASETS for c in CONFIGS for d in DEVICES]):rows.extend(f.result())
 pd.DataFrame(rows).to_csv(ROOT/'gate_duration_profile.csv',index=False)
 print('gate-only duration profiles',len(rows),'PASS',flush=True)
