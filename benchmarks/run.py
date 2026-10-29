import os,sys,time,json,platform,argparse
from pathlib import Path
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','RAYON_NUM_THREADS'):os.environ[key]='1'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from itertools import combinations
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from experiments.quantum import context,state_circuit
from experiments.z7 import templates,angles
from baselines.baseline import fit_pegasos
from src.topology import map_circuit
from src.data import sha256
from src.cirillo import stratified_cap

ROOT=Path('results/timing');DATA=Path('results/detection')
DATASETS=('ToN-IoT','RT-IoT','NSL-KDD','UNSW-NB15');METHODS=('dense','adaptive16','adaptive8','z7')

class Runner:
 def __init__(self,dev,q):
  self.b,self.layout,nm,*_=context(dev,q);self.q=q
  self.sim=AerSimulator(noise_model=nm,method='density_matrix',seed_simulator=11,max_parallel_threads=1,max_parallel_experiments=1,max_parallel_shots=1)
  c=QuantumCircuit(q);c.x(0);c.measure_all();self.sim.run(c,shots=32).result();self.reset()
 def reset(self):self.seconds=0.;self.circuits=0;self.shots=0;self.calls=0
 def measure(self,cs,shots):
  out=[]
  for start in range(0,len(cs),32):
   batch=cs[start:start+32];t=time.perf_counter();res=self.sim.run(batch,shots=shots,seed_simulator=11+self.calls).result();assert res.success
   for j in range(len(batch)):
    counts=res.get_counts(j);assert sum(counts.values())==shots
    h=np.zeros(2**self.q)
    for bits,n in counts.items():h[int(bits.replace(' ',''),2)]=n/shots
    out.append(h)
   self.seconds+=time.perf_counter()-t;self.calls+=1;self.circuits+=len(batch);self.shots+=len(batch)*shots
  return np.asarray(out)
 def counters(self):return dict(quantum_seconds=self.seconds,circuits=self.circuits,shots=self.shots,batches=self.calls)

def projected(r,ds,method):
 a=np.load(DATA/ds/'data.npz');ix={'train':stratified_cap(a['y_train'],200,1301),'test':np.sort(np.random.default_rng(1302).choice(len(a['y_test']),20,replace=False))};bits=1-2*((np.arange(32)[:,None]>>np.arange(5))&1);obs=np.column_stack([bits]+[bits[:,i]*bits[:,j] for i,j in combinations(range(5),2)])
 def extract(part,n):
  t=time.perf_counter();cs=[];perms=[]
  for v in a[method+'_'+part][ix[part]]:
   prep=state_circuit(v,method,r.b,r.layout)
   for basis in ('Z','X'):
    c=prep.copy()
    if basis=='X':
     for j in range(5):c.h(j)
    c=map_circuit(c,r.b,r.layout,710);final=c.layout.final_index_layout() if c.layout is not None else list(range(5))
    perms.append([sum(((k>>j)&1)<<final[j] for j in range(5)) for k in range(32)]);c.measure_all();cs.append(c)
  compile_s=time.perf_counter()-t;h=r.measure(cs,512);t=time.perf_counter();x=(np.take_along_axis(h,np.array(perms),axis=1)@obs).reshape(n,30);return x,compile_s,time.perf_counter()-t
 result={};t=time.perf_counter();x,compile_s,post=extract('train',200);ct=time.perf_counter();sc=StandardScaler().fit(x);model=SVC(C=10,gamma=1/30,class_weight='balanced').fit(sc.transform(x),a['y_train'][ix['train']]);fit_s=time.perf_counter()-ct
 result['train']=dict(wall_seconds=time.perf_counter()-t,compile_seconds=compile_s,classical_seconds=fit_s+post,**r.counters());r.reset()
 t=time.perf_counter();x,compile_s,post=extract('test',20);ct=time.perf_counter();score=model.decision_function(sc.transform(x));predict_s=time.perf_counter()-ct
 result['predict']=dict(wall_seconds=time.perf_counter()-t,compile_seconds=compile_s,classical_seconds=predict_s+post,**r.counters());result['refs']=len(model.support_);return result,score

def z7(r,ds,dev):
 a=np.load(DATA/ds/'data.npz');it=stratified_cap(a['y_train'],200,1301);iq=np.sort(np.random.default_rng(1302).choice(len(a['y_test']),20,replace=False));v=angles(ds,float(np.pi));v['train']=v['train'][it];v['test']=v['test'][iq];ct=time.perf_counter();c,inv,_=templates(dev)
 def bind(template,vals):return [template.assign_parameters(dict(zip(template.parameters,row))) for row in vals]
 train=bind(c,v['train'][:200]);inverse=bind(inv,v['train'][:200]);compile_s=time.perf_counter()-ct;bind_seconds=compile_s
 def overlaps(query,js):
  cs=[];t=time.perf_counter()
  for j in js:
   circuit=query.compose(inverse[int(j)]);circuit.measure_all();cs.append(circuit)
  nonlocal compile_s
  compile_s+=time.perf_counter()-t
  return r.measure(cs,1024)[:,0] if len(cs) else np.empty(0)
 def row(i,js):
  same=np.all(v['train'][js]==v['train'][i],axis=1);out=np.ones(len(js));out[~same]=overlaps(train[i],js[~same]);return out
 t=time.perf_counter();sv,coef,b=fit_pegasos(row,a['y_train'][it],11);wall=time.perf_counter()-t
 # Include initial template binding in total training time.
 result={}
 # initial binding measured separately before starting fit.
 result['train']=dict(wall_seconds=wall+bind_seconds,compile_seconds=compile_s,classical_seconds=max(0,wall+bind_seconds-compile_s-r.seconds),**r.counters())
 r.reset();compile_s=0;t=time.perf_counter();ct=time.perf_counter();query=bind(c,v['test'][:20]);compile_s+=time.perf_counter()-ct;score=[]
 for q in query:score.append(overlaps(q,sv)@coef+b)
 wall=time.perf_counter()-t;result['predict']=dict(wall_seconds=wall,compile_seconds=compile_s,classical_seconds=max(0,wall-compile_s-r.seconds),**r.counters());result['refs']=len(sv);return result,np.array(score)

def run(pilot=False):
 ROOT.mkdir(exist_ok=True);jobs=[]
 for dev in ('nairobi','lagos','perth'):
  for ds in DATASETS:
   for rep in range(3):
    for method in METHODS[rep:]+METHODS[:rep]:jobs.append((ds,dev,rep,method))
 if pilot:jobs=jobs[:4]
 for ds,dev,rep,method in jobs:
  dest=ROOT/f'{ds}_{dev}_{rep}_{method}.json'
  if dest.exists():continue
  print('START',ds,dev,rep,method,flush=True);r=Runner(dev,7 if method=='z7' else 5)
  result,score=z7(r,ds,dev) if method=='z7' else projected(r,ds,method)
  assert np.isfinite(score).all();assert result['predict']['shots']==(20*result['refs']*1024 if method=='z7' else 20*1024)
  payload=dict(dataset=ds,device=dev,repetition=rep,method=method,train_rows=200,test_rows=20,seed=11,score=score.tolist(),**result)
  dest.write_text(json.dumps(payload,indent=2));print('DONE',method,round(result['train']['wall_seconds'],3),round(result['predict']['wall_seconds'],3),flush=True)
 files=[Path(__file__),Path('benchmarks/PROTOCOL.md')]+[DATA/ds/'data.npz' for ds in DATASETS]
 import qiskit,qiskit_aer,sklearn
 (ROOT/'environment.json').write_text(json.dumps(dict(platform=platform.platform(),python=sys.version,qiskit=qiskit.__version__,aer=qiskit_aer.__version__,sklearn=sklearn.__version__,cpu=Path('/proc/cpuinfo').read_text().split('model name')[1].split('\n')[0],hashes={str(p):sha256(p) for p in files}),indent=2))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--pilot',action='store_true');run(p.parse_args().pilot)
