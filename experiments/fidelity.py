"""Noise-aware fidelity Nyström control: train-only landmarks, no test tuning.

This is a rank-8/16 approximation, not the original full Gram-matrix SVC.
It retains actual noisy overlap measurements (1,024 shots/landmark).
"""
import sys,json,os,time
from pathlib import Path
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor,as_completed
from itertools import product
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd,joblib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from src.pipeline import synthesize_structured,quality
from src.topology import device_overlap_circuit,check_noise_coverage,noiseless_gates,audit_circuit
from src.noise import exact_probabilities
from experiments.prepare import ROOT,DATASETS
from experiments.quantum import context
from experiments.learn import SEEDS,DEVICES,threshold
from experiments.fidelity_engine import canonical,inverse_block,effect,density,contract

ENCODERS=('dense','adaptive16')
GRID=list(product((8,16),(.01,.1),(.1,1.,10.),(None,'balanced')))

def landmarks(ds):
 y=np.load(ROOT/ds/'data.npz')['y_train'];rng=np.random.default_rng(818)
 # Interleave equal counts from both classes, all from the 2,000 train rows.
 return np.column_stack([rng.choice(np.flatnonzero(y==c),8,replace=False) for c in (0,1)]).ravel()

@lru_cache(maxsize=8)
def reference(ds,enc):
 a=np.load(ROOT/ds/'data.npz');z=a[enc+'_train'][landmarks(ds)]
 return [synthesize_structured(v,710)[0] for v in z]

@lru_cache(maxsize=8)
def reference_effects(ds,enc,dev):
 a=np.load(ROOT/ds/'data.npz');z=a[enc+'_train'][landmarks(ds)];b,layout,nm,sim,R=context(dev,5)
 refs=[inverse_block(canonical(v,b,layout),b,layout) for v in z]
 check_noise_coverage(refs,nm,allow=noiseless_gates(b,layout))
 assert all(audit_circuit(c,b,layout)['off_edge']==0 for c in refs)
 return refs,np.array([effect(c,nm,R) for c in refs])

def probability_job(task):
 ds,enc,dev,part,start=task
 out=ROOT/ds/'fidelity'/enc/dev/part;out.mkdir(parents=True,exist_ok=True);dest=out/f'{start:04d}.npz'
 if dest.exists():return
 a=np.load(ROOT/ds/'data.npz');z=a[enc+'_'+part][start:start+10]
 b,layout,nm,sim,_=context(dev,5);refs,effects=reference_effects(ds,enc,dev)
 cs=[canonical(v,b,layout) for v in z];check_noise_coverage(cs,nm,allow=noiseless_gates(b,layout));assert all(audit_circuit(c,b,layout)['off_edge']==0 for c in cs)
 ps=contract(density(cs,sim),effects)
 cost=[[[c.count_ops().get('cx',0)+r.count_ops().get('cx',0),c.compose(r).depth()] for r in refs] for c in cs]
 tmp=dest.with_suffix(f'.{os.getpid()}.tmp')
 with tmp.open('wb') as stream:np.savez_compressed(stream,p=ps,cost=cost)
 tmp.replace(dest)

def simulate(parts,workers):
 tasks=[]
 for ds in DATASETS:
  for enc in ENCODERS:
   for dev in DEVICES:
    for part in parts:
     for start in range(0,{'train':2000,'val':500,'test':1000}[part],10):
      if not(ROOT/ds/'fidelity'/enc/dev/part/f'{start:04d}.npz').exists():tasks.append((ds,enc,dev,part,start))
 print('fidelity simulation',parts,len(tasks),flush=True);t=time.time()
 with ProcessPoolExecutor(max_workers=workers) as pool:
  for i,f in enumerate(as_completed([pool.submit(probability_job,t) for t in tasks]),1):
   f.result()
   if i%50==0:print('fidelity',i,'/',len(tasks),'elapsed',round(time.time()-t),flush=True)

def kernel(ds,enc,dev,part,seed):
 if dev=='ideal':
  a=np.load(ROOT/ds/'data.npz');p=(a[enc+'_'+part]@a[enc+'_train'][landmarks(ds)].T)**2
 else:
  files=sorted((ROOT/ds/'fidelity'/enc/dev/part).glob('*.npz'));assert len(files)=={'train':200,'val':50,'test':100}[part]
  p=np.concatenate([np.load(f)['p'] for f in files])
 rng=np.random.default_rng(seed+{'train':400000,'val':500000,'test':600000}[part])
 return rng.binomial(1024,np.clip(p,0,1))/1024

def projection(k,ix,rank,reg):
 W=k[ix[:rank],:rank];W=(W+W.T)/2
 w,v=np.linalg.eigh(W)
 # Positive-semidefinite regularization uses train data only.
 return v/np.sqrt(np.maximum(w,0)+reg*max(float(w.max()),1e-8))[None,:]

def train_job(ds,seed,enc):
 out=ROOT/ds/'fidelity_models'/str(seed)/enc;out.mkdir(parents=True,exist_ok=True)
 if (out/'DONE').exists():return
 a=np.load(ROOT/ds/'data.npz');y=a['y_train'];yv=a['y_val'];ix=landmarks(ds);rows=[];ks={}
 for dev in ('ideal',)+DEVICES:
  k=kernel(ds,enc,dev,'train',seed);kv=kernel(ds,enc,dev,'val',seed);ks[dev]=(k,kv)
  for gid,(rank,reg,C,weight) in enumerate(GRID):
   P=projection(k,ix,rank,reg);x=k[:,:rank]@P;v=kv[:,:rank]@P
   # No feature-wise standardization: preserves the regularized Nyström kernel.
   m=SVC(C=C,kernel='linear',class_weight=weight).fit(x,y);s=m.decision_function(v);th=threshold(yv,s)
   rows.append(dict(dataset=ds,seed=seed,encoder=enc,device=dev,gid=gid,rank=rank,regularization=reg,C=C,weight=str(weight),threshold=th,**quality(yv,s-th)))
 df=pd.DataFrame(rows);df.to_csv(out/'validation.csv',index=False)
 gid=int(df[df.device!='ideal'].groupby('gid').macro_f1.mean().idxmax());ig=int(df[df.device=='ideal'].sort_values(['macro_f1','gid'],ascending=[False,True]).iloc[0].gid)
 for dev in ('ideal',)+DEVICES:
  g=ig if dev=='ideal' else gid;rank,reg,C,weight=GRID[g];k,kv=ks[dev];P=projection(k,ix,rank,reg)
  m=SVC(C=C,kernel='linear',class_weight=weight).fit(k[:,:rank]@P,y);th=threshold(yv,m.decision_function(kv[:,:rank]@P))
  joblib.dump(dict(model=m,projection=P,rank=rank,threshold=th,gid=g),out/(dev+'.joblib'))
 (out/'selection.json').write_text(json.dumps(dict(gid=gid,ideal_gid=ig,validation_macro_f1=float(df[(df.device!='ideal')&(df.gid==gid)].macro_f1.mean())),indent=2))
 (out/'DONE').write_text('complete\n');print('fidelity trained',ds,seed,enc,flush=True)

def train(workers):
 with ProcessPoolExecutor(max_workers=workers) as pool:
  for f in as_completed([pool.submit(train_job,ds,s,e) for ds in DATASETS for s in SEEDS for e in ENCODERS]):f.result()
 chosen=[]
 for ds in DATASETS:
  for s in SEEDS:
   opts=[(json.loads((ROOT/ds/'fidelity_models'/str(s)/e/'selection.json').read_text()),e) for e in ENCODERS]
   r,e=max(opts,key=lambda t:t[0]['validation_macro_f1']);chosen.append(dict(dataset=ds,seed=s,encoder=e,**r))
 p=ROOT/'fidelity_selection.json';text=json.dumps(chosen,indent=2)
 if p.exists():assert p.read_text()==text
 else:p.write_text(text)
 from src.data import sha256
 manifest={str(p):sha256(p)}
 for f in ROOT.glob('*/fidelity_models/**/*.joblib'):manifest[str(f)]=sha256(f)
 path=ROOT/'fidelity_manifest.json'
 if path.exists():assert json.loads(path.read_text())==manifest
 else:path.write_text(json.dumps(manifest,indent=2))

def evaluate():
 from src.data import sha256
 for p,h in json.loads((ROOT/'fidelity_manifest.json').read_text()).items():assert sha256(p)==h
 chosen=json.loads((ROOT/'fidelity_selection.json').read_text());rows=[]
 for ds in DATASETS:
  a=np.load(ROOT/ds/'data.npz');y=a['y_test']
  for s in SEEDS:
   selected=next(r['encoder'] for r in chosen if r['dataset']==ds and r['seed']==s)
   for e in ENCODERS:
    out=ROOT/ds/'fidelity_evaluation'/str(s)/e;out.mkdir(parents=True,exist_ok=True)
    ideal=joblib.load(ROOT/ds/'fidelity_models'/str(s)/e/'ideal.joblib')
    for dev in ('ideal',)+DEVICES:
     k=kernel(ds,e,dev,'test',s);models=[('ideal_trained',ideal)]
     if dev!='ideal':models.append(('noise_trained',joblib.load(ROOT/ds/'fidelity_models'/str(s)/e/(dev+'.joblib'))))
     for regime,m in models:
      rank=m['rank'];score=m['model'].decision_function(k[:,:rank]@m['projection']);th=m['threshold'];pred=score>=th
      path=out/(dev+'_'+regime+'.npz');np.savez_compressed(path,labels=y,ids=a['ids_test'],score=score,threshold=th,pred=pred)
      rows.append(dict(dataset=ds,seed=s,encoder=e,device=dev,regime=regime,selected=e==selected,rank=rank,shots_per_pair=1024,shots_per_sample=rank*1024,artifact=str(path),**quality(y,score-th)))
 df=pd.DataFrame(rows);df.to_csv(ROOT/'fidelity_metrics.csv',index=False)
 df[df.selected&(df.regime=='noise_trained')].groupby(['dataset','device'])[['accuracy','macro_f1','recall','fpr']].agg(['mean','std']).to_csv(ROOT/'fidelity_summary.csv')
 (ROOT/'FIDELITY_COMPLETE').write_text('Nyström approximation, not full 2000x2000 noisy Gram matrix\n')

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=20);p.add_argument('--dev-only',action='store_true');args=p.parse_args()
 simulate(('train','val'),args.workers)
 if not args.dev_only:
  train(min(args.workers,12));simulate(('test',),args.workers);evaluate()
