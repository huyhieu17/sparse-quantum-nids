"""Validation-only model selection. This module never loads test labels."""
import sys,json,hashlib
from pathlib import Path
from itertools import product,combinations
from concurrent.futures import ProcessPoolExecutor,as_completed
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd,joblib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from src.pipeline import quality
from experiments.prepare import ROOT,DATASETS,CONFIGS
from experiments.quantum import context

SEEDS=(11,23,37,53);DEVICES=('nairobi','lagos','perth')
GRID=list(product((.1,1.,10.,100.),(.03,.1,1.),(None,'balanced')))

def threshold(y,s):
 order=np.argsort(s,kind='stable');v=np.asarray(s)[order];yy=np.asarray(y)[order]
 ends=np.r_[np.flatnonzero(np.diff(v)!=0)+1,len(v)]
 n0=(yy==0).sum();n1=(yy==1).sum();fn=np.r_[0,np.cumsum(yy)[ends-1]];tn=np.r_[0,ends]-fn
 tp=n1-fn;fp=n0-tn
 f1=(np.divide(2*tp,2*tp+fp+fn,out=np.zeros_like(tp,dtype=float),where=2*tp+fp+fn>0)+np.divide(2*tn,2*tn+fp+fn,out=np.zeros_like(tp,dtype=float),where=2*tn+fp+fn>0))/2
 ts=np.r_[v[0],np.nextafter(v[ends-1],np.inf)]
 best=np.flatnonzero(np.isclose(f1,f1.max(),rtol=0,atol=1e-14));j=best[np.argmin(np.abs(ts[best]))]
 return float(ts[j])

def get_prob(ds,config,device,part):
 files=sorted((ROOT/ds/'probabilities'/config/device/part).glob('*.npz'))
 expected={'train':40,'val':10,'test':20}[part];assert len(files)==expected,(ds,config,device,part,len(files))
 a=[np.load(p) for p in files]
 return np.concatenate([x['p'] for x in a]),np.concatenate([x['perms'] for x in a]),a[0]['readout'],np.concatenate([x['cost'] for x in a])

def feature_sample(p,perms,device,seed,shots,mitigated=False):
 q=int(np.log2(p.shape[-1]));rng=np.random.default_rng(seed)
 hist=np.array([rng.multinomial(shots//2,x) for x in p.reshape(-1,2**q)],float).reshape(p.shape)/(shots//2)
 if mitigated and device!='ideal':
  nm=context(device,q)[2];rngcal=np.random.default_rng(seed%1000+98200);R=np.ones((1,1))
  for j in reversed(range(q)):
   r=np.asarray(nm._local_readout_errors[(j,)].probabilities)
   estimated=np.array([rngcal.multinomial(4096,row)/4096 for row in r]);R=np.kron(R,estimated)
  hist=hist@np.linalg.inv(R)
 logical=np.take_along_axis(hist,perms,axis=-1)
 bits=1-2*((np.arange(2**q)[:,None]>>np.arange(q))&1)
 obs=np.column_stack([bits]+[bits[:,i]*bits[:,j] for i,j in combinations(range(q),2)])
 return (logical@obs).reshape(len(p),-1)

def features(ds,config,device,part,seed,shots=1024,mitigated=False):
 p,perms,_,_=get_prob(ds,config,device,part)
 # Separate train/validation/test RNG streams; same draws for mitigation ablation.
 stream=seed+{'train':100000,'val':200000,'test':300000}[part]
 return feature_sample(p,perms,device,stream,shots,mitigated)

def fit_candidate(ds,config,seed):
 out=ROOT/ds/'models'/str(seed)/config;out.mkdir(parents=True,exist_ok=True)
 if (out/'DONE').exists():return
 data=np.load(ROOT/ds/'data.npz');y=data['y_train'];yv=data['y_val'];rows=[];cached={}
 for device in ('ideal',)+DEVICES:
  x=features(ds,config,device,'train',seed);v=features(ds,config,device,'val',seed)
  sc=StandardScaler().fit(x);x=sc.transform(x);v=sc.transform(v);cached[device]=(sc,x,v)
  for gid,(C,gamma,weight) in enumerate(GRID):
   model=SVC(C=C,gamma=gamma/x.shape[1],class_weight=weight).fit(x,y);s=model.decision_function(v);th=threshold(yv,s)
   rows.append(dict(dataset=ds,config=config,seed=seed,device=device,gid=gid,C=C,gamma=gamma,weight=str(weight),threshold=th,**quality(yv,s-th),zero_macro_f1=quality(yv,s)['macro_f1']))
 df=pd.DataFrame(rows);df.to_csv(out/'validation.csv',index=False)
 # Same grid index across the three devices, selected by mean validation F1.
 gid=int(df[df.device!='ideal'].groupby('gid').macro_f1.mean().idxmax())
 ideal_gid=int(df[df.device=='ideal'].sort_values(['macro_f1','gid'],ascending=[False,True]).iloc[0].gid)
 for device in ('ideal',)+DEVICES:
  g=ideal_gid if device=='ideal' else gid;C,gamma,weight=GRID[g];sc,x,v=cached[device]
  model=SVC(C=C,gamma=gamma/x.shape[1],class_weight=weight).fit(x,y);th=threshold(yv,model.decision_function(v))
  joblib.dump(dict(model=model,scaler=sc,threshold=th,gid=g),out/(device+'.joblib'))
 (out/'selection.json').write_text(json.dumps(dict(gid=gid,ideal_gid=ideal_gid,validation_macro_f1=float(df[(df.device!='ideal')&(df.gid==gid)].macro_f1.mean())),indent=2))
 (out/'DONE').write_text('validation selection complete\n');print('fit',ds,seed,config,flush=True)

def fit_controls(ds,seed):
 out=ROOT/ds/'models'/str(seed)/'classical';out.mkdir(parents=True,exist_ok=True)
 if (out/'DONE').exists():return
 a=np.load(ROOT/ds/'data.npz');y=a['y_train'];yv=a['y_val'];rows=[]
 for name,key in (('rbf_pca','pca'),('extra_trees_raw','raw')):
  x=a[key+'_train'];v=a[key+'_val'];sc=StandardScaler().fit(x);x=sc.transform(x);v=sc.transform(v);best=None
  configs=GRID if name=='rbf_pca' else (1,2,5,10)
  for param in configs:
   if name=='rbf_pca':C,g,w=param;model=SVC(C=C,gamma=g/x.shape[1],class_weight=w)
   else:model=ExtraTreesClassifier(n_estimators=300,min_samples_leaf=param,class_weight='balanced',random_state=seed,n_jobs=1)
   model.fit(x,y);s=model.decision_function(v) if name=='rbf_pca' else model.predict_proba(v)[:,1]-.5;th=threshold(yv,s);m=quality(yv,s-th)
   rows.append(dict(model=name,param=str(param),threshold=th,**m))
   if best is None or m['macro_f1']>best[0]:best=(m['macro_f1'],dict(model=model,scaler=sc,threshold=th,key=key,decision=name=='rbf_pca'))
  joblib.dump(best[1],out/(name+'.joblib'))
 pd.DataFrame(rows).to_csv(out/'validation.csv',index=False);(out/'DONE').write_text('complete\n')

def select():
 chosen=[]
 for ds in DATASETS:
  for seed in SEEDS:
   rows=[]
   for config in CONFIGS:
    r=json.loads((ROOT/ds/'models'/str(seed)/config/'selection.json').read_text());rows.append((r['validation_macro_f1'],config,r))
   _,config,r=max(rows,key=lambda t:t[0]);chosen.append(dict(dataset=ds,seed=seed,config=config,**r))
 path=ROOT/'frozen_selection.json'
 text=json.dumps(chosen,indent=2)
 if path.exists():assert path.read_text()==text
 else:path.write_text(text)
 (ROOT/'selection.sha256').write_text(hashlib.sha256(path.read_bytes()).hexdigest())
 print('FROZEN',text,flush=True)

def supplement(ds,seed):
 chosen=json.loads((ROOT/'frozen_selection.json').read_text());r=next(r for r in chosen if r['dataset']==ds and r['seed']==seed);config=r['config']
 a=np.load(ROOT/ds/'data.npz');y=a['y_train'];yv=a['y_val'];C,gamma,weight=GRID[r['gid']]
 for device in DEVICES:
  for shots,mit in ((1024,True),(4096,False),(4096,True)):
   path=ROOT/ds/'models'/str(seed)/config/f'{device}_{shots}_{mit}.joblib'
   if path.exists():continue
   x=features(ds,config,device,'train',seed,shots,mit);v=features(ds,config,device,'val',seed,shots,mit);sc=StandardScaler().fit(x)
   model=SVC(C=C,gamma=gamma/x.shape[1],class_weight=weight).fit(sc.transform(x),y);s=model.decision_function(sc.transform(v));th=threshold(yv,s)
   joblib.dump(dict(model=model,scaler=sc,threshold=th,validation=quality(yv,s-th)),path)

def run(workers=12):
 with ProcessPoolExecutor(max_workers=workers) as pool:
  tasks=[pool.submit(fit_candidate,ds,c,s) for ds in DATASETS for s in SEEDS for c in CONFIGS]
  tasks +=[pool.submit(fit_controls,ds,s) for ds in DATASETS for s in SEEDS]
  for f in as_completed(tasks):f.result()
 select()
 with ProcessPoolExecutor(max_workers=workers) as pool:
  for f in as_completed([pool.submit(supplement,ds,s) for ds in DATASETS for s in SEEDS]):f.result()
 (ROOT/'TRAINING_COMPLETE').write_text('primary and supplementary models frozen before test scoring\n')

if __name__=='__main__':run()
