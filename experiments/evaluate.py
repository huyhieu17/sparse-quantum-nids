"""Score frozen models; all test output is written only in this stage."""
import sys,json,hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd,joblib
from sklearn.metrics import confusion_matrix
from src.pipeline import quality
from experiments.prepare import ROOT,DATASETS,CONFIGS
from experiments.learn import SEEDS,DEVICES,features,get_prob

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def freeze():
 assert (ROOT/'TRAINING_COMPLETE').exists()
 assert sha(ROOT/'frozen_selection.json')==(ROOT/'selection.sha256').read_text()
 files=sorted(ROOT.glob('*/models/**/*.joblib'))+sorted(ROOT.glob('*/models/**/selection.json'))+[ROOT/'frozen_selection.json']
 manifest={str(p):sha(p) for p in files}
 path=ROOT/'model_manifest.json'
 if path.exists():assert json.loads(path.read_text())==manifest
 else:path.write_text(json.dumps(manifest,indent=2))

def job(ds,seed):
 out=ROOT/ds/'evaluation'/str(seed);out.mkdir(parents=True,exist_ok=True)
 if (out/'metrics.csv').exists():return
 chosen=next(r for r in json.loads((ROOT/'frozen_selection.json').read_text()) if r['dataset']==ds and r['seed']==seed)['config']
 a=np.load(ROOT/ds/'data.npz');y=a['y_test'];ids=a['ids_test'];rows=[]
 def record(name,device,config,score,t,shots,mit,cost,regime,selected=False):
  pred=(score>=t).astype(int);tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
  key=f'{name}_{device}_{config}_{shots}_{mit}_{regime}'
  np.savez_compressed(out/(key+'.npz'),labels=y,ids=ids,score=score,threshold=t,pred=pred)
  rows.append(dict(dataset=ds,seed=seed,method=name,config=config,device=device,shots=shots,mitigated=mit,regime=regime,selected=selected,threshold=t,n_test=len(y),tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp),cx_per_sample=float(cost[0]),depth_per_circuit=float(cost[1]),circuits_per_sample=0 if device=='classical' else 2,calibration_shots=0 if not mit else 2*(3 if config=='angle3x2' else 5)*4096,artifact=str(out/(key+'.npz')),**quality(y,score-t)))
 for config in CONFIGS:
  modeldir=ROOT/ds/'models'/str(seed)/config
  ideal=joblib.load(modeldir/'ideal.joblib')
  for device in ('ideal',)+DEVICES:
   x=features(ds,config,device,'test',seed);cost=get_prob(ds,config,device,'test')[3];cost=(cost[:,:,0].sum(axis=1).mean(),cost[:,:,1].mean())
   models=[('ideal_trained',ideal)]
   if device!='ideal':models.append(('noise_trained',joblib.load(modeldir/(device+'.joblib'))))
   for regime,m in models:
    score=m['model'].decision_function(m['scaler'].transform(x))
    for name,t in (('projected',m['threshold']),('projected_zero_threshold',0.)):
     record(name,device,config,score,t,1024,False,cost,regime,config==chosen)
   if config==chosen and device!='ideal':
    for shots,mit in ((1024,True),(4096,False),(4096,True)):
     m=joblib.load(modeldir/f'{device}_{shots}_{mit}.joblib');x=features(ds,config,device,'test',seed,shots,mit)
     score=m['model'].decision_function(m['scaler'].transform(x));record('projected_supplement',device,config,score,m['threshold'],shots,mit,cost,'noise_trained',True)
 for name in ('rbf_pca','extra_trees_raw'):
  m=joblib.load(ROOT/ds/'models'/str(seed)/'classical'/(name+'.joblib'));x=m['scaler'].transform(a[m['key']+'_test'])
  score=m['model'].decision_function(x) if m['decision'] else m['model'].predict_proba(x)[:,1]-.5
  record(name,'classical',name,score,m['threshold'],0,False,(0,0),'classical',False)
 pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
 print('evaluated',ds,seed,flush=True)

def run():
 freeze()
 with ProcessPoolExecutor(max_workers=8) as pool:
  for f in as_completed([pool.submit(job,ds,s) for ds in DATASETS for s in SEEDS]):f.result()
 df=pd.concat([pd.read_csv(p) for p in ROOT.glob('*/evaluation/*/metrics.csv')],ignore_index=True)
 df.to_csv(ROOT/'metrics.csv',index=False)
 keys=['dataset','device','method','config','regime','shots','mitigated']
 df.groupby(keys)[['accuracy','macro_f1','recall','fpr','cx_per_sample','depth_per_circuit']].agg(['mean','std']).to_csv(ROOT/'all_candidates_summary.csv')
 primary=df[(df.method=='projected')&(df.regime=='noise_trained')&df.selected]
 primary.groupby(['dataset','device'])[['accuracy','macro_f1','recall','fpr','cx_per_sample']].agg(['mean','std']).to_csv(ROOT/'selected_summary.csv')
 (ROOT/'EVALUATED').write_text('complete\n')

if __name__=='__main__':run()
