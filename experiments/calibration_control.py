"""Isolate noisy training from just recalibrating the ideal model threshold."""
import sys,json,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd,joblib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from experiments.prepare import ROOT
from experiments.learn import GRID,DEVICES,features,threshold
from src.pipeline import quality
from src.data import sha256

def fit():
 selections=json.loads((ROOT/'frozen_selection.json').read_text())
 for r in selections:
  ds,seed,config=r['dataset'],r['seed'],r['config'];out=ROOT/ds/'calibration_control'/str(seed);out.mkdir(parents=True,exist_ok=True)
  if (out/'DONE').exists():continue
  data=np.load(ROOT/ds/'data.npz');x=features(ds,config,'ideal','train',seed);sc=StandardScaler().fit(x);C,g,w=GRID[r['gid']]
  model=SVC(C=C,gamma=g/x.shape[1],class_weight=w).fit(sc.transform(x),data['y_train'])
  for dev in DEVICES:
   v=features(ds,config,dev,'val',seed);s=model.decision_function(sc.transform(v));th=threshold(data['y_val'],s)
   joblib.dump(dict(model=model,scaler=sc,threshold=th,config=config),out/(dev+'.joblib'))
  (out/'DONE').write_text('fit only ideal measurements; threshold from noisy validation\n')
 manifest={str(p):sha256(p) for p in ROOT.glob('*/calibration_control/*/*.joblib')};p=ROOT/'calibration_control_manifest.json'
 if p.exists():assert json.loads(p.read_text())==manifest
 else:p.write_text(json.dumps(manifest,indent=2))

def evaluate():
 for p,h in json.loads((ROOT/'calibration_control_manifest.json').read_text()).items():assert sha256(p)==h
 rows=[]
 for r in json.loads((ROOT/'frozen_selection.json').read_text()):
  ds,seed,config=r['dataset'],r['seed'],r['config'];out=ROOT/ds/'calibration_control'/str(seed);a=np.load(ROOT/ds/'data.npz')
  for dev in DEVICES:
   m=joblib.load(out/(dev+'.joblib'));x=features(ds,config,dev,'test',seed);s=m['model'].decision_function(m['scaler'].transform(x));th=m['threshold'];path=out/(dev+'_prediction.npz')
   np.savez_compressed(path,score=s,threshold=th,pred=s>=th,labels=a['y_test'],ids=a['ids_test'])
   rows.append(dict(dataset=ds,seed=seed,config=config,device=dev,artifact=str(path),**quality(a['y_test'],s-th)))
 pd.DataFrame(rows).to_csv(ROOT/'calibration_control.csv',index=False);(ROOT/'CONTROL_COMPLETE').write_text('complete\n')

if __name__=='__main__':
 fit()
 while not(ROOT/'EVALUATED').exists():time.sleep(10)
 evaluate()
