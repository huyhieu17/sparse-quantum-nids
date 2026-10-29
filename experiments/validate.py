import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from sklearn.metrics import confusion_matrix
from src.pipeline import quality
from experiments.prepare import ROOT,DATASETS,CONFIGS
from experiments.learn import SEEDS,DEVICES,get_prob
from src.data import FILES,sha256

def run():
 manifest=json.loads((ROOT/'model_manifest.json').read_text())
 for path,digest in manifest.items():assert sha256(path)==digest,path
 assert sha256(ROOT/'frozen_selection.json')==(ROOT/'selection.sha256').read_text()
 for ds in DATASETS:
  a=np.load(ROOT/ds/'data.npz');parts=('train','val','test')
  for i,p in enumerate(parts):
   for r in parts[:i]:assert not(set(a['groups_'+p])&set(a['groups_'+r]))
  assert sha256(FILES[ds])==json.loads((ROOT/ds/'data_protocol.json').read_text())['source_sha256']
 df=pd.read_csv(ROOT/'metrics.csv');assert not df.duplicated(['dataset','seed','method','config','device','shots','mitigated','regime']).any()
 count=0
 for _,r in df.iterrows():
  a=np.load(r.artifact);source=np.load(ROOT/r.dataset/'data.npz')
  assert np.array_equal(a['ids'],source['ids_test']);assert np.array_equal(a['labels'],source['y_test'])
  assert np.array_equal(a['pred'],a['score']>=float(a['threshold']))
  for key,value in quality(a['labels'],a['score']-float(a['threshold'])).items():assert abs(value-r[key])<1e-8,(r.artifact,key)
  assert np.array_equal(confusion_matrix(a['labels'],a['pred'],labels=[0,1]).ravel(),[r.tn,r.fp,r.fn,r.tp]);count+=len(a['pred'])
 expected=len(DATASETS)*len(SEEDS)*(len(CONFIGS)*14+9+2)
 assert len(df)==expected,(len(df),expected)
 for ds in DATASETS:
  for config in CONFIGS:
   for device in ('ideal',)+DEVICES:
    for part in ('train','val','test'):
     p,perms,R,cost=get_prob(ds,config,device,part)
     assert np.isfinite(p).all() and p.min()>=0 and np.allclose(p.sum(axis=-1),1)
     assert np.all(np.sort(perms,axis=-1)==np.arange(p.shape[-1]))
     assert cost.min()>=0
 out=dict(status='PASS',metrics=len(df),predictions_verified=count,models_hashed=len(manifest),datasets=list(DATASETS),model_seeds=list(SEEDS),test_rows_per_dataset=1000,scope='fixed split; no historical non-exposure claim')
 (ROOT/'validation.json').write_text(json.dumps(out,indent=2));print(json.dumps(out),flush=True)
if __name__=='__main__':run()
