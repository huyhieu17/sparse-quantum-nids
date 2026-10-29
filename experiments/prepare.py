"""Prepare one group-disjoint, train-only-fitted dataset; no test scoring."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.decomposition import PCA
from src.data import load_dataset,FILES,sha256
from src.cirillo import _columns,stratified_cap
from src.pipeline import DenseEncoder
from src.ablation import Encoder

ROOT=Path('results/detection')
DATASETS=('ToN-IoT','RT-IoT','NSL-KDD','UNSW-NB15')
CONFIGS=('dense','adaptive16','adaptive8','angle3x2','angle5x2','angle5x4','product5x4')

def prepare(ds):
 out=ROOT/ds;out.mkdir(parents=True,exist_ok=True)
 if (out/'data.npz').exists():return
 f=load_dataset(ds);counts=f.groupby('_fingerprint').label.nunique();bad=set(counts[counts>1].index)
 f=f[~f._fingerprint.isin(bad)].copy()
 groups=f.drop_duplicates('_fingerprint');g=groups._fingerprint.to_numpy();y=groups.label.to_numpy()
 tr,hold=train_test_split(np.arange(len(g)),test_size=.3,stratify=y,random_state=20261026)
 va,te=train_test_split(hold,test_size=2/3,stratify=y[hold],random_state=20261027)
 # Sample rows after assigning whole groups, retaining natural row prevalence.
 parts=[]
 for ids,n,seed in ((tr,2000,710),(va,500,711),(te,1000,712)):
  sub=f[f._fingerprint.isin(g[ids])];ix=stratified_cap(sub.label.to_numpy(),n,seed);parts.append(sub.iloc[ix])
 a,b,c=parts
 assert all(not(set(parts[i]._fingerprint)&set(parts[j]._fingerprint)) for i in range(3) for j in range(i))
 num,cat=_columns(a);cols=num+cat
 for p in parts:
  for col in num:p.loc[:,col]=__import__('pandas').to_numeric(p[col],errors='coerce').replace([np.inf,-np.inf],np.nan)
 pre=ColumnTransformer([('num',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),num),('cat',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('encode',OneHotEncoder(handle_unknown='ignore',min_frequency=5,sparse_output=False))]),cat)])
 raw=[np.asarray(pre.fit_transform(a[cols]),float),np.asarray(pre.transform(b[cols]),float),np.asarray(pre.transform(c[cols]),float)]
 assert all(np.isfinite(x).all() for x in raw)
 sc=StandardScaler().fit(raw[0]);pca=PCA(32,random_state=710).fit(sc.transform(raw[0]));z=[pca.transform(sc.transform(x)) for x in raw]
 base=DenseEncoder().fit(z[0],np.ones(32));amp=[base.transform(x) for x in z]
 scale=np.maximum(np.quantile(np.abs(z[0]),.75,axis=0),.1)
 arrays={}
 for i,part in enumerate(('train','val','test')):
  arrays['raw_'+part]=raw[i];arrays['pca_'+part]=z[i];arrays['angle_'+part]=2*np.arctan(z[i]/scale)
  arrays['dense_'+part]=amp[i];arrays['y_'+part]=parts[i].label.to_numpy(dtype=int)
  arrays['ids_'+part]=parts[i]._row_id.to_numpy();arrays['groups_'+part]=parts[i]._fingerprint.to_numpy(dtype=str)
 for k in (8,16):
  enc=Encoder('adaptive',k,.3,710).fit(amp[0])
  for i,part in enumerate(('train','val','test')):arrays[f'adaptive{k}_{part}']=enc.transform(amp[i])
 np.savez_compressed(out/'data.npz',**arrays)
 (out/'data_protocol.json').write_text(json.dumps(dict(dataset=ds,source_sha256=sha256(FILES[ds]),conflicting_groups_removed=len(bad),sizes=[len(p) for p in parts],class_counts=[p.label.value_counts().to_dict() for p in parts],features=cols,scope='v11 holdout, historical non-exposure not guaranteed'),indent=2))
 print('prepared',ds,[len(p) for p in parts],flush=True)

if __name__=='__main__':
 for ds in DATASETS:prepare(ds)
