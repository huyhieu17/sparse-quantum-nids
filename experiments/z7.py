"""Same-cohort Z7r2 Pegasos baseline; exact tensor-product noise contraction."""
import sys,json,time
from pathlib import Path
from functools import lru_cache
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from qiskit.quantum_info import Kraus,Operator
from src.cirillo import angle_circuit
from src.topology import map_circuit,check_noise_coverage,noiseless_gates
from experiments.quantum import context
from experiments.prepare import ROOT,DATASETS
from experiments.learn import SEEDS,DEVICES,threshold
from baselines.baseline import fit_pegasos
from src.pipeline import quality
from src.data import sha256

@lru_cache(None)
def templates(dev):
 b,layout,nm,_,_=context(dev,7)
 c=map_circuit(angle_circuit('z',7,reps=2),b,layout,710)
 inv=map_circuit(c.inverse(),b,layout,710)
 assert c.count_ops().get('cx',0)==0 and inv.count_ops().get('cx',0)==0
 if nm is not None:check_noise_coverage([c,inv],nm,allow=noiseless_gates(b,layout))
 return c,inv,nm

def channel(a,dev,inverse_effect=False):
 c,inv,nm=templates(dev);c=inv if inverse_effect else c;n=len(a);out=np.empty((n,7,2,2),complex)
 for j in range(7):
  if inverse_effect:
   r=np.eye(2) if nm is None else np.asarray(nm._local_readout_errors[(j,)].probabilities)
   m=np.broadcast_to(np.diag(r[:,0]),(n,2,2)).astype(complex).copy()
  else:m=np.zeros((n,2,2),complex);m[:,0,0]=1
  instructions=list(reversed(c.data)) if inverse_effect else c.data
  for inst in instructions:
   name=inst.operation.name
   if name=='barrier':continue
   qs=tuple(c.find_bit(q).index for q in inst.qubits)
   if qs!=(j,):continue
   if name=='rz':
    p=inst.operation.params[0]
    if hasattr(p,'parameters') and p.parameters:
     assert len(p.parameters)==1;par=next(iter(p.parameters));offset=float(p.bind({par:0}));slope=float(p.bind({par:1}))-offset;angle=offset+slope*a[:,j]
    else:angle=np.full(n,float(p))
    U=np.zeros((n,2,2),complex);U[:,0,0]=np.exp(-.5j*angle);U[:,1,1]=np.exp(.5j*angle)
   else:U=np.broadcast_to(Operator(inst.operation).data,(n,2,2))
   err=None if nm is None else nm._local_quantum_errors.get(name,{}).get(qs,nm._default_quantum_errors.get(name))
   def noise(v,adjoint=False):
    if err is None:return v
    ks=Kraus(err.to_quantumchannel()).data
    if adjoint:ks=[k.conj().T for k in ks]
    return sum(k@v@k.conj().T for k in ks)
   if inverse_effect:
    m=noise(m,True);m=U.conj().transpose(0,2,1)@m@U
   else:m=noise(U@m@U.conj().transpose(0,2,1))
  out[:,j]=m
 return out

def pair_probability(rho,effects):
 p=np.einsum('nqij,mqji->nmq',rho,effects).real.prod(axis=-1)
 assert p.min()>-1e-9 and p.max()<1+1e-9
 return np.clip(p,0,1)

def angles(ds,scale):
 a=np.load(ROOT/ds/'data.npz');x=a['pca_train'][:,:7];lo=x.min(0);span=np.maximum(x.max(0)-lo,1e-12)
 return {p:np.clip(scale*(a['pca_'+p][:,:7]-lo)/span,0,scale) for p in ('train','val','test')}

def fit():
 rows=[];selections=[]
 for ds in DATASETS:
  data=np.load(ROOT/ds/'data.npz');y=data['y_train'];yv=data['y_val'];cache={}
  for scale in (1.,float(np.pi)):
   a=angles(ds,scale)
   for dev in ('ideal',)+DEVICES:
    cache[scale,dev]=(a['train'],channel(a['train'],dev),channel(a['train'],dev,True),channel(a['val'],dev))
  for seed in SEEDS:
   out=ROOT/ds/'z7'/str(seed);out.mkdir(parents=True,exist_ok=True);models={};local=[]
   for scale in (1.,float(np.pi)):
    for dev in ('ideal',)+DEVICES:
     a,rho,eff,val=cache[scale,dev];rng=np.random.default_rng(seed+84000)
     def row(i,js):
      p=pair_probability(rho[i:i+1],eff[js])[0]
      same=np.all(a[js]==a[i],axis=1);p[same]=1.
      return rng.binomial(1024,p)/1024
     sv,coef,b=fit_pegasos(row,y,seed);p=pair_probability(val,eff[sv]);s=np.random.default_rng(seed+85000).binomial(1024,p)/1024@coef+b;th=threshold(yv,s)
     models[scale,dev]=dict(sv=sv,coef=coef,b=b,threshold=th,scale=scale)
     r=dict(dataset=ds,seed=seed,scale=scale,device=dev,refs=len(sv),zero_macro_f1=quality(yv,s)['macro_f1'],**quality(yv,s-th));local.append(r);rows.append(r)
   df=pd.DataFrame(local);scale=float(df[df.device!='ideal'].groupby('scale').macro_f1.mean().idxmax());iscale=float(df[df.device=='ideal'].sort_values('macro_f1',ascending=False).iloc[0].scale)
   selections.append(dict(dataset=ds,seed=seed,scale=scale,ideal_scale=iscale))
   for dev in ('ideal',)+DEVICES:np.savez_compressed(out/(dev+'.npz'),**models[iscale if dev=='ideal' else scale,dev])
  print('z7 fit',ds,flush=True)
 pd.DataFrame(rows).to_csv(ROOT/'z7_validation.csv',index=False);(ROOT/'z7_selection.json').write_text(json.dumps(selections,indent=2))
 manifest={str(p):sha256(p) for p in ROOT.glob('*/z7/*/*.npz')};manifest[str(ROOT/'z7_selection.json')]=sha256(ROOT/'z7_selection.json');(ROOT/'z7_manifest.json').write_text(json.dumps(manifest,indent=2))

def evaluate():
 for p,h in json.loads((ROOT/'z7_manifest.json').read_text()).items():assert sha256(p)==h
 rows=[]
 for ds in DATASETS:
  data=np.load(ROOT/ds/'data.npz');y=data['y_test'];cache={}
  for scale in (1.,float(np.pi)):
   a=angles(ds,scale)
   for dev in ('ideal',)+DEVICES:cache[scale,dev]=(channel(a['test'],dev),channel(a['train'],dev,True))
  for seed in SEEDS:
   folder=ROOT/ds/'z7'/str(seed);ideal=np.load(folder/'ideal.npz')
   for dev in ('ideal',)+DEVICES:
    models=[('ideal_trained',ideal)]
    if dev!='ideal':models.append(('noise_trained',np.load(folder/(dev+'.npz'))))
    for regime,m in models:
     scale=float(m['scale']);rho,eff=cache[scale,dev];sv=m['sv'];p=pair_probability(rho,eff[sv]);score=np.random.default_rng(seed+86000).binomial(1024,p)/1024@m['coef']+float(m['b'])
     for mode,th in (('tuned',float(m['threshold'])),('zero',0.)):
      path=folder/f'prediction_{dev}_{regime}_{mode}.npz';np.savez_compressed(path,score=score,threshold=th,pred=score>=th,labels=y,ids=data['ids_test'])
      rows.append(dict(dataset=ds,seed=seed,device=dev,regime=regime,threshold_mode=mode,refs=len(sv),shots_per_sample=len(sv)*1024,artifact=str(path),**quality(y,score-th)))
 pd.DataFrame(rows).to_csv(ROOT/'z7_metrics.csv',index=False);(ROOT/'Z7_COMPLETE').write_text('same cohort, block-compiled ZFeatureMap7r2 + Pegasos, C1000 steps100 offset1\n')

if __name__=='__main__':
 if not(ROOT/'z7_manifest.json').exists():fit()
 evaluate()
