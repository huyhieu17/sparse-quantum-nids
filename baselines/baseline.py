"""Frozen Pegasos helper used by v11 and v13; obsolete v9 driver omitted."""
import numpy as np

def fit_pegasos(row,y,seed):
 # Sign convention reversed from Qiskit so positive score means attack.
 rng=np.random.default_rng(seed);y=2*np.asarray(y)-1;alpha={}
 for step in range(1,101):
  i=int(rng.integers(len(y)));js=np.array(list(alpha),dtype=int)
  value=float(np.dot([alpha[j]*y[j] for j in js],row(i,js)+1)) if len(js) else 0.
  if y[i]*1000/step*value<1:alpha[i]=alpha.get(i,0)+1
 sv=np.array(list(alpha));coef=np.array([alpha[j]*y[j] for j in sv],float)
 return sv,coef,float(coef.sum())
