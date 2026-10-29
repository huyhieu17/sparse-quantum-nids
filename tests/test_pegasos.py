import unittest
import numpy as np
from src.cirillo import pegasos_qsvc,pegasos_score
class PegasosRegression(unittest.TestCase):
 def test_callable_bias_and_repeated_training_indices(self):
  for seed in (11,23,37):
   rng=np.random.default_rng(seed);X=rng.normal(size=(8,5));X/=np.linalg.norm(X,axis=1)[:,None];K=(X@X.T)**2;y=np.arange(8)%2
   class Cache:
    def row(self,i,js):return K[i,js]
   cache=Cache();model=pegasos_qsvc(cache,len(y),y,seed=seed)
   alpha=np.zeros(len(y));labels=1-2*y;random=np.random.default_rng(seed)
   for step in range(1,101):
    i=int(random.integers(len(y)));value=np.dot(alpha*labels,K[i]+1)
    if labels[i]*1000/step*value<1:alpha[i]+=1
   expected=-(K+1)@(alpha*labels)
   np.testing.assert_allclose(pegasos_score(cache,model,np.arange(len(y))),expected,atol=1e-12)
if __name__=='__main__':unittest.main()
