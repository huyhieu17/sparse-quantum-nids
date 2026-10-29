import unittest
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit_aer.noise import NoiseModel,ReadoutError
from src.noise import make_simulator
from src.pipeline import normalize
from src.topology import map_circuit
from experiments.quantum import probabilities,context,state_circuit
from experiments.learn import threshold,feature_sample
from src.pipeline import quality

class NoiseAware(unittest.TestCase):
 def test_joint_asymmetric_readout(self):
  c=QuantumCircuit(2);c.x(0);c.h(1)
  r0=np.array([[.9,.1],[.3,.7]]);r1=np.array([[.8,.2],[.4,.6]])
  nm=NoiseModel();nm.add_readout_error(ReadoutError(r0),[0]);nm.add_readout_error(ReadoutError(r1),[1])
  R=np.kron(r1,r0);p=probabilities([c],nm,make_simulator(nm,max_parallel_threads=1),R)[0]
  self.assertTrue(np.allclose(p,Statevector.from_instruction(c).probabilities()@R))
  c.measure_all();counts=make_simulator(nm,seed=1,max_parallel_threads=1).run(c,shots=100000).result().get_counts()
  empirical=np.array([counts.get(format(i,'02b'),0)/100000 for i in range(4)])
  self.assertLess(np.max(np.abs(p-empirical)),.006)

 def test_routed_logical_observables(self):
  rng=np.random.default_rng(3)
  for device in ('nairobi','lagos','perth'):
   b,layout,_,_,_=context(device,5)
   for config in ('dense','adaptive16','angle5x4'):
    v=normalize(rng.normal(size=32))
    if config=='adaptive16':v[16:]=0;v=normalize(v)
    logical=state_circuit(v,config,b,layout)
    for basis in ('X','Z'):
     c=logical.copy()
     if basis=='X':
      for j in range(5):c.h(j)
     routed=map_circuit(c,b,layout,710);final=routed.layout.final_index_layout()
     perm=[sum(((k>>j)&1)<<final[j] for j in range(5)) for k in range(32)]
     self.assertTrue(np.allclose(Statevector.from_instruction(c).probabilities(),Statevector.from_instruction(routed).probabilities()[perm],atol=1e-10))

 def test_threshold_exhaustive_with_ties(self):
  rng=np.random.default_rng(5)
  for _ in range(30):
   y=rng.integers(2,size=50);s=np.round(rng.normal(size=50),1);t=threshold(y,s)
   candidates=np.r_[s.min(),np.nextafter(np.unique(s),np.inf)]
   expected=max(quality(y,s-x)['macro_f1'] for x in candidates)
   self.assertAlmostEqual(quality(y,s-t)['macro_f1'],expected)

 def test_joint_sampling_preserves_bell_correlation(self):
  p=np.array([[[.5,0,0,.5],[.5,0,0,.5]]]);perms=np.tile(np.arange(4),(1,2,1))
  f=feature_sample(p,perms,'ideal',11,1024)
  self.assertEqual(f[0,2],1.);self.assertEqual(f[0,5],1.)
  self.assertEqual(f[0,0],f[0,1])

 def test_channel_contraction_equals_full_noisy_overlap(self):
  from experiments.fidelity_engine import canonical,inverse_block,effect,density,contract
  from src.noise import exact_probabilities
  rng=np.random.default_rng(471)
  for device in ('nairobi','lagos','perth'):
   b,layout,nm,sim,R=context(device,5)
   vs=normalize(rng.normal(size=(3,32)));cs=[canonical(v,b,layout) for v in vs]
   for v,c in zip(vs,cs):
    self.assertTrue(Statevector.from_instruction(c).equiv(Statevector(v)))
   inv=[inverse_block(c,b,layout) for c in cs[:2]]
   fast=contract(density(cs,sim),np.array([effect(c,nm,R) for c in inv]))
   full=np.array([exact_probabilities([c.compose(r) for r in inv],nm,sim) for c in cs])
   self.assertTrue(np.allclose(fast,full,atol=1e-10),np.max(np.abs(fast-full)))

 def test_z7_factorization_equals_full_seven_qubit_simulation(self):
  from experiments.z7 import templates,channel,pair_probability
  from src.noise import exact_probabilities
  a=np.random.default_rng(621).uniform(0,np.pi,(3,7))
  for device in ('ideal','nairobi','lagos','perth'):
   c,inv,nm=templates(device);sim=context(device,7)[3]
   full=[]
   for x in a:
    q=c.assign_parameters(dict(zip(c.parameters,x)))
    cs=[q.compose(inv.assign_parameters(dict(zip(inv.parameters,r)))) for r in a[:2]]
    full.append(exact_probabilities(cs,nm,sim))
   fast=pair_probability(channel(a,device),channel(a[:2],device,True))
   self.assertTrue(np.allclose(fast,full,atol=1e-10),np.max(np.abs(fast-full)))

if __name__=='__main__':unittest.main()
