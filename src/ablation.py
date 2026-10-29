"""Controlled encoder ablations on the same fitted 32-amplitude front end."""
from itertools import combinations
import hashlib
import numpy as np
from .pipeline import normalize


def cubes(support):
    k=int(np.log2(support));result=[]
    for active in combinations(range(5),k):
        fixed=[b for b in range(5) if b not in active]
        for pattern in range(2**len(fixed)):
            base=sum(((pattern>>i)&1)<<b for i,b in enumerate(fixed))
            result.append([base+sum(((j>>i)&1)<<b for i,b in enumerate(active)) for j in range(support)])
    return np.asarray(result)


class Encoder:
    def __init__(self,kind,support=32,loss=1.,seed=11,permutation=None,rank=None):
        self.kind,self.support,self.loss,self.seed=kind,support,loss,seed
        self.permutation=np.arange(32) if permutation is None else np.asarray(permutation)
        self.rank=np.arange(31) if rank is None else np.asarray(rank)
        self.candidates=cubes(support) if support<32 else np.arange(32)[None,:]

    def fit(self,V):
        a=np.asarray(V)[:,self.permutation]
        energy=np.mean(a*a,axis=0)
        self.fixed=self.candidates[np.argmax(energy[self.candidates].sum(axis=1))]
        return self

    def transform(self,V):
        V=np.atleast_2d(V)
        if self.kind=='dense':
            if self.support==32:return V.copy()
            return normalize(V[:,np.r_[self.rank[:self.support-1],31]])
        a=V[:,self.permutation];out=np.zeros_like(a)
        for start in range(0,len(a),512):
            z=a[start:start+512]
            top=np.argsort(-z*z,axis=1,kind='stable')[:,:self.support]
            if self.kind=='topk':chosen=top
            elif self.kind=='fixed':chosen=np.tile(self.fixed,(len(z),1))
            elif self.kind=='random':
                # A deterministic function of each flow, never its label/order.
                ids=[int.from_bytes(hashlib.blake2b(v.tobytes(),digest_size=8,person=str(self.seed).encode()).digest(),'little')%len(self.candidates) for v in z]
                chosen=self.candidates[ids]
            elif self.kind=='adaptive':
                energies=(z[:,self.candidates]**2).sum(axis=2)
                best=energies.argmax(axis=1)
                retained=np.take_along_axis(z*z,top,axis=1).sum(axis=1)
                use=energies[np.arange(len(z)),best]>=(1-self.loss)*retained
                chosen=top.copy();chosen[use]=self.candidates[best[use]]
            else:raise ValueError(self.kind)
            np.put_along_axis(out[start:start+len(z)],chosen,np.take_along_axis(z,chosen,axis=1),axis=1)
        return normalize(out)


def predict(V,refs,coef,b):
    return np.concatenate([(V[i:i+512]@refs.T)**2@coef+b for i in range(0,len(V),512)])
