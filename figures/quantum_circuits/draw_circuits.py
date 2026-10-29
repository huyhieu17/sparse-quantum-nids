"""Qiskit drawings of actual frozen v11 input encodings; no model changes."""
import os,sys,json
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/icc-circuit-matplotlib')
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from qiskit import QuantumCircuit,qasm3
from qiskit.circuit.library import StatePreparation
from qiskit.quantum_info import Statevector,state_fidelity
from src.pipeline import synthesize_structured
from src.data import sha256

OUT=Path(__file__).resolve().parent
DATA=Path(__file__).resolve().parents[2]/'results/detection'
DATASETS=('ToN-IoT','RT-IoT','NSL-KDD','UNSW-NB15')
CONFIGS=('dense','adaptive16','adaptive8')
TITLES=('Before: dense (32 amplitudes)','After: sparse (at most 16 amplitudes)','After: sparse (at most 8 amplitudes)')

def compact(v):
 support=np.flatnonzero(v!=0);active=[];fixed={};c=QuantumCircuit(5)
 for b in range(5):
  bits=(support>>b)&1
  if np.all(bits==bits[0]):fixed[b]=int(bits[0])
  else:active.append(b)
 for b,bit in fixed.items():
  if bit:c.x(b)
 if active:
  small=np.zeros(2**len(active))
  for j in support:small[sum(((int(j)>>b)&1)<<i for i,b in enumerate(active))]=v[j]
  block=QuantumCircuit(len(active),name='StatePrep')
  block.append(StatePreparation(small),range(len(active)))
  c.append(block.to_gate(label='StatePrep'),active)
 elif v[support[0]]<0:c.global_phase=np.pi
 return c,active,fixed

def run():
 rows=[];manifest={};fig,axes=plt.subplots(4,3,figsize=(23,17))
 for row,ds in enumerate(DATASETS):
  bundle=PdfPages(OUT/f'{ds}_before_after.pdf')
  path=DATA/ds/'data.npz';manifest[str(path)]=sha256(path);a=np.load(path);idx=0
  for col,(config,title) in enumerate(zip(CONFIGS,TITLES)):
   v=a[config+'_train'][idx];c,active,fixed=compact(v);actual,stats=synthesize_structured(v,710)
   fidelity=float(state_fidelity(Statevector.from_instruction(actual),v));assert fidelity>1-1e-10
   assert state_fidelity(Statevector.from_instruction(c),v)>1-1e-10
   retained=float(abs(np.vdot(a['dense_train'][idx],v))**2)
   label=f'{ds} | {title}'
   blockfig=c.draw(output='mpl',fold=-1,initial_state=True,style={'name':'iqp'},scale=1.)
   blockpath=OUT/f'{ds}_{config}_block.png'
   blockfig.savefig(blockpath,dpi=160,bbox_inches='tight');plt.close(blockfig)
   axes[row,col].imshow(plt.imread(blockpath));axes[row,col].axis('off')
   axes[row,col].set_title(label+'\n'+f'Active prep qubits: {len(active)}/5 | CX: {stats["cx"]} | depth: {stats["depth"]}',fontsize=12,pad=15)
   # Gate-level figure is the actual output of the synthesis used in experiments.
   detail=actual.draw(output='mpl',fold=32,initial_state=True,style={'name':'iqp'},scale=.75)
   detail.suptitle(label+f'\ntrain index 0 | CX={stats["cx"]}, depth={stats["depth"]} | logical state preparation, before hardware routing',fontsize=12,y=1.01)
   stem=OUT/f'{ds}_{config}_gates'
   detail.savefig(str(stem)+'.png',dpi=180,bbox_inches='tight');detail.savefig(str(stem)+'.pdf',bbox_inches='tight');bundle.savefig(detail,bbox_inches='tight');plt.close(detail)
   actual.draw(output='text',fold=120,filename=str(stem)+'.txt')
   (OUT/f'{ds}_{config}.qasm').write_text(qasm3.dumps(actual))
   rows.append(dict(dataset=ds,config=config,train_index=idx,source_row_id=int(a['ids_train'][idx]),nonzero_amplitudes=int(np.count_nonzero(v)),physical_wires=5,active_prep_qubits=len(active),fixed_bits=fixed,cx=stats['cx'],depth=stats['depth'],gate_counts=dict(actual.count_ops()),state_preparation_fidelity=fidelity,dense_sparse_state_fidelity=retained))
  bundle.close()
 fig.suptitle('Dense vs structured sparse amplitude encoding — Qiskit circuits',fontsize=22,y=.995)
 fig.text(.5,.012,'Same first training sample within each dataset; all circuits retain 5 wires. StatePrep blocks are expanded in the accompanying gate-level files.\nCX/depth refer to logical state preparation in basis {rx, ry, rz, h, cx}; no hardware routing, noise channels, or measurement gates.',ha='center',fontsize=12)
 fig.subplots_adjust(top=.94,bottom=.075,hspace=.5,wspace=.22)
 fig.savefig(OUT/'all_datasets_dense_vs_sparse.png',dpi=190,bbox_inches='tight');fig.savefig(OUT/'all_datasets_dense_vs_sparse.pdf',bbox_inches='tight');plt.close(fig)
 (OUT/'circuit_metrics.json').write_text(json.dumps(rows,indent=2));(OUT/'source_manifest.json').write_text(json.dumps(manifest,indent=2))
 lines=['# Quantum circuits: dense → sparse','','Generated directly with Qiskit `QuantumCircuit.draw(output="mpl")`.','','The overview shows StatePreparation blocks; the twelve `*_gates.pdf/png` figures expand the real synthesized circuits. QASM and text files are also supplied. All use the first training sample (index 0, chosen without performance filtering) from each frozen v11 dataset. Dense, adaptive16 and adaptive8 use the same sample and frontend.','','These are encoder illustrations, not a claim that all selected classifiers use sparse encoding. The UNSW v12/v12b supervised frontends are different; these figures deliberately use the common four-dataset v11 ablation for a controlled comparison.','','Metrics are for logical state preparation before backend routing and measurement. Noise is applied by the simulator during execution, not drawn as fictitious circuit gates. All five wires are retained; active preparation qubits can decrease when sparse support forms a subcube. Adaptive sparsification may fall back to unstructured top-k, so qubit/CX reductions are sample-dependent.','','| Dataset | Encoding | Nonzero | Active prep qubits | CX | Depth | Dense-state fidelity |','|---|---|---:|---:|---:|---:|---:|']
 for r in rows:lines.append(f'| {r["dataset"]} | {r["config"]} | {r["nonzero_amplitudes"]} | {r["active_prep_qubits"]} | {r["cx"]} | {r["depth"]} | {r["dense_sparse_state_fidelity"]:.4f} |')
 lines+=['','Every synthesized state was verified against its target using statevector fidelity (>1−1e−10). Dense-state fidelity measures the approximation caused by sparsification; it is not classification accuracy.','','![Overview](all_datasets_dense_vs_sparse.png)']
 (OUT/'README.md').write_text('\n'.join(lines)+'\n');print(json.dumps(rows,indent=2))

if __name__=='__main__':run()
