"""Combine completed, independently frozen experiments without test selection."""
import sys,json,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from src.pipeline import quality
from src.data import sha256
from experiments.prepare import ROOT,DATASETS
from experiments.learn import SEEDS,DEVICES

def macro(y,pred):
 y=np.asarray(y,dtype=bool);pred=np.asarray(pred,dtype=bool)
 tp=(pred&y).sum(axis=-1);tn=(~pred&~y).sum(axis=-1);fp=(pred&~y).sum(axis=-1);fn=(~pred&y).sum(axis=-1)
 return 100*(2*tp/np.maximum(2*tp+fp+fn,1)+2*tn/np.maximum(2*tn+fp+fn,1))/2

def paired(a,b,seed=710):
 aa=[np.load(p) for p in a.sort_values('seed').artifact];bb=[np.load(p) for p in b.sort_values('seed').artifact]
 assert len(aa)==len(bb)==4
 for x,z in zip(aa,bb):assert np.array_equal(x['ids'],z['ids']) and np.array_equal(x['labels'],z['labels'])
 y=aa[0]['labels'];pa=np.array([x['pred'] for x in aa]);pb=np.array([x['pred'] for x in bb]);ix=np.random.default_rng(seed).integers(len(y),size=(1000,len(y)))
 d=(macro(y[ix],pa[:,ix])-macro(y[ix],pb[:,ix])).mean(axis=0)
 return dict(delta_macro_f1=float((macro(y,pa)-macro(y,pb)).mean()),ci_low=float(np.percentile(d,2.5)),ci_high=float(np.percentile(d,97.5)))

def run():
 for flag in ('COMPLETE','FIDELITY_COMPLETE','CONTROL_COMPLETE','Z7_COMPLETE'):assert (ROOT/flag).exists(),flag
 primary=pd.read_csv(ROOT/'metrics.csv');fidelity=pd.read_csv(ROOT/'fidelity_metrics.csv');z7=pd.read_csv(ROOT/'z7_metrics.csv');control=pd.read_csv(ROOT/'calibration_control.csv')
 selected=primary[(primary.method=='projected')&(primary.regime=='noise_trained')&primary.selected]
 zselected=z7[(z7.regime=='noise_trained')&(z7.threshold_mode=='tuned')]
 fselected=fidelity[fidelity.selected&(fidelity.regime=='noise_trained')]
 # Validate all additional records and frozen artifacts independently.
 n=0
 for df in (primary,fidelity,z7,control):
  for _,r in df.iterrows():
   a=np.load(r.artifact);data=np.load(ROOT/r.dataset/'data.npz')
   assert np.array_equal(a['ids'],data['ids_test']) and np.array_equal(a['labels'],data['y_test'])
   assert np.array_equal(a['pred'],a['score']>=float(a['threshold']))
   for k,v in quality(a['labels'],a['score']-float(a['threshold'])).items():assert abs(v-r[k])<1e-8
   n+=len(a['pred'])
 for filename in ('model_manifest.json','fidelity_manifest.json','z7_manifest.json','calibration_control_manifest.json','historical_hashes.json'):
  for p,h in json.loads((ROOT/filename).read_text()).items():assert sha256(p)==h,p
 for df,manifest in ((primary,'model_manifest.json'),(fidelity,'fidelity_manifest.json'),(z7,'z7_manifest.json'),(control,'calibration_control_manifest.json')):
  frozen_at=(ROOT/manifest).stat().st_mtime
  assert all(Path(p).stat().st_mtime>=frozen_at for p in df.artifact)
 from experiments.fidelity import ENCODERS,landmarks
 probability_batches=0
 for ds in DATASETS:
  data=np.load(ROOT/ds/'data.npz');ix=landmarks(ds)
  assert len(set(ix))==16 and np.array_equal(np.bincount(data['y_train'][ix]),[8,8])
  for enc in ENCODERS:
   for dev in DEVICES:
    for part,count in (('train',200),('val',50),('test',100)):
     paths=list((ROOT/ds/'fidelity'/enc/dev/part).glob('*.npz'));assert len(paths)==count
     for path in paths:
      a=np.load(path);p=a['p'];cost=a['cost'];assert p.shape==(10,16) and cost.shape==(10,16,2)
      assert np.isfinite(p).all() and p.min()>=0 and p.max()<=1 and cost.min()>=0
      probability_batches+=1
 expected=[1744,224,224,48];assert [len(x) for x in (primary,fidelity,z7,control)]==expected
 comparison=[];stats=[]
 for ds in DATASETS:
  for dev in DEVICES:
   a=selected[(selected.dataset==ds)&(selected.device==dev)];z=zselected[(zselected.dataset==ds)&(zselected.device==dev)];f=fselected[(fselected.dataset==ds)&(fselected.device==dev)];c=control[(control.dataset==ds)&(control.device==dev)]
   r=dict(dataset=ds,device=dev,accuracy=a.accuracy.mean(),accuracy_sd=a.accuracy.std(),macro_f1=a.macro_f1.mean(),macro_f1_sd=a.macro_f1.std(),z7_accuracy=z.accuracy.mean(),z7_macro_f1=z.macro_f1.mean(),fidelity_accuracy=f.accuracy.mean(),fidelity_macro_f1=f.macro_f1.mean(),calibrated_ideal_macro_f1=c.macro_f1.mean(),fpr=a.fpr.mean(),recall=a.recall.mean(),projected_shots=1024,z7_shots=z.shots_per_sample.mean(),fidelity_shots=f.shots_per_sample.mean())
   comparison.append(r)
   for name,b in [('Z7 noise-trained, tuned threshold',z),('Ideal-trained, noisy threshold, same hyperparameters',c)]+[(m,primary[(primary.dataset==ds)&(primary.method==m)]) for m in ('rbf_pca','extra_trees_raw')]:stats.append(dict(dataset=ds,device=dev,comparator=name,**paired(a,b)))
 comp=pd.DataFrame(comparison);comp.to_csv(ROOT/'comparison.csv',index=False);pd.DataFrame(stats).to_csv(ROOT/'paired_comparisons.csv',index=False)
 # Sensitivity to identical model inputs after train-fitted preprocessing.
 duplicate_counts={};sensitivity=[]
 for ds in DATASETS:
  a=np.load(ROOT/ds/'data.npz');hashes={p:pd.util.hash_pandas_object(pd.DataFrame(a['raw_'+p]),index=False).to_numpy() for p in ('train','val','test')}
  keep=~np.isin(hashes['test'],np.r_[hashes['train'],hashes['val']]);duplicate_counts[ds]=int((~keep).sum())
  for _,r in selected[selected.dataset==ds].iterrows():
   p=np.load(r.artifact);sensitivity.append(dict(dataset=ds,seed=r.seed,device=r.device,n_test=int(keep.sum()),**quality(p['labels'][keep],p['score'][keep]-float(p['threshold']))))
 pd.DataFrame(sensitivity).to_csv(ROOT/'postprocess_duplicate_sensitivity.csv',index=False)
 lines=['','## Đối chứng trên cùng test cohort: projected / fidelity / Z7','',
 'Fidelity dùng Nyström rank 8/16, 1.024 shots/cặp tham chiếu; Z7–Pegasos dùng 7 qubit, r=2, C=1000, 100 steps, kernel offset=1. Cả hai được train dưới nhiễu, chọn tham số và ngưỡng trên validation. Tất cả dùng cùng 2.000 nhãn train và cùng 1.000 test rows.',
 'Fidelity và Z7 dùng các block chuẩn bị/inverse được compile riêng; giữ đầy đủ lỗi inverse và readout. Đây là thay đổi compiler so với v10, không phải tái lập chính xác paper. Nyström không phải full Gram-matrix SVC.',
 '', '| Dataset | Backend | Projected Macro-F1 | Fidelity Nyström Macro-F1 | Z7–Pegasos Macro-F1 |','|---|---|---:|---:|---:|']
 for r in comparison:lines.append(f"| {r['dataset']} | {r['device']} | {r['macro_f1']:.2f} | {r['fidelity_macro_f1']:.2f} | {r['z7_macro_f1']:.2f} |")
 lines+=['','## Huấn luyện có nhiễu so với chỉ chỉnh ngưỡng','',
 'Đối chứng giữ cùng encoder và C/gamma/class weight với lựa chọn noise-aware, train bằng phép đo ideal, rồi chọn ngưỡng trên validation noisy. Vì vậy chênh lệch dưới đây không chỉ là lợi ích của việc được chỉnh ngưỡng.',
 '', '| Dataset | Backend | Ideal train + noisy calibration F1 | Noisy train F1 |','|---|---|---:|---:|']
 for r in comparison:lines.append(f"| {r['dataset']} | {r['device']} | {r['calibrated_ideal_macro_f1']:.2f} | {r['macro_f1']:.2f} |")
 lines+=['','## Readout mitigation và số shots','',
 'Đây là các ablation đã quy định trước. Không chọn nhánh tốt nhất trên test làm headline. Calibration là mô phỏng 4.096 shots cho mỗi trạng thái |0>, |1> của từng qubit, giả định chuẩn bị calibration lý tưởng.',
 '', '| Dataset | Total shots/sample | Mitigated | Macro-F1 trung bình 3 backend |','|---|---:|---|---:|']
 supplemental=primary[primary.method=='projected_supplement']
 for (ds,shots,mit),g in supplemental.groupby(['dataset','shots','mitigated']):lines.append(f'| {ds} | {shots} | {mit} | {g.macro_f1.mean():.2f} |')
 lines+=['','## Sparsity, mạch nông và chi phí','',
 'Bảng ablation trung bình trên ba backend và bốn seeds, không dùng để chọn lại mô hình sau test. CX là tổng cho hai measurement settings mỗi mẫu.',
 '', '| Dataset | Encoder | Macro-F1 (%) | CX/sample |','|---|---|---:|---:|']
 for (ds,c),g in primary[(primary.method=='projected')&(primary.regime=='noise_trained')].groupby(['dataset','config']):lines.append(f'| {ds} | {c} | {g.macro_f1.mean():.2f} | {g.cx_per_sample.mean():.2f} |')
 lines+=['','## Độ tin cậy và diễn giải cho paper','',
 '- `paired_comparisons.csv`: paired bootstrap 1.000 lần, cùng row resampling cho bốn fitted models. CI có điều kiện trên split/model đã fit, không phải CI qua nhiều phép chia độc lập; chưa hiệu chỉnh multiple comparisons.',
 '- Không tuyên bố quantum advantage: đối chứng cổ điển rất mạnh và nhìn chung ngang hoặc hơn projected pipeline. Product-state control cũng cạnh tranh ở một số dataset.',
 '- Kết quả hỗ trợ khả năng chống nhiễu của pipeline projected + noise-aware fitting, không chứng minh adaptive encoding luôn tốt nhất. Cấu hình headline được khóa bằng validation; giữ nguyên cả khi encoder khác tốt hơn trên test.',
 '- UNSW-NB15 chưa đạt 95%; FPR còn cao. Cần external/temporal validation và các split độc lập trước khi kết luận tổng quát.',
 f'- Kiểm tra trùng sau preprocessing: {duplicate_counts}. UNSW có 3/1.000 test vectors trùng input đã biến đổi của train/validation dù fingerprint gốc khác. Có bảng sensitivity loại 3 mẫu này, không refit hoặc đổi lựa chọn.',
 f'- Đã đối chiếu {sum(expected):,} dòng metrics với {n:,} dự đoán; kiểm tra hash mô hình, nguồn và kết quả v10. Sáu numerical tests mới kiểm chứng routing, readout, threshold, joint sampling, channel contraction và factorization Z7.','']
 report=Path('experiments/REPORT_VI.md');base=report.read_text().split('\n## Đối chứng trên cùng test cohort:')[0];report.write_text(base+'\n'.join(lines))
 tex=['% V11 frozen validation selection; fixed internal split, four training/measurement seeds.',r'\begin{tabular}{llrrrr}',r'\hline',r'Dataset & Backend & Accuracy & Macro-F1 & Z7 F1 & Nystr\"om F1 \\',r'\hline']
 for r in comparison:tex.append(f"{r['dataset']} & {r['device'].title()} & {r['accuracy']:.2f} & {r['macro_f1']:.2f} & {r['z7_macro_f1']:.2f} & {r['fidelity_macro_f1']:.2f} "+r'\\')
 tex +=[r'\hline',r'\end{tabular}','% Projected: 1024 total shots/sample; fidelity: 8192/16384; Z7: refs*1024.', '% Not external validation, QPU execution, or evidence of quantum advantage.']
 (ROOT/'paper_table.tex').write_text('\n'.join(tex)+'\n')
 validation=dict(status='PASS',metrics=sum(expected),predictions_verified=n,fidelity_probability_batches=probability_batches,model_freeze_precedes_test_predictions=True,rows_by_family=dict(zip(('projected_and_classical','fidelity_nystrom','z7','threshold_control'),expected)),transformed_duplicate_test_rows=duplicate_counts,tests='6 numerical tests passed; see experiments/tests.log')
 (ROOT/'final_validation.json').write_text(json.dumps(validation,indent=2))
 from experiments.provenance import run as provenance
 provenance();(ROOT/'ALL_COMPLETE').write_text('all four datasets, projected/fidelity/Z7/control/mitigation/shot ablations validated\n')
 print(comp.to_string(index=False),flush=True);print(json.dumps(validation),flush=True)

if __name__=='__main__':
 while not all((ROOT/f).exists() for f in ('COMPLETE','FIDELITY_COMPLETE','CONTROL_COMPLETE','Z7_COMPLETE')):time.sleep(10)
 run()
