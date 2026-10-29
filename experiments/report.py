import sys,json,os
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from sklearn.metrics import f1_score
from experiments.prepare import ROOT,DATASETS
from experiments.learn import DEVICES,SEEDS

def run():
 df=pd.read_csv(ROOT/'metrics.csv');selected=df[(df.method=='projected')&(df.regime=='noise_trained')&df.selected]
 rows=[]
 for (ds,device),g in selected.groupby(['dataset','device']):
  row=dict(dataset=ds,device=device,accuracy=g.accuracy.mean(),accuracy_sd=g.accuracy.std(),macro_f1=g.macro_f1.mean(),macro_f1_sd=g.macro_f1.std(),recall=g.recall.mean(),fpr=g.fpr.mean(),configs=', '.join(sorted(set(g.config))))
  rows.append(row)
 tab=pd.DataFrame(rows);tab.to_csv(ROOT/'headline.csv',index=False)
 paired=[]
 for _,r in selected.iterrows():
  a=np.load(r.artifact);control=df[(df.dataset==r.dataset)&(df.seed==r.seed)&(df.device==r.device)&(df.config==r.config)&(df.method=='projected')&(df.regime=='ideal_trained')].iloc[0]
  b=np.load(control.artifact);assert np.array_equal(a['ids'],b['ids']);y=a['labels'];rng=np.random.default_rng(int(r.seed)+900)
  # Conditional paired bootstrap over rows: no claim of split-level uncertainty.
  from experiments.finalize import macro
  ix=rng.integers(len(y),size=(1000,len(y)))
  diffs=macro(y[ix],a['pred'][ix])-macro(y[ix],b['pred'][ix])
  paired.append(dict(dataset=r.dataset,seed=r.seed,device=r.device,config=r.config,delta_macro_f1=r.macro_f1-control.macro_f1,ci_low=np.percentile(diffs,2.5),ci_high=np.percentile(diffs,97.5)))
 pd.DataFrame(paired).to_csv(ROOT/'paired_noise_training.csv',index=False)
 lines=['# V11 — Huấn luyện và chọn mô hình dưới nhiễu','',
 '**Đây là phương pháp projected quantum kernel mới**, không phải kết quả fidelity kernel v10 được sửa lại. Không so sánh chênh lệch điểm v10/v11 như cùng test cohort.','',
 '## Thiết lập','',
 '- Bốn dataset; 2.000 train / 500 validation / 1.000 test mỗi dataset, tách theo fingerprint. Preprocessing/PCA chỉ fit 2.000 train; không dùng MLP giám sát.',
 '- Một phép chia cố định, bốn lần huấn luyện/đo độc lập (11,23,37,53). SD phản ánh measurement/training randomness trên split này, không phải độ biến thiên giữa các split.',
 '- Nairobi/Lagos/Perth: gate + thermal + asymmetric readout, cùng snapshot v10. Mô phỏng density matrix rồi lấy mẫu đa thức joint outcomes; không chạy QPU.',
 '- Bảng chính tổng 1.024 shots/mẫu, chia 512 cho Z và 512 cho X. Kernel RBF trên single-/two-qubit correlators; các observables dùng chung bitstrings, không giả lập độc lập.',
 '- Chọn encoder và C/gamma/class weight theo Macro-F1 validation trung bình ba backend; mỗi backend có model và ngưỡng riêng. Không chọn backend tốt nhất trên test.',
 '- Supplement tăng lên 4.096 shots và/hoặc mitigation readout từ calibration 4.096 shots/trạng thái/qubit, báo riêng chi phí. Mitigation không sửa gate errors.',
 '- Đây là holdout chưa dùng trong lựa chọn v11; dữ liệu nguồn đã xuất hiện trong các vòng nghiên cứu trước, không tuyên bố chưa từng được sử dụng lịch sử.','',
 '## Kết quả được chọn bằng validation','',
 '| Dataset | Backend | Accuracy (%) | Macro-F1 (%) | Recall attack (%) | FPR (%) |',
 '|---|---|---:|---:|---:|---:|']
 for r in rows:lines.append(f"| {r['dataset']} | {r['device']} | {r['accuracy']:.2f} ± {r['accuracy_sd']:.2f} | {r['macro_f1']:.2f} ± {r['macro_f1_sd']:.2f} | {r['recall']:.2f} | {r['fpr']:.2f} |")
 lines +=['','## Ablation trên cùng cohort','', '| Dataset | Backend | Ideal-trained F1 | Noise-trained, threshold 0 F1 | Noise-trained, tuned threshold F1 |','|---|---|---:|---:|---:|']
 for ds in DATASETS:
  for dev in DEVICES:
   a=df[(df.dataset==ds)&(df.device==dev)&df.selected]
   vals=[a[(a.method==m)&(a.regime==reg)].macro_f1.mean() for m,reg in (('projected','ideal_trained'),('projected_zero_threshold','noise_trained'),('projected','noise_trained'))]
   lines.append(f'| {ds} | {dev} | '+ ' | '.join(f'{v:.2f}' for v in vals)+' |')
 lines +=['','## Đối chứng cổ điển','', '| Dataset | Model | Accuracy (%) | Macro-F1 (%) |','|---|---|---:|---:|']
 for (ds,m),g in df[df.device=='classical'].groupby(['dataset','method']):lines.append(f'| {ds} | {m} | {g.accuracy.mean():.2f} | {g.macro_f1.mean():.2f} |')
 lines +=['','## Chi phí và giới hạn','',
 'Các kết quả chỉ hỗ trợ so sánh pipeline trên phép chia này. PCA32+RBF và raw-preprocessing+ExtraTrees đều học cùng 2.000 nhãn. Product-state control không có entangling gate; nếu nó cạnh tranh tốt, chưa có bằng chứng lợi ích của entanglement. Không tuyên bố quantum advantage.',
 'Projected kernel dùng hai mạch/mẫu và lưu feature vectors; v10 fidelity dùng một overlap/mỗi support vector. Vì định nghĩa kernel khác, báo cả chất lượng và tổng shot/CX; không chỉ so sánh shots/mạch.',
 'Không triển khai ZNE: kéo dài các mạch sâu không được mặc định là cải thiện. Thử nghiệm readout mitigation và shots là ablation có chi phí rõ ràng.','',
 '## Artifacts','',
 '- `metrics.csv`: mọi cấu hình, ngưỡng 0/tuned, ideal-trained/noise-trained, supplement và đối chứng.',
 '- `all_candidates_summary.csv`, `headline.csv`, `paired_noise_training.csv`.',
 '- `frozen_selection.json`, `selection.sha256`, `model_manifest.json`: lựa chọn và model đóng băng trước test.',
 '- Mỗi dòng metric liên kết NPZ chứa labels, row IDs, scores, threshold và predictions.',
 '- `validation.json`: kiểm tra metrics, split, hash và số kết quả.',
 '','![V11 results]('+str((ROOT/'results.png').resolve())+')']
 Path('experiments/REPORT_VI.md').write_text('\n'.join(lines)+'\n')
 os.environ.setdefault('MPLCONFIGDIR','/tmp/icc-v11-mpl')
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(2,2,figsize=(12,8),sharey=True)
 for ax,ds in zip(axes.flat,DATASETS):
  a=tab[tab.dataset==ds].set_index('device').loc[list(DEVICES)]
  ax.errorbar(range(3),a.macro_f1,yerr=a.macro_f1_sd,fmt='o-',capsize=4,label='Noise-trained, validation-selected')
  old=[]
  for dev in DEVICES:old.append(df[(df.dataset==ds)&(df.device==dev)&df.selected&(df.method=='projected')&(df.regime=='ideal_trained')].macro_f1.mean())
  ax.plot(range(3),old,'s--',label='Same encoder, ideal-trained');ax.set_xticks(range(3),[x.title() for x in DEVICES]);ax.set_title(ds);ax.set_ylim(0,105);ax.grid(alpha=.2);ax.set_ylabel('Macro-F1 (%)')
 axes[0,0].legend(fontsize=8);fig.suptitle('New projected kernels: 1,024 total shots/sample; 1,000 test rows\nMean ± SD of four measurement/training seeds on one fixed split')
 fig.tight_layout();fig.savefig(ROOT/'results.png',dpi=180);fig.savefig(ROOT/'results.pdf');plt.close(fig)
 print(tab.to_string(index=False),flush=True)

if __name__=='__main__':run()
