import json,os
os.environ.setdefault('MPLCONFIGDIR','/tmp/icc-timing-mpl')
from pathlib import Path
import numpy as np,pandas as pd

ROOT=Path('results/timing')
def run():
 rows=[]
 for p in ROOT.glob('*.json'):
  a=json.loads(p.read_text())
  if 'method' not in a:continue
  for stage in ('train','predict'):
   r=a[stage];assert all(r[k]>=0 for k in ('wall_seconds','quantum_seconds','compile_seconds','classical_seconds'))
   assert r['wall_seconds']+1e-6>=r['quantum_seconds']+r['compile_seconds']
   if a['method']!='z7':assert r['circuits']==(400 if stage=='train' else 40) and r['shots']==r['circuits']*512
   else:assert r['shots']==r['circuits']*1024
   rows.append(dict(dataset=a['dataset'],device=a['device'],method=a['method'],repetition=a['repetition'],stage=stage,refs=a['refs'],per_input_seconds=r['wall_seconds']/(200 if stage=='train' else 20),**r))
 df=pd.DataFrame(rows);assert len(df)==288,len(df)
 assert not df.duplicated(['dataset','device','method','repetition','stage']).any()
 # Identical random seeds and workload must produce identical predictions per run.
 for keys,sub in df[df.stage=='predict'].groupby(['dataset','device','method']):
  assert len(sub)==3
  ds,dev,method=keys;scores=[json.loads((ROOT/f'{ds}_{dev}_{rep}_{method}.json').read_text())['score'] for rep in range(3)]
  assert all(np.allclose(scores[0],s,rtol=0,atol=1e-12) for s in scores[1:]),keys
 df.to_csv(ROOT/'timings.csv',index=False)
 g=df.groupby(['dataset','device','method','stage'])[['wall_seconds','quantum_seconds','compile_seconds','classical_seconds','per_input_seconds','circuits','shots','refs']].agg(['median','min','max']);g.to_csv(ROOT/'summary.csv')
 paired=[]
 med=df.groupby(['dataset','device','method','stage']).wall_seconds.median()
 for ds in df.dataset.unique():
  for dev in df.device.unique():
   for stage in ('train','predict'):
    for method in ('adaptive16','adaptive8'):
     value=med.loc[ds,dev,method,stage]
     paired.append(dict(dataset=ds,device=dev,stage=stage,method=method,dense_over_sparse=med.loc[ds,dev,'dense',stage]/value,z7_over_sparse=med.loc[ds,dev,'z7',stage]/value))
 ratios=pd.DataFrame(paired);ratios.to_csv(ROOT/'paired_speed_ratios.csv',index=False)
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,2,figsize=(12,4.7));datasets=['ToN-IoT','RT-IoT','NSL-KDD','UNSW-NB15'];x=np.arange(4)
 # Mean of per-backend medians, not a pooled minimum/best device.
 av=med.groupby(['dataset','method','stage']).mean()
 av.unstack('stage').to_csv(ROOT/'dataset_timing.csv')
 for i,(method,label) in enumerate([('dense','Dense'),('adaptive16','Sparse-16'),('adaptive8','Sparse-8'),('z7','Z7 Pegasos')]):
  for ax,stage in zip(axes,['train','predict']):ax.bar(x+(i-1.5)*.2,[av.loc[ds,method,stage] for ds in datasets],.2,label=label)
 for ax,title in zip(axes,['Train: 200 rows (seconds)','Predict: batch of 20 inputs (seconds)']):
  ax.set_xticks(x,datasets);ax.set_title(title);ax.set_axisbelow(True);ax.grid(axis='y',alpha=.2)
 axes[0].legend();fig.suptitle('Single-thread Aer density-matrix CPU benchmark | mean of 3 backend medians');fig.tight_layout();fig.savefig(ROOT/'timing.png',dpi=200);fig.savefig(ROOT/'timing.pdf');plt.close(fig)
 lines=['# Benchmark thời gian CPU simulator — v13','','Cùng 200 mẫu train phân tầng và 20 test inputs chọn ngẫu nhiên trong từng dataset; ba lần đo tuần tự với cùng seed, một luồng, batch tối đa 32. Nairobi/Lagos/Perth; Aer density_matrix chạy thực tế measurement/counts. Không dùng cache xác suất hoặc phép phân tích tensor riêng cho Z7. Frontend lấy từ v11 đã fit trên 2.000 train rows. Đây là workload timing nhỏ, không phải kết quả accuracy full-data.','','## Thời gian toàn pipeline','','Trung vị ba lần; train gồm tạo/ánh xạ mạch, chạy simulator và fit classifier. Predict gồm tạo query, simulator và scoring trên batch 20 mẫu. ms/mẫu là throughput batch, không phải latency một request. Không gồm preprocessing đã đóng băng, khởi tạo backend/warmup, tìm siêu tham số hay validation.','','| Dataset | Backend | Method | Train 200 (s) | Predict 20 (s) | Predict (ms/mẫu) | Shots/test input |','|---|---|---|---:|---:|---:|---:|']
 overview=['## Tóm tắt qua ba backend','','Lấy trung vị ba lần trên từng backend, sau đó trung bình đều Nairobi/Lagos/Perth. Mỗi ô là **train 200 mẫu / predict batch 20 mẫu**, đơn vị giây.','','| Dataset | Dense | Sparse-16 | Sparse-8 | Z7 Pegasos |','|---|---:|---:|---:|---:|']
 for ds in datasets:
  vals=[f'{av.loc[ds,m,"train"]:.2f} / {av.loc[ds,m,"predict"]:.2f}' for m in ('dense','adaptive16','adaptive8','z7')]
  overview.append('| '+ds+' | '+' | '.join(vals)+' |')
 overview+=['','Tỷ số trên từng cặp dataset/backend (không lấy tỷ số của các extrema không tương ứng):']
 for method in ('adaptive16','adaptive8'):
  for stage in ('train','predict'):
   r=ratios[(ratios.method==method)&(ratios.stage==stage)]
   overview.append(f'- {method}, {stage}: dense/sparse {r.dense_over_sparse.min():.2f}–{r.dense_over_sparse.max():.2f}×; Z7/sparse {r.z7_over_sparse.min():.2f}–{r.z7_over_sparse.max():.2f}×.')
 lines[4:4]=overview+['']
 for (ds,dev,method),sub in df.groupby(['dataset','device','method']):
  tr=sub[sub.stage=='train'];te=sub[sub.stage=='predict'];lines.append(f'| {ds} | {dev} | {method} | {tr.wall_seconds.median():.3f} | {te.wall_seconds.median():.3f} | {te.per_input_seconds.median()*1000:.2f} | {te.shots.median()/20:.0f} |')
 lines+=['','## Cách diễn giải','','- Sparse/dense cùng projected protocol: 2 mạch/input, mỗi mạch 512 shots, SVC cố định C=10/gamma=1/30. Z7 dùng Pegasos 100 bước C=1.000, mỗi overlap 1.024 shots; reference count phụ thuộc fit. Đây là native-budget pipeline comparison, không phải cùng kernel hay cùng số phép lượng tử.', '- Quantum seconds gồm simulator submission/execution và lấy counts. Compile seconds tách riêng. File timings.csv lưu từng lần đo và các thành phần; summary.csv có median/min/max.', '- Thời gian CPU simulator không thể quy đổi thành latency QPU. Không so tỷ số với số giây trong paper vì máy, workload và triển khai khác nhau. Không ghép các timing này với accuracy v12/v12b.', '- Predict tái sử dụng reference representations đã chuẩn bị khi train; không có kernel-result cache. Frontend, snapshot và random seed cố định giữa các lần timing.', '- Ba lần đo lặp là biến thiên thời gian cùng workload, không phải ba lần chia dữ liệu; chưa có CI mạnh về tốc độ triển khai thực tế.']
 lines+=['','Z7 prediction xử lý một query với các reference theo batch tối đa 32; projected gom các measurement circuits theo batch tối đa 32. Do đó overhead submit/batch là một phần của triển khai được đo. Có thể tối ưu thêm cross-query batching hoặc kernel caching cho Z7; không xem tỷ số này là giới hạn tốc độ của thuật toán.','','![Timing](../results/timing/timing.png)']
 Path('benchmarks/REPORT_VI.md').write_text('\n'.join(lines)+'\n')
 (ROOT/'validation.json').write_text(json.dumps(dict(status='PASS',timing_rows=len(df),jobs=144,repetitions=3,identical_predictions_across_repetitions=True),indent=2));(ROOT/'COMPLETE').write_text('complete\n')
 print(df.groupby(['method','stage'])[['wall_seconds','quantum_seconds','compile_seconds']].median().to_string())

if __name__=='__main__':run()
