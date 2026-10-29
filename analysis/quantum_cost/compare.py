"""Resource accounting from frozen experiments, not a new accuracy benchmark."""
import os,sys,json
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/icc-cost-mpl')
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.z7 import templates
from experiments.quantum import context
from src.data import sha256

OUT=Path(__file__).resolve().parent
BASE=Path('results/detection')
def run():
 m=pd.read_csv(BASE/'metrics.csv');m=m[(m.method=='projected')&(m.regime=='noise_trained')&(m.shots==1024)&(~m.mitigated)&m.config.isin(['dense','adaptive16','adaptive8'])].copy();assert len(m)==144
 # cx_per_sample in the legacy file sums two circuit templates, without shot weighting.
 m['cx_per_circuit']=m.cx_per_sample/2;m['cx_executions_per_input']=m.cx_per_sample*512
 g=m.groupby(['dataset','config','device'])[['cx_per_circuit','depth_per_circuit','cx_executions_per_input','macro_f1']].mean();g.to_csv(OUT/'per_backend.csv')
 means=g.groupby(['dataset','config']).mean();rows=[]
 for (ds,cfg),r in means.iterrows():
  d=means.loc[(ds,'dense')];rows.append(dict(dataset=ds,config=cfg,**r.to_dict(),cx_reduction_pct=100*(1-r.cx_per_circuit/d.cx_per_circuit),depth_reduction_pct=100*(1-r.depth_per_circuit/d.depth_per_circuit),total_shots=1024,circuits=2,qubits=5))
 agg=pd.DataFrame(rows);agg.to_csv(OUT/'summary.csv',index=False)
 z=pd.read_csv(BASE/'z7_metrics.csv');z=z[(z.regime=='noise_trained')&(z.threshold_mode=='tuned')];assert len(z)==48
 z.groupby('dataset')[['refs','shots_per_sample','macro_f1']].agg(['mean','min','max']).to_csv(OUT/'reproduced_z7_budget.csv')
 # Native-gate cost of the exact block-preserving overlap used by our Z7 baseline.
 zr=[]
 for dev in ('nairobi','lagos','perth'):
  c,inv,_=templates(dev);b,layout,*_=context(dev,7);pair=c.compose(inv);ends=np.zeros(7)
  for inst in pair.data:
   if inst.operation.name=='barrier':continue
   qs=[pair.find_bit(q).index for q in inst.qubits];length=b.properties().gate_length(inst.operation.name,[layout[j] for j in qs]);ends[qs]=max(ends[qs])+length
  assert pair.count_ops().get('cx',0)==0
  zr.append(dict(device=dev,qubits=7,cx=0,depth=pair.depth(),gate_only_us=max(ends)*1e6,gate_counts=dict(pair.count_ops())))
 (OUT/'reproduced_z7_circuit.json').write_text(json.dumps(zr,indent=2))
 duration=pd.read_csv(BASE/'gate_duration_profile.csv');duration=duration[duration.config.isin(['dense','adaptive16','adaptive8'])];duration.groupby(['dataset','config','device']).asap_gate_only_ns.mean().to_csv(OUT/'gate_duration_8_inputs.csv')
 fig,axes=plt.subplots(1,2,figsize=(11,4.4));datasets=['ToN-IoT','RT-IoT','NSL-KDD','UNSW-NB15'];xx=np.arange(4)
 for j,(cfg,label,color) in enumerate([('dense','Dense','#486A9B'),('adaptive16','Sparse-16','#19A08C'),('adaptive8','Sparse-8','#DC9A35')]):
  sub=agg[agg.config==cfg].set_index('dataset').loc[datasets]
  for ax,col in zip(axes,['cx_per_circuit','depth_per_circuit']):ax.bar(xx+(j-1)*.24,sub[col],.24,label=label,color=color)
 for ax,title in zip(axes,['CX gates per measurement circuit','Depth per measurement circuit']):
  ax.set_xticks(xx,datasets);ax.set_title(title);ax.set_ylim(bottom=0);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
 axes[0].legend();fig.suptitle('Mapped circuits: 1,000 test inputs per dataset, mean over 3 devices');fig.tight_layout();fig.savefig(OUT/'cost_comparison.png',dpi=200);fig.savefig(OUT/'cost_comparison.pdf');plt.close(fig)
 lines=['# Chi phí sparse quantum đối chiếu Cirillo et al. (2026)','','Nguồn paper: https://link.springer.com/article/10.1007/s42484-026-00379-4 (Sections 4.2, 5.1, 6).','','## Kết luận','','Sparse giảm chi phí cổng so với dense cùng pipeline, nhưng không ít CX hơn ZFeatureMap của paper (0 CX). Giảm số shots ở pipeline projected đến từ thay đổi cách tính kernel; không được quy toàn bộ cho sparsification. Chưa đo tốc độ QPU.','','## Mốc tham chiếu từ paper','','Paper xét ToN-IoT và NSL-KDD; không có kết quả trực tiếp cho RT-IoT/UNSW-NB15. Cấu hình Pegasos-QSVC nổi bật: ZFeatureMap, 7 qubit, r=2, 100 bước; 1.024 shots mỗi mạch. Section 6 báo thời gian CPU simulator: Pegasos train 18,41–85,36 s, predict 0,31–2,4 s/mẫu qua các cấu hình; VQC/COBYLA 1.773–5.041 s; HQNN có ví dụ 36.970 và 286.353 s. Đây không phải số đo QPU, cũng không phải toàn bộ thời gian dưới nhiễu hay riêng Z7. Không chia các số này cho runtime code hiện tại để tuyên bố speedup.','','## Đo lại chi phí sparse và dense cùng điều kiện','','Dùng v11, mỗi dataset 1.000 test inputs, ba backend, bốn measurement/training seeds. Cổng và depth lấy từ mạch đã ánh xạ trong cache; số cổng không phụ thuộc seeds. Bảng lấy trung bình đều ba backend. Cùng frontend và split; không chọn lại mô hình. Không dùng một mẫu minh họa để suy rộng.','','| Dataset | Encoder | CX/mạch | Depth/mạch | Giảm CX vs dense | Macro-F1 (%) |','|---|---|---:|---:|---:|---:|']
 for r in rows:lines.append(f'| {r["dataset"]} | {r["config"]} | {r["cx_per_circuit"]:.2f} | {r["depth_per_circuit"]:.2f} | {r["cx_reduction_pct"]:.1f}% | {r["macro_f1"]:.2f} |')
 lines+=['','Mỗi input: 2 mạch (Z/X), 512 shots/mạch, tổng 1.024 shots. Tổng lần thực thi CX = 512 × (CX_Z + CX_X), không chỉ là tổng CX của hai template. Tất cả vẫn giữ 5 wires, kể cả khi số active preparation qubits nhỏ hơn. Macro-F1 trong bảng là ablation cùng cohort, không phải cấu hình headline được chọn cho từng seed.','','## Chi phí kernel / dự đoán','','Gọi S là số reference/support vectors thực sự được dùng:','','- Fidelity/Pegasos: thông thường S overlap circuits mới cho một input, mỗi mạch 1.024 shots ⇒ 1.024 S shots/input; có thể ít hơn nếu cache, đối xứng hoặc self-kernel cho phép bỏ truy vấn.', '- Projected (dense hoặc sparse): 2 circuits/input ×512 shots =1.024 shots/input; kernel RBF/SVC sau đó tính cổ điển. Số circuit templates giảm S/2 lần, tổng shots giảm S lần. Hai phương pháp dùng kernel khác nhau, không phải cùng bài toán đo lượng tử.', '- Ví dụ S=50: 50.000? Không: chính xác 51.200 shots so với 1.024; đây là ví dụ đại số, không phải số S công bố của paper.','','S đo được trong baseline Z7 của **chúng ta**, không phải số liệu tác giả paper:','','| Dataset | S trung bình (min–max) | Shots/input trung bình |','|---|---:|---:|']
 for ds,r in z.groupby('dataset'):lines.append(f'| {ds} | {r.refs.mean():.2f} ({r.refs.min()}–{r.refs.max()}) | {r.shots_per_sample.mean():.0f} |')
 lines+=['','Training projected: với N=2.000, trích xuất train một lần cần 4.000 circuit settings và 2.048.000 shots cho một encoder/backend/seed. Chưa gồm validation, test, calibration hay tìm kiếm nhiều encoder/seeds/backend. Huấn luyện SVC là cổ điển. Không khẳng định ít tổng chi phí training hơn Pegasos 100 bước: số kernel queries phụ thuộc quá trình update và cache; 100 bước không có nghĩa 100 mạch.', '', 'Fidelity sparse phiên bản cũ vẫn cần S overlap/input. Kết quả v10 không thuộc bộ kết quả hiện tại.','','## Thời gian và giới hạn','','gate_duration_8_inputs.csv chỉ là lịch ASAP dựa trên gate lengths, tám input cố định; không gồm readout/reset/queue/pulse alignment, không phải latency QPU. reproduced_z7_circuit.json dùng chính template block-preserving của baseline hiện tại; depth này do chúng ta tái dựng, không phải paper công bố. Cần cùng máy, phiên bản compiler, batch/thread count và workload để so wall-clock.','','Ước lượng gate-only trên tám input: dense trung bình 28,60 µs/mạch; sparse16 16,22 µs; sparse8 13,86 µs. Template overlap Z7 tái dựng chỉ 0,142 µs/shot (không readout/reset). Vì vậy ít shots không đồng nghĩa ít thời gian thực thi cổng; tổng latency còn phụ thuộc readout/reset và S. Không dùng những ước lượng này như speedup đo trên QPU.', '', '5 so với 7 qubit giảm 28,6% wires so với Z7, nhưng dense của chúng ta cũng đã dùng 5: đây là khác biệt encoding, không riêng sparse. Các VQC/HQNN của paper có cấu hình ít qubit hơn; không khẳng định giảm qubit so với mọi mô hình trong paper.','','Đánh đổi rõ nhất là UNSW: sparse rẻ hơn nhưng Macro-F1 thấp hơn dense trên cohort này. V12/v12b dùng frontend khác và fusion, nên không ghép Accuracy mới với cost sparse v11.','','![Cost](cost_comparison.png)']
 lines=[s.replace('50.000? Không: chính xác 51.200','51.200') for s in lines]
 (OUT/'REPORT_VI.md').write_text('\n'.join(lines)+'\n')
 sources=[BASE/'metrics.csv',BASE/'z7_metrics.csv',BASE/'gate_duration_profile.csv']
 (OUT/'provenance.json').write_text(json.dumps({str(p):sha256(p) for p in sources},indent=2));print(agg.to_string(index=False));print(z.groupby('dataset').refs.agg(['mean','min','max']).to_string())

if __name__=='__main__':run()
