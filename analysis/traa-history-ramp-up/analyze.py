# Same RMSE, temporal bias, and consecutive-frame RMS definitions as companion analysis/gpu/analyze_realtime.py.
from pathlib import Path
import json, numpy as np
root=Path(__file__).parent
raw=root/'raw'
runs=json.loads((raw/'runs.json').read_text())
rms=lambda a:float(np.sqrt(np.mean(a*a)))
def read(name):return np.fromfile(raw/(name+'.f32'),dtype=np.float32).astype(np.float64)
def spread(vals):return dict(mean=float(np.mean(vals)),min=min(vals),max=max(vals),runs=vals)
report={}
for fx in ['ssr','vxgi']:
 ref=read(fx+'_reference_mean')
 entry={'resolution':[640,480],'reference':'512-frame post-TRAA temporal mean following 512 warm-up frames; unchanged sampler, old weighting','signalRMS':rms(ref),'convergence':{},'steady':{},'throughput':{}}
 for variant in ['old','new']:
  entry['convergence'][variant]={}
  for f in [1,2,4,8,16,32,64]:
   files=[raw/f'{fx}_{variant}_{r}_f{f}.f32' for r in range(3)]
   if all(p.exists() for p in files):entry['convergence'][variant][f]=spread([rms(read(p.stem)-ref) for p in files])
  metrics={k:[] for k in ['rmse','bias','flicker','noise']}
  for r in runs:
   if r.get('kind')=='quality' and r['effect']==fx and r['variant']==variant:
    s=r['steady'];bias=rms(read(s['name']+'_mean')-ref)
    vals={'bias':bias,'rmse':float(np.sqrt(s['temporalVariance']+bias*bias)),'noise':float(np.sqrt(s['temporalVariance'])),'flicker':float(np.sqrt(s['flickerMSE']))}
    for k,v in vals.items():metrics[k].append(v)
  entry['steady'][variant]={k:spread(v) for k,v in metrics.items() if v}
 for variant in ['original','old','new']:
  vals=[r['throughput']['fps'] for r in runs if r['kind']=='throughput' and r['effect']==fx and r['variant']==variant]
  if vals:entry['throughput'][variant]=spread(vals)
 report[fx]=entry
(root/'metrics.json').write_text(json.dumps(report,indent=2))
for fx,e in report.items():
 print('\n',fx,'signal',e['signalRMS'])
 for f in [1,2,4,8,16,32,64]:
  if f in e['convergence']['new'] and f in e['convergence']['old']:
   a=e['convergence']['old'][f];b=e['convergence']['new'][f];print(f, 'old %.6f [%.6f, %.6f]'%(a['mean'],a['min'],a['max']), 'new %.6f [%.6f, %.6f]'%(b['mean'],b['min'],b['max']), 'change %.1f%%'%((b['mean']/a['mean']-1)*100))
 for v,s in e['steady'].items(): print(v,'steady', ' '.join('%s %.6f [%.6f, %.6f]'%(k,a['mean'],a['min'],a['max']) for k,a in s.items()))
 for v,a in e['throughput'].items():print(v,'fps',a)
