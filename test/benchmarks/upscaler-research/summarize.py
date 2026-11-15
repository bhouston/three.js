from pathlib import Path
import re,json
static={}
for name in ['psnr-spatial.log','psnr-temporal.log','psnr-combined.log','psnr-spatial-mip.log','psnr-final.log','psnr-ablation.log','psnr-responsive.log','psnr-responsive-2x.log']:
 text=Path(name).read_text() if Path(name).exists() else ''
 for line in text.splitlines():
  m=re.match(r'^(\S+) @(\d+)x: ([\d.]+) / ([\d.]+) \| ([\d.]+) / ([\d.]+)',line)
  if m:
   mode,f,nfull,nhelmet,sfull,shelmet=m.groups()
   static[f'{mode}@{f}']={'native_helmet':float(nhelmet),'ssaa_helmet':float(shelmet),'log':name}
bench={}
for name in ['bench-screen.log','bench-final.log','bench-responsive.log']:
 text=Path(name).read_text() if Path(name).exists() else ''
 for mode,ms in re.findall(r'^round \d+ (\S+): ([\d.]+) ms',text,re.M):
  bench.setdefault(name,{}).setdefault(mode,[]).append(float(ms))
motion={}
for name in ['motion-final.log','motion-responsive.log']:
 p=Path(name)
 if p.exists():
  for mode,err in re.findall(r'^(\S+) @3x deterministic motion: ([\d.]+)',p.read_text(),re.M): motion[mode]=float(err)
Path('results.json').write_text(json.dumps({'static':static,'bench':bench,'motion':motion},indent=2)+'\n')
for mode in ['joint-bilateral','joint-bilateral-fused','joint-bilateral-mip-fused','taau','taau-mip-fast-fused','taau-mip-fused']:
 ss=static.get(mode+'@3',{}).get('ssaa_helmet','')
 nn=static.get(mode+'@3',{}).get('native_helmet','')
 bs=bench.get('bench-final.log',{}).get(mode,[])
 print(mode,'ssaa',ss,'native',nn,'ms',bs,'motion',motion.get(mode,''))
