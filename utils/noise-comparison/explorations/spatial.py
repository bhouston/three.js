"""Spatial phase ablations: no asset, no preprocessing; CPU reference only."""
import sys,json
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import compare as c
U=c.U

def phase(levels=3,mode='hash',seed=1,affine=False):
 x,y=c.X,c.Y
 if affine: x,y=x+2*y,2*x+y
 rank=np.zeros(x.shape,dtype=U)
 for level in range(levels):
  q=c.pixel_hash(x>>U(level+1),y>>U(level+1),seed^(((level+1)*0x632be5ab)&0xffffffff))
  if mode.startswith('fixed-'): q=c.hash32(U(seed^(((level+1)*0x632be5ab)&0xffffffff)))
  a,b=(x>>U(level))&U(1),(y>>U(level))&U(1)
  a,b=np.where(q&U(4),b,a),np.where(q&U(4),a,b)
  rank|=((((a^b)<<U(1))|b)^(q&U(3)))<<U(30-2*level)
 mode=mode.removeprefix('fixed-')
 if mode=='hash': jitter=c.pixel_hash(x>>U(levels),y>>U(levels),seed^0x02e5be93)
 elif mode=='zero': jitter=U(int(c.hash32(U(seed))))
 elif mode=='lattice': jitter=((x>>U(levels))*U(2882487921)+(y>>U(levels))*U(250703381))
 elif mode=='small': jitter=c.pixel_hash(x>>U(levels),y>>U(levels),seed^0x02e5be93)>>U(2*levels)
 return rank+jitter

def ternary_phase(levels=2,seed=1,fixed=False):
 rank=np.zeros(c.X.shape); x,y=c.X.copy(),c.Y.copy()
 for level in range(levels):
  q=c.pixel_hash(x//U(3),y//U(3),seed^(((level+1)*0x632be5ab)&0xffffffff))
  if fixed: q=c.hash32(U(seed^(((level+1)*0x632be5ab)&0xffffffff)))
  a,b=x%U(3),y%U(3)
  a,b=np.where(q&U(4),b,a),np.where(q&U(4),a,b)
  a=np.where(q&U(8),U(2)-a,a); b=np.where(q&U(16),U(2)-b,b)
  digit=(a+U(3)*b+q%U(9))%U(9)
  rank+=digit.astype(float)/(9**(level+1)); x//=U(3);y//=U(3)
 rank+=c.pixel_hash(x,y,seed^0x02e5be93).astype(float)/4294967296/(9**levels)
 rank+=int(c.hash32(U(seed)))/4294967296
 return (np.remainder(rank,1)*4294967296).astype(U)

def evaluate(p,starts=(0,),nonlinear=False):
 res={}
 for start in starts:
  acc=np.zeros(c.X.shape); th=np.zeros_like(acc); sq=np.zeros_like(acc); rows=[]
  for n in range(1,1025):
   v=((p+U(((n-1+start)*2654435769)&0xffffffff))>>U(8)).astype(float)/16777216
   acc+=v
   if nonlinear: th+=v<.37; sq+=v*v
   if n in c.FRAMES:
    row={'n':n,'blur':float(c.gauss(acc/n).std()),'uniformRMSE':float(np.sqrt(np.mean((c.gauss(acc/n)-.5)**2)))}
    if nonlinear: row.update(threshold=float(np.sqrt(np.mean((c.gauss(th/n)-.37)**2))),square=float(np.sqrt(np.mean((c.gauss(sq/n)-1/3)**2))))
    rows.append(row)
  res[str(start)]=rows
 return res

def main():
 out={'screen':{},'confirm':{}}
 candidates={}
 for levels in (1,2,3,4,5,6):
  for mode in ('hash','zero','lattice','small'):
   name=f'L{levels}-{mode}'; candidates[name]=lambda seed,l=levels,m=mode:phase(l,m,seed)
 for levels in (1,2,3,4):
  for mode in ('fixed-hash','fixed-zero','fixed-small','fixed-lattice'):
   candidates[f'L{levels}-{mode}']=lambda seed,l=levels,m=mode:phase(l,m,seed)
 for alpha,beta in ((.06711056*52.9829189%1,.00583715*52.9829189%1),(.7548776662466927,.5698402909980532),(.6180339887498949,.4142135623730951),(.5,.25),(.555,.309),(.556,.31),(.556,.306),(.55,.31),(.556,.3125),(.5625,.3125),(.5546875,.30859375)):
  candidates[f'lattice-{alpha:.8f}-{beta:.8f}']=lambda seed,a=alpha,b=beta:np.remainder((c.X*a+c.Y*b+int(c.hash32(U(seed)))/4294967296),1).__mul__(4294967296).astype(U)
 for levels in (2,3,4): candidates[f'L{levels}-affine']=lambda seed,l=levels:phase(l,'hash',seed,True)
 for levels in (1,2,3):
  for fixed in (False,True): candidates[f'ternary-{levels}-{fixed}']=lambda seed,l=levels,f=fixed:ternary_phase(l,seed,f)
 for seed in (1,):
  ignphase=(c.ign(0)*4294967296).astype(U)
  candidates['IGN-phase-Weyl']=lambda seed:ignphase+c.hash32(U(seed))
  for name,fn in candidates.items():
   out['screen'][name]=evaluate(fn(seed))
   print(name,[round(r['blur'],7) for r in out['screen'][name]['0']],flush=True)
 for name in ('L3-hash','L3-small','L3-zero','L2-fixed-small','L1-fixed-zero','IGN-phase-Weyl','lattice-0.55600000-0.31000000','lattice-0.75487767-0.56984029','ternary-2-False','ternary-2-True'):
  out['confirm'][name]={str(seed):evaluate(candidates[name](seed),starts=(0,17,113),nonlinear=True) for seed in (1,2)}
 Path(__file__).with_name('spatial-results.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
