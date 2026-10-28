"""No-asset IGN phase/Weyl screen. CPU reference, not GPU timings."""
import importlib.util
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('comparison', HERE.parent / 'compare.py')
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)
FRAMES = [1, 2, 3, 4, 7, 8, 13, 16, 23, 32, 47, 64, 95, 128, 191, 256, 383, 512, 767, 1024]

def samplers(seed=1, phase=0):
    # Random coordinate offset; IGN's noninteger outer multiplier makes its
    # marginal only approximately uniform, as in the original IGN reference.
    cp = (int(c.hash32(c.U(seed))) >> 8) / 16777216
    x, y = c.X.astype(np.float64), c.Y.astype(np.float64)
    inner = (.06711056 * x + .00583715 * y + cp) % 1
    ign_phase = (52.9829189 * inner) % 1
    # Integer approximation to the same irrational lattice avoids growing coordinates.
    ip = ((c.X * c.U(round(.06711056 * 2**32)) + c.Y * c.U(round(.00583715 * 2**32))) >> c.U(8)).astype(np.float64) / 16777216
    fixed = (52.9829189 * ((ip + cp) % 1)) % 1
    hp = c.hierarchy_phase(seed)
    def ign(n):
        shift = c.F(c.F(n) * c.F(5.588238))
        offset = c.F(cp * 256)
        px = c.F(c.F(c.X.astype(c.F) + shift) + offset)
        py = c.F(c.F(c.Y.astype(c.F) + shift) + offset)
        v = c.F(c.F(c.F(.06711056) * px) + c.F(c.F(.00583715) * py))
        v = c.F(c.F(52.9829189) * c.F(v - np.floor(v)))
        return (v - np.floor(v)).astype(np.float64)
    def h(n):
        return ((hp + c.U(((n + phase) * 2654435769) & 0xffffffff)) >> c.U(8)).astype(np.float64) / 16777216
    def step(n):
        return (((n + phase) * 2654435769) & 0xffffffff) / 2**32
    methods = {
        '3LHW': h,
        'Animated IGN': lambda n: ign(n + phase),
        'IGN phase + Weyl': lambda n: (ign_phase + step(n)) % 1,
        'Fixed IGN phase + Weyl': lambda n: (fixed + step(n)) % 1,
        'IGN inner + Weyl': lambda n: (52.9829189 * ((inner + step(n)) % 1)) % 1,
    }
    for block in (4, 8, 16, 32):
        methods[f'IGN block {block} + Weyl'] = lambda n, b=block: (ign((n + phase) % b) + (((((n + phase) // b) * 2654435769) & 0xffffffff) / 2**32)) % 1
    def reverse(n):
        n = ((n & 0x55555555) << 1) | ((n >> 1) & 0x55555555)
        n = ((n & 0x33333333) << 2) | ((n >> 2) & 0x33333333)
        n = ((n & 0x0f0f0f0f) << 4) | ((n >> 4) & 0x0f0f0f0f)
        n = ((n & 0x00ff00ff) << 8) | ((n >> 8) & 0x00ff00ff)
        return ((n << 16) | (n >> 16)) & 0xffffffff
    methods['IGN phase + radical inverse'] = lambda n: (ign_phase + reverse(n+phase)/2**32) % 1
    for gamma in (.5, .25, (5**.5-1)/2, 2**.5-1, 1/3):
        methods[f'Animated IGN + outer {gamma:.9g}'] = lambda n, g=gamma: (ign(n+phase) + ((n+phase)*g)%1) % 1
    return methods

def score(seed, phase, full=False):
    result = {}
    for name, sample in samplers(seed, phase).items():
        acc = np.zeros((128,128))
        sums = {p: np.zeros_like(acc) for p in (.137, .371, .5, .829)}
        nonlinear = np.zeros_like(acc)
        rows=[]
        for n in range(1024):
            v = sample(n)
            acc += v
            if full:
                for p in sums: sums[p] += v < p
                nonlinear += v*v
            if n+1 in FRAMES:
                row={'frames':n+1,'raw':float((acc/(n+1)).std()),'blur':float(c.gauss(acc/(n+1)).std())}
                if full:
                    row['threshold']={str(p):float(np.sqrt(np.mean((c.gauss(s/(n+1))-p)**2))) for p,s in sums.items()}
                    row['square']=float(np.sqrt(np.mean((c.gauss(nonlinear/(n+1))-1/3)**2)))
                rows.append(row)
        result[name]=rows
    return result

if __name__ == '__main__':
    results={'frames':FRAMES,'screen':score(1,0,True),'holdout':{'seed2-phase17':score(2,17,True),'seed7-phase113':score(7,113,True)}}
    (HERE/'ign-phase-results.json').write_text(json.dumps(results,indent=2)+'\n')
    for name,rows in results['screen'].items():
        print(name, 'blur', ' '.join(f'{r["blur"]:.3g}' for r in rows))
