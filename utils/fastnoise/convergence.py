# running average error at N frames, unblurred and after a wrapped sigma=1 gaussian
import sys, json, numpy as np
from pathlib import Path
from PIL import Image
load = lambda f: np.asarray(Image.open(f)).astype(np.float64).reshape(32,128,256,3)[:,:,:128,0]
prod, sep = load(sys.argv[1]), load(sys.argv[2])
def gauss(img):
    k = np.exp(-0.5*np.arange(-3,4)**2); k /= k.sum()
    for ax in (-1,-2): img = sum(w*np.roll(img, i-3, axis=ax) for i,w in enumerate(k))
    return img
def fast(vol, n):
    c = n // 32
    return (np.roll(vol[n % 32], (-(71*c) % 128, -(37*c) % 128), axis=(0,1)) + 0.5) / 256
base = (prod[0] + 0.5) / 256
golden = lambda n: (base + ((n * 2654435769) % 2**32) / 2**32) % 1
rng = np.random.default_rng(0)
Ns = [2**i for i in range(11)]
series = [('White', lambda n: rng.random((128,128))), ('FAST Spatial', lambda n: fast(prod, n)), ('FAST Spatio-Temporal', lambda n: fast(sep, n)), ('Golden Ratio', golden)]
if len(sys.argv) > 3:
    Ns = [1, 2, 4, 8, 16, 32, 128, 256, 512, 1024]
    for name, file in [('Hierarchy', 'hierarchy-1.f32'), ('Lattice', 'lattice-1.f32'), ('Relaxation', 'relaxation-1.f32')]:
        v = np.fromfile(Path(sys.argv[3]) / file, np.float32).reshape(32,128,128)
        quantized = np.floor(np.clip(v, 0, .999999) * 256).astype(np.float64)
        series.append((name, lambda n, vol=quantized: fast(vol, n)))
out = {'frames': Ns, 'raw': {}, 'blur': {}}
out['protocol'] = 'Original PR chart: atlas left R channel; fresh white RNG seed 0; golden ratio rotates first spatial slice; frozen volumes shift (-71,-37) per 32-frame cycle. Alternatives use seed 1, 8-bit bin centers. Phase-zero scalar-value standard deviation, not threshold-integrand RMSE.'
for name, f in series:
    acc = np.zeros((128,128)); out['raw'][name] = []; out['blur'][name] = []
    for n in range(1, max(Ns) + 1):
        acc += f(n - 1)
        if n in Ns:
            out['raw'][name].append(float((acc / n).std())); out['blur'][name].append(float(gauss(acc / n).std()))
print(json.dumps(out, indent=1))
