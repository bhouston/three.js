# code-only noises vs FAST spatio-temporal: running average after N frames, raw and after a wrapped sigma=1 gaussian
import sys, numpy as np
from PIL import Image
def gauss(img):
    k = np.exp(-0.5*np.arange(-3,4)**2); k /= k.sum()
    for ax in (-1,-2): img = sum(w*np.roll(img, i-3, axis=ax) for i,w in enumerate(k))
    return img
y, x = np.mgrid[0:128, 0:128].astype(np.float64)
st = np.asarray(Image.open(sys.argv[1])).astype(np.float64).reshape(32,128,256,3)[:,:,:128,0]
def fast(n):
    c = n // 32
    return (np.roll(st[n % 32], (-(71*c) % 128, -(37*c) % 128), axis=(0,1)) + 0.5) / 256
ign = lambda n: (52.9829189 * ((0.06711056*(x + 5.588238*(n % 64)) + 0.00583715*(y + 5.588238*(n % 64))) % 1)) % 1
r2 = lambda n: (x*0.7548776662 + y*0.5698402910 + n*0.6180339887) % 1
rng = np.random.default_rng(0)
Ns = [1, 4, 32, 256, 1024]
print('frames            ', Ns)
for name, f in (('White', lambda n: rng.random((128,128))), ('FAST Spatio-Temp', fast), ('IGN (animated)', ign), ('R2 + golden', r2)):
    acc = np.zeros((128,128)); raw = []; blur = []
    for n in range(1, max(Ns)+1):
        acc += f(n-1)
        if n in Ns: raw.append(f'{(acc/n).std():.5f}'); blur.append(f'{gauss(acc/n).std():.5f}')
    print(f'{name:18s} raw  ', raw); print(f'{"":18s} blur ', blur)
