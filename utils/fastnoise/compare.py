# compare FAST volumes: label=path.f32 (our generator, float32 z,y,x) or label=ea:<png prefix> (EA slices _0.._31, channel R)
import sys, numpy as np
from PIL import Image
def load(spec):
    if spec.startswith('ea:'):
        v = np.array([np.asarray(Image.open(f'{spec[3:]}_{t}.png'))[..., 0] for t in range(32)]).astype(np.float64)
    else:
        v = np.floor(np.clip(np.fromfile(spec, np.float32).reshape(32, 128, 128), 0, 0.999999) * 256)
    return (v + 0.5) / 256
def gauss(img):
    k = np.exp(-0.5*np.arange(-3,4)**2); k /= k.sum()
    for ax in (-1,-2): img = sum(w*np.roll(img, i-3, axis=ax) for i,w in enumerate(k))
    return img
fy, fx = np.meshgrid(np.fft.fftfreq(128), np.fft.fftfreq(128), indexing='ij'); r = np.hypot(fx, fy)
print(f"{'':16s} {'tmpLow':>7s} {'spLow':>7s} {'pp32':>7s} {'pp256':>7s} {'bl1':>7s} {'bl4':>7s} {'bl32':>7s} {'bl256':>7s} {'ema':>7s} {'ema+bl':>7s}")
for arg in sys.argv[1:]:
    label, spec = arg.split('=', 1)
    v = load(spec)
    P = np.abs(np.fft.fft(v - 0.5, axis=0))**2; p = P.mean(axis=(1,2)); tlow = p[1:4].mean() / p[1:].mean()
    Ps = np.mean([np.abs(np.fft.fft2(v[t]-0.5))**2 for t in range(32)], 0); slow = Ps[(r<.1)&(r>0)].mean()/Ps[r>.3].mean()
    fetch = lambda n: np.roll(v[n % 32], (-(71*(n//32)) % 128, -(37*(n//32)) % 128), axis=(0,1))
    acc = np.zeros((128,128)); res = {}
    for n in range(1, 257):
        acc += fetch(n-1)
        if n in (1, 4, 32, 256): res[n] = ((acc/n).std(), gauss(acc/n).std())
    h = v.mean(0); out = []
    for it in range(4):
        for t in range(32):
            h = h + 0.1*(v[t]-h)
            if it == 3: out.append(h.copy())
    ema = np.mean([x.std() for x in out]); emab = np.mean([gauss(x).std() for x in out])
    print(f"{label:16s} {tlow:7.3f} {slow:7.4f} {res[32][0]:7.4f} {res[256][0]:7.4f} {res[1][1]:7.4f} {res[4][1]:7.4f} {res[32][1]:7.5f} {res[256][1]:7.5f} {ema:7.4f} {emab:7.4f}")
