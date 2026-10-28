"""Reproduce PR 35244's NumPy benchmark and add two image-free samplers.
Run from the repository root: python utils/noise-comparison/compare.py
Requires NumPy and Pillow. CPU reference arithmetic, not GPU timing.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
FRAMES = [2**i for i in range(11)]
Y, X = np.indices((128, 128), dtype=np.uint32)
U = np.uint32
F = np.float32

def hash32(v):
    with np.errstate(over='ignore'):
        v = (v ^ (v >> U(16))) * U(0x21f0aaad)
        v = (v ^ (v >> U(15))) * U(0x735a2d97)
        return v ^ (v >> U(15))

def pixel_hash(x, y, seed):
    return hash32(x * U(0x9e3779b1) ^ y * U(0x85ebca77) ^ U(seed))

def hierarchy_phase(seed=1):
    rank = np.zeros(X.shape, dtype=np.uint32)
    for level in range(3):
        q = pixel_hash(X >> U(level + 1), Y >> U(level + 1), seed ^ (((level + 1) * 0x632be5ab) & 0xffffffff))
        a, b = (X >> U(level)) & U(1), (Y >> U(level)) & U(1)
        a, b = np.where(q & U(4), b, a), np.where(q & U(4), a, b)
        rank |= ((((a ^ b) << U(1)) | b) ^ (q & U(3))) << U(30 - 2 * level)
    return rank + pixel_hash(X >> U(3), Y >> U(3), seed ^ 0x02e5be93)

PHASE = hierarchy_phase()

def hierarchy(frame):
    return ((PHASE + U((frame * 2654435769) & 0xffffffff)) >> U(8)).astype(np.float64) / 16777216

def ign(frame):
    shift = F(F(frame) * F(5.588238))
    offset = F((int(hash32(U(1))) >> 8) / 16777216 * 256)
    px, py = F(F(X.astype(F) + shift) + offset), F(F(Y.astype(F) + shift) + offset)
    v = F(F(F(.06711056) * px) + F(F(.00583715) * py))
    v = F(F(52.9829189) * F(v - np.floor(v)))
    return (v - np.floor(v)).astype(np.float64)

def gauss(img):
    k = np.exp(-.5 * np.arange(-3, 4)**2)
    k /= k.sum()
    for ax in (-1, -2):
        img = sum(w * np.roll(img, i - 3, axis=ax) for i, w in enumerate(k))
    return img

def texture(filename):
    path = ROOT / 'examples/textures/noises/fast' / filename
    volume = np.asarray(Image.open(path)).astype(np.float64).reshape(32, 128, 256, 3)[:, :, :128, 0]
    return volume, {'file': str(path.relative_to(ROOT)), 'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}

def fast(v, frame):
    c = frame // 32
    return (np.roll(v[frame % 32], (-(71*c) % 128, -(37*c) % 128), axis=(0, 1)) + .5) / 256

def main():
    prod, prod_info = texture('gauss1-ema01-spatial.png')
    sep, sep_info = texture('gauss1-ema01-spatio-temporal.png')
    rng = np.random.default_rng(0)
    base = (prod[0] + .5) / 256
    samplers = {'White': lambda n: rng.random((128, 128)), 'FAST Spatial (Product)': lambda n: fast(prod, n), 'FAST Spatio-Temporal (Separate)': lambda n: fast(sep, n), 'Golden Ratio': lambda n: (base + ((n * 2654435769) % 2**32) / 2**32) % 1, '3LHW': hierarchy, 'Animated IGN': ign}
    result = {'frames': FRAMES, 'raw': {}, 'blur': {}, 'threshold': {}, 'spectral': {}, 'metadata': {'baseline': 'mrdoob/three.js#35244 f14867640c96eb324665cb68948fccb12318231e', 'resolution': [128, 128], 'channel': 'R; x < 128; midpoint decode', 'proceduralSeed': 1, 'whiteSeed': 0, 'precision': 'FAST and accumulation float64; IGN float32 operations; 3LHW uint32, upper 24 bits', 'textures': [prod_info, sep_info]}}
    for name, sampler in samplers.items():
        acc = np.zeros((128, 128))
        thresholds = {p: np.zeros_like(acc) for p in (.1, .5, .9)}
        first = []
        result['raw'][name], result['blur'][name], result['threshold'][name] = [], [], {}
        for frame in range(1024):
            value = sampler(frame)
            if frame < 32:
                first.append(value)
            acc += value
            for p in thresholds:
                thresholds[p] += value < p
            n = frame + 1
            if n in FRAMES:
                result['raw'][name].append(float((acc / n).std()))
                result['blur'][name].append(float(gauss(acc / n).std()))
            if n in (1, 32, 256, 1024):
                result['threshold'][name][n] = {str(p): float(np.sqrt(np.mean((gauss(a / n) - p)**2))) for p, a in thresholds.items()}
        v = np.array(first)
        fy, fx = np.meshgrid(np.fft.fftfreq(128), np.fft.fftfreq(128), indexing='ij')
        r = np.hypot(fx, fy)
        power = (abs(np.fft.fft2(v - .5, axes=(-2, -1)))**2).mean(axis=0)
        temporal = (abs(np.fft.fft(v - .5, axis=0))**2).mean(axis=(1, 2))
        result['spectral'][name] = {'spatialLowHigh': float(power[(r > 0) & (r < .1)].mean() / power[r > .3].mean()), 'temporalLowAll': float(temporal[1:4].mean() / temporal[1:].mean())}
    result['referenceSamples'] = [{'x': x, 'y': y, 'frame': frame, 'hierarchy': float(hierarchy(frame)[y, x]), 'ign': float(ign(frame)[y, x])} for frame in (0, 1, 31, 32, 255, 1023) for x, y in ((0, 0), (3, 7), (12, 98), (127, 127))]
    (OUT / 'results.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    names = list(samplers)
    lines = ['# Reproduced comparison', '', 'Uniform field: standard deviation after arithmetic running average and wrapped sigma=1 Gaussian. Lower is better. Full precision is in results.json.', '', '| Frames | ' + ' | '.join(names) + ' |', '|---|' + '---|' * len(names)]
    for i, n in enumerate(FRAMES):
        lines.append('| ' + str(n) + ' | ' + ' | '.join(f"{result['blur'][name][i]:.6g}" for name in names) + ' |')
    lines += ['', '## Nonlinear check', '', 'RMSE against constant probability after thresholding each sample, averaging 256 frames, and applying the same Gaussian. Unlike standard deviation, RMSE includes bias.', '', '| Method | p=0.1 | p=0.5 | p=0.9 |', '|---|---|---|---|']
    for name in names:
        lines.append('| ' + name + ' | ' + ' | '.join(f"{result['threshold'][name][256][str(p)]:.6g}" for p in (.1, .5, .9)) + ' |')
    lines += ['', '## First 32 frames: spectrum diagnostics', '', 'Ratios of mean power: spatial frequencies 0<r<0.1 / r>0.3; temporal bins 1–3 / all non-DC bins. Lower indicates more suppression in those bands, not universally better rendering.', '', '| Method | Spatial low/high | Temporal low/all |', '|---|---|---|']
    for name in names:
        lines.append(f"| {name} | {result['spectral'][name]['spatialLowHigh']:.6g} | {result['spectral'][name]['temporalLowAll']:.6g} |")
    (OUT / 'TABLES.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('Wrote results.json and TABLES.md')

if __name__ == '__main__':
    main()
