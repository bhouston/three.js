"""CPU exploration of asset-free temporal schedules; no fitted weights.
Run with NumPy from repository root. Produces temporal-results.json beside this file.
"""
import importlib.util
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('comparison', HERE.parent / 'compare.py')
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)
U = np.uint32
FRAMES = [1, 2, 3, 4, 7, 8, 16, 31, 32, 64, 100, 128, 256, 512, 1024]

def reverse32(n):
    return int(f'{n:032b}'[::-1], 2)

def sample(phase, n, mode):
    if mode == 'complement-Weyl':
        v = ((phase + U(((n // 2) * 2654435769) & 0xffffffff)) >> U(8)).astype(np.float64) / 16777216
        return (1 - 1 / 16777216 - v) if n & 1 else v
    if mode.startswith('sign-'):
        level = int(mode.split('-')[1])
        signs = np.where(c.pixel_hash(c.X >> U(level), c.Y >> U(level), 71) & U(1), 1, -1)
        offset = (signs * ((n * 2654435769) & 0xffffffff)).astype(np.uint32)
        return ((phase + offset) >> U(8)).astype(np.float64) / 16777216
    if mode == '3LHW':
        offset = n * 2654435769
    elif mode == 'silver':
        offset = n * 1779033703
    elif mode == 'sqrt3':
        offset = n * 3144134277
    elif mode == 'vdc-add':
        offset = reverse32(n)
    elif mode == 'vdc-xor':
        return ((phase ^ U(reverse32(n))) >> U(8)).astype(np.float64) / 16777216
    elif mode == 'half-strata-Weyl':
        offset = (n // 2) * 2654435769 + (n & 1) * 2147483648
    elif mode == 'four-strata-Weyl':
        offset = (n // 4) * 2654435769 + (0, 2, 1, 3)[n & 3] * 1073741824
    return ((phase + U(offset & 0xffffffff)) >> U(8)).astype(np.float64) / 16777216

def evaluate(phase, mode, start):
    acc = np.zeros(c.X.shape)
    th = {p: np.zeros_like(acc) for p in (.1, .5, .9)}
    out = {}
    for i in range(max(FRAMES)):
        v = c.ign(i + start) if mode == 'IGN' else sample(phase, i + start, mode)
        acc += v
        for p in th:
            th[p] += v < p
        if i + 1 in FRAMES:
            avg = acc / (i + 1)
            out[i + 1] = {'raw': float(avg.std()), 'blur': float(c.gauss(avg).std()), 'threshold': {str(p): float(np.sqrt(np.mean((c.gauss(a / (i + 1)) - p)**2))) for p, a in th.items()}}
    return out

def main():
    modes = ['IGN', '3LHW', 'silver', 'sqrt3', 'vdc-add', 'vdc-xor', 'half-strata-Weyl', 'four-strata-Weyl', 'sign-0', 'sign-1', 'sign-3', 'complement-Weyl']
    result = {'frames': FRAMES, 'note': 'CPU uint32 / float64 accumulation; same original spatial phase; IGN original seed-1 offset. Seed 2 changes hierarchy only.', 'runs': []}
    for seed, start in [(1, 0), (2, 0), (1, 13), (2, 97)]:
        phase = c.hierarchy_phase(seed)
        row = {'seed': seed, 'start': start, 'methods': {}}
        for mode in modes:
            row['methods'][mode] = evaluate(phase, mode, start)
            print(seed, start, mode, flush=True)
        result['runs'].append(row)
    (HERE / 'temporal-results.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')

if __name__ == '__main__':
    main()
