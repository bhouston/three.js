"""Frozen research suite. NumPy only; no rendering/performance measurement.
Usage: python research-score.py VOLUME_DIRECTORY OUTPUT_JSON
All lower-is-better error metrics are retained, including regressions. Cache by
input hash so subsequent independent generation profiles can be scored cheaply.
"""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

PROTOCOL = 'frontier-v1'
FRAMES = (1, 4, 32)
THRESHOLDS = (.1, .25, .5, .75, .9)
HELDOUT = (.07, .19, .37, .63, .83, .94)


def gaussian(a, sigma=1):
    radius = 3 if sigma == 1 else int(np.ceil(3 * sigma))
    k = np.exp(-.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    k /= k.sum()
    for axis in (-1, -2):
        a = sum(w * np.roll(a, i - radius, axis=axis) for i, w in enumerate(k))
    return a


def fetch(v, n):
    return np.roll(v[n % 32], (-71 * (n // 32), -37 * (n // 32)), axis=(0, 1))


def original(v):
    p = (np.abs(np.fft.fft(v - .5, axis=0)) ** 2).mean(axis=(1, 2))
    y, x = np.meshgrid(np.fft.fftfreq(128), np.fft.fftfreq(128), indexing='ij')
    r = np.hypot(x, y)
    power = (np.abs(np.fft.fft2(v - .5)) ** 2).mean(axis=0)
    out = {'tmpLow': float(p[1:4].mean() / p[1:].mean()),
           'spLow': float(power[(r < .1) & (r > 0)].mean() / power[r > .3].mean())}
    acc = np.zeros((128, 128))
    for n in range(1, 257):
        acc += fetch(v, n - 1)
        if n in (1, 4, 32, 256):
            out[f'bl{n}'] = float(gaussian(acc / n).std())
            if n in (32, 256):
                out[f'pp{n}'] = float((acc / n).std())
    h = v.mean(0)
    ema, emab = [], []
    for n in range(128):
        h += .1 * (v[n % 32] - h)
        if n >= 96:
            ema.append(float(h.std()))
            emab.append(float(gaussian(h).std()))
    out['ema'] = float(np.mean(ema))
    out['ema+bl'] = float(np.mean(emab))
    return out


def integration(v, truth):
    sequence = np.array([fetch(v, n) for n in range(159)])
    filtered = gaussian(sequence)
    sums = np.concatenate((np.zeros((1, 128, 128)), filtered.cumsum(axis=0)))
    phases = np.arange(32)
    metrics, bias = {}, {}
    for n in FRAMES:
        errors = (sums[phases + n] - sums[phases]) / n - truth
        metrics[str(n)] = float(np.sqrt(np.mean(errors ** 2)))
        bias[str(n)] = float(np.sqrt(np.mean(errors.mean(axis=(1, 2)) ** 2)))
    h = gaussian(np.repeat(v.mean(0)[None], 32, axis=0))
    mse = []
    for n in range(128):
        h += .1 * (filtered[n + phases] - h)
        if n >= 96:
            mse.append(float(np.mean((h - truth) ** 2)))
    metrics['ema'] = float(np.sqrt(np.mean(mse)))
    return metrics, bias


def evaluate(v):
    metrics = {f'original/{k}': val for k, val in original(v).items()}
    biases = {}
    functions = [('identity', v, .5), ('squared', v ** 2, 1 / 3),
                 ('fourth', v ** 4, 1 / 5), ('cosine', np.cos(2 * np.pi * v), 0)]
    functions += [(f'threshold-{p}', (v < p).astype(float), p) for p in THRESHOLDS + HELDOUT]
    threshold_volumes = []
    for name, values, truth in functions:
        result, bias = integration(values, truth)
        metrics.update({f'phase/{name}/{n}': val for n, val in result.items()})
        biases[name] = bias
        if name in [f'threshold-{p}' for p in THRESHOLDS]:
            threshold_volumes.append((values, truth))
    # Held-out reconstruction widths: aggregate thresholds, no parameter tuning.
    for sigma in (.5, 2):
        mse = {n: [] for n in (4, 32)}
        phase = np.arange(32)
        for values, truth in threshold_volumes:
            sequence = gaussian(np.array([fetch(values, n) for n in range(64)]), sigma)
            sums = np.concatenate((np.zeros((1, 128, 128)), sequence.cumsum(axis=0)))
            for n in mse:
                mse[n].append(float(np.mean(((sums[phase + n] - sums[phase]) / n - truth) ** 2)))
        for n in mse:
            metrics[f'heldout-sigma{sigma}/threshold-aggregate/{n}'] = float(np.sqrt(np.mean(mse[n])))
    return metrics, biases


output = Path(sys.argv[2])
report = json.loads(output.read_text()) if output.exists() else {'protocol': PROTOCOL, 'methods': {}}
assert report['protocol'] == PROTOCOL
report['definitions'] = {
    'reference': 'Unmodified FAST separate 10000 steps, same seed and scalar output',
    'primary': 'All 74 metrics: original 10; 15 integrands x (1,4,32-frame means and EMA); 4 held-out filter-width aggregates',
    'seeds': [1, 2], 'quantization': 'floor(x*256), then midpoint',
    'phase': 'RMSE against known integral, all 32 starting phases, cycle shift y/x=(-71,-37), wrapped sigma1 Gaussian radius3',
    'ema': 'alpha .1; four cycles from per-pixel volume mean; RMS over last cycle and all start phases; same shifted playback',
    'heldout_thresholds': HELDOUT, 'heldout_sigmas': [.5, 2],
    'dominance': 'Pointwise strictly lower in every primary metric for both seeds; no averaging away regressions. Empirical suite, not universal claim.',
    'bias': 'Phase-dependent spatial mean error RMS recorded separately, already included in primary RMSE',
}
for file in sorted(Path(sys.argv[1]).glob('*.f32')):
    if file.stem.startswith('control-'):
        continue  # Byte-identity controls are validated separately, not candidates.
    digest = hashlib.sha256(file.read_bytes()).hexdigest()
    if report['methods'].get(file.stem, {}).get('sha256') == digest:
        continue
    identical = next((row for row in report['methods'].values() if row['sha256'] == digest), None)
    if identical is not None:
        report['methods'][file.stem] = identical.copy()
        output.write_text(json.dumps(report, indent=2) + '\n')
        print(file.stem, 'reused identical-volume scores', flush=True)
        continue
    raw = np.fromfile(file, np.float32).reshape(32, 128, 128)
    assert np.isfinite(raw).all() and (raw >= 0).all() and (raw < 1).all(), file
    volume = (np.floor(raw.astype(np.float64) * 256) + .5) / 256
    metrics, biases = evaluate(volume)
    assert len(metrics) == 74
    report['methods'][file.stem] = {'sha256': digest, 'metrics': metrics, 'bias_diagnostics': biases}
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(file.stem, 'scored', len(metrics), flush=True)
comparisons = {}
for name, row in report['methods'].items():
    if name.startswith('fast-'):
        continue
    seed = name.rsplit('-seed', 1)[-1]
    baseline = report['methods'].get('fast-seed' + seed)
    if baseline is None:
        continue
    ratios = {k: val / baseline['metrics'][k] for k, val in row['metrics'].items()}
    comparisons[name] = {
        'better': sum(v < 1 for v in ratios.values()), 'equal': sum(v == 1 for v in ratios.values()),
        'worse': sum(v > 1 for v in ratios.values()), 'all_strictly_better': all(v < 1 for v in ratios.values()),
        'worst_ratio': max(ratios.values()), 'worst_metric': max(ratios, key=ratios.get), 'ratios': ratios,
    }
report['comparisons'] = comparisons
output.write_text(json.dumps(report, indent=2) + '\n')
for name, row in comparisons.items():
    print(name, row['better'], '/', len(row['ratios']), 'improved; worst', row['worst_ratio'], row['worst_metric'], flush=True)
