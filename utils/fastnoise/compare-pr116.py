"""Common scalar-integrand diagnostics for PRs #116 and #117; NumPy only.

Usage: python utils/fastnoise/compare-pr116.py VOLUME_DIRECTORY OUTPUT_JSON
Frozen seed-1 volumes; all 32 starting phases; no rendering or timing claims.
Conventions follow PR #116's score-alternatives.py, with both atlas channels
explicitly separated. Also test identity and a nonlinear periodic integrand.
"""
import json
import hashlib
import sys
from pathlib import Path
import numpy as np


def blur(image):
    kernel = np.exp(-0.5 * np.arange(-3, 4) ** 2)
    kernel /= kernel.sum()
    for axis in (-1, -2):
        image = sum(w * np.roll(image, k - 3, axis=axis) for k, w in enumerate(kernel))
    return image


def score(volume, function, truth):
    transformed = function(volume)
    timeline = np.array([
        np.roll(transformed[n % 32], (-71 * (n // 32), -37 * (n // 32)), axis=(0, 1))
        for n in range(64)
    ])
    result = {}
    phase = np.arange(32)
    for label, values in (('raw', timeline), ('sigma1', blur(timeline))):
        cumulative = np.concatenate((np.zeros((1, 128, 128)), values.cumsum(axis=0)))
        result[label] = {}
        for frames in (1, 4, 32):
            estimates = (cumulative[phase + frames] - cumulative[phase]) / frames
            error = estimates - truth
            result[label][str(frames)] = {
                'rmse': float(np.sqrt(np.mean(error ** 2))),
                'mean_bias': float(error.mean()),
                'phase_bias_rms': float(np.sqrt(np.mean(error.mean(axis=(1, 2)) ** 2))),
            }
    return result


report = {
    'protocol': {
        'reference_pr116_commit': '54a6dfe336cd33fc1fe129e4e1e6b10900395832',
        'seed': 1, 'shape_zyx': [32, 128, 128], 'frames': [1, 4, 32],
        'phases': list(range(32)), 'shift_yx_per_cycle': [-71, -37],
        'quantization': '8-bit midpoint', 'spatial_filter': 'wrapped sigma=1 radius=3 Gaussian',
        'metric': 'RMSE against analytic continuous integral, averaged over pixels and all 32 phases',
        'note': 'Numerical integrands, not rendered scenes. Common protocol, no new timing comparison.',
    },
    'methods': {},
}
for file in sorted(Path(sys.argv[1]).glob('*.f32')):
    raw = np.fromfile(file, np.float32).reshape(32, 128, 128)
    assert np.isfinite(raw).all() and (raw >= 0).all() and (raw < 1).all()
    volume = (np.floor(raw.astype(np.float64) * 256) + 0.5) / 256
    row = {'sha256_f32': hashlib.sha256(file.read_bytes()).hexdigest(),
           'identity': score(volume, lambda x: x, 0.5)}
    for threshold in (0.1, 0.25, 0.5, 0.75, 0.9):
        row[f'threshold_{threshold}'] = score(volume, lambda x: x < threshold, threshold)
    row['squared'] = score(volume, lambda x: x ** 2, 1 / 3)
    row['fourth'] = score(volume, lambda x: x ** 4, 1 / 5)
    row['cosine'] = score(volume, lambda x: np.cos(2 * np.pi * x), 0)
    row['threshold_aggregate_sigma1'] = {
        str(n): float(np.sqrt(np.mean([
            row[f'threshold_{p}']['sigma1'][str(n)]['rmse'] ** 2
            for p in (0.1, 0.25, 0.5, 0.75, 0.9)
        ]))) for n in (1, 4, 32)
    }
    report['methods'][file.stem] = row
    print(file.stem, 'identity32', row['identity']['sigma1']['32']['rmse'],
          'threshold32', row['threshold_aggregate_sigma1']['32'],
          'squared32', row['squared']['sigma1']['32']['rmse'],
          'fourth32', row['fourth']['sigma1']['32']['rmse'], flush=True)
Path(sys.argv[2]).write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
