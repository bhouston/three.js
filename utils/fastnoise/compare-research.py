"""Compare frozen scalar volumes from PRs 116/117 under one scoring protocol.
Usage: python utils/fastnoise/compare-research.py VOLUME_DIRECTORY OUTPUT_JSON
Uses score-alternatives.py; requires NumPy and Pillow. No rendering benchmark.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image

spec = importlib.util.spec_from_file_location('scorer', Path(__file__).with_name('score-alternatives.py'))
scorer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scorer)
directory = Path(sys.argv[1])
required = {'116-hierarchy', '116-lattice', '116-relaxation', '117-filter-rank', '117-hierarchy', '117-local-energy'}
if not required.issubset({p.stem for p in directory.glob('*.f32')}):
    raise ValueError('Expected all six 116/117 default seed-1 volumes.')
report = {'protocol': {
    'pr117_commit': '502a026ac59540f305d6f05fa2e8a53c1301815a',
    'seed': 1, 'size_depth': [128, 128, 32], 'phases': 32,
    'quantization': '8-bit bin centers', 'filter': 'wrapped sigma=1 radius=3',
    'ema': 'alpha .1, four cycles, average last cycle, all32 start phases',
    'cycle_shift_yx': [-71, -37], 'threshold_training_grid_117': '(c+.5)/15',
    'holdout_thresholds': [.17, .37, .63, .83],
    'note': 'Frozen-volume numerical diagnostics; no scene, performance, or unbiasedness claim.'
}, 'methods': {}}
volumes = {}
for file in sorted(directory.glob('*.f32')):
    raw = np.fromfile(file, np.float32).reshape(32, 128, 128)
    if not np.all(np.isfinite(raw) & (raw >= 0) & (raw < 1)):
        raise ValueError(f'Invalid volume {file}')
    volumes[file.stem] = ((np.floor(raw * 256).astype(float) + .5) / 256,
                          hashlib.sha256(file.read_bytes()).hexdigest())
atlas = Path(__file__).resolve().parents[2] / 'examples/textures/noises/fast/gauss1-ema01-spatio-temporal.png'
v = np.asarray(Image.open(atlas).convert('RGB')).reshape(32, 128, 256, 3)[:, :, :128, 2]
volumes['FAST-scalar'] = ((v.astype(float) + .5) / 256, hashlib.sha256(atlas.read_bytes()).hexdigest())
rng = np.random.default_rng(1)
volumes['White'] = ((np.floor(rng.random((32, 128, 128)) * 256) + .5) / 256, None)
for name, (v, digest) in volumes.items():
    result = {'input_sha256': digest, 'compare': scorer.compare(v), 'integrands': scorer.integrands(v)}
    result['integrands']['identity'] = scorer.score_integrand(v, .5)
    for threshold in report['protocol']['holdout_thresholds']:
        result['integrands'][f'holdout_lt_{threshold:g}'] = scorer.score_integrand((v < threshold).astype(float), threshold)
    for frequency in (1, 4):
        result['integrands'][f'cos_{frequency}'] = scorer.score_integrand(np.cos(2 * np.pi * frequency * v), 0)
    report['methods'][name] = result
    print(name, 'scalar bl32', result['compare']['bl32'], flush=True)
    Path(sys.argv[2]).write_text(json.dumps(report, indent=2), encoding='utf-8')
