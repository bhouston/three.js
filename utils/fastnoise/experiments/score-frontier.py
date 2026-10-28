"""Finite Pareto audit, not a state-of-the-art certification.
Usage: python score-frontier.py VOLUME_DIRECTORY OUTPUT_JSON [NAME ...]
Scores ten legacy metrics plus 22 integrands x raw/blur x seven frame/EMA results.
The original fourteen functions are retained, with additional held-out mid-grid
thresholds .2/.4/.6/.8, cos frequencies 2/8, and sin frequencies 3/7.
"""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image

spec = importlib.util.spec_from_file_location('score', Path(__file__).resolve().parents[1] / 'score-alternatives.py')
score = importlib.util.module_from_spec(spec)
spec.loader.exec_module(score)


def integrands(v):
    fs = {'identity': (v, .5), 'x_squared': (v*v, 1/3), 'x_fourth': (v**4, 1/5)}
    for t in (.1, .25, .5, .75, .9, .17, .37, .63, .83, .2, .4, .6, .8):
        fs[f'lt_{t:g}'] = ((v < t).astype(float), t)
    for k in (1, 2, 4, 8):
        fs[f'cos_{k}'] = (np.cos(2*np.pi*k*v), 0)
    for k in (3, 7):
        fs[f'sin_{k}'] = (np.sin(2*np.pi*k*v), 0)
    return fs


def fast_errors(v, truth):
    result = {}
    phases = np.arange(32)
    for mode, planes in [('raw', v), ('sigma1', score.gauss(v))]:
        sequence = np.array([score.fetch(planes, n) for n in range(159)])
        cumulative = np.concatenate([np.zeros((1, 128, 128)), sequence.cumsum(axis=0)])
        result[mode] = {str(n): score.errors((cumulative[phases+n]-cumulative[phases])/n, truth) for n in score.FRAMES}
        h = np.repeat(planes.mean(0)[None], 32, axis=0)
        final = []
        for n in range(128):
            h += .1*(sequence[n+phases]-h)
            if n >= 96:
                final.append(score.errors(h, truth))
        result[mode]['ema01_4cycles'] = {
            'rmse': float(np.sqrt(np.mean([x['rmse']**2 for x in final]))),
            'bias': float(np.mean([x['bias'] for x in final])),
            'bias_rms': float(np.sqrt(np.mean([x['bias_rms']**2 for x in final]))),
            'spatial_std_rms': float(np.sqrt(np.mean([x['spatial_std_rms']**2 for x in final])))}
    return result


def evaluate(v, digest=None):
    return {'input_sha256': digest, 'compare': score.compare(v), 'integrands': {name: fast_errors(f, truth) for name, (f, truth) in integrands(v).items()}}


def main():
    directory, output = Path(sys.argv[1]), Path(sys.argv[2])
    report = json.loads(output.read_text()) if output.exists() else {'protocol': {'frames': list(score.FRAMES), 'phases': 32, 'seeds': 'per filename', 'quantization': '8-bit centers', 'cycle_shift_yx': [-71, -37], 'note': 'One finite synthetic test suite. No renderer or general superiority claim.'}, 'methods': {}}
    if 'FAST-shipped' not in report['methods']:
        atlas = Path(__file__).resolve().parents[3] / 'examples/textures/noises/fast/gauss1-ema01-spatio-temporal.png'
        v = np.asarray(Image.open(atlas).convert('RGB')).reshape(32,128,256,3)[:,:,:128,2]
        report['methods']['FAST-shipped'] = evaluate((v.astype(float)+.5)/256, hashlib.sha256(atlas.read_bytes()).hexdigest())
        output.write_text(json.dumps(report, indent=2))
        print('scored FAST-shipped', flush=True)
    names = sys.argv[3:] or [p.stem for p in sorted(directory.glob('*.f32'))]
    for name in names:
        if name == 'FAST-shipped':
            continue
        file = directory / (name+'.f32')
        digest = hashlib.sha256(file.read_bytes()).hexdigest()
        if report['methods'].get(name, {}).get('input_sha256') == digest:
            continue
        raw = np.fromfile(file, np.float32).reshape(32,128,128)
        if not np.all(np.isfinite(raw) & (raw>=0) & (raw<1)):
            raise ValueError(f'Invalid volume {file}')
        v = (np.floor(raw*256).astype(float)+.5)/256
        report['methods'][name] = evaluate(v, digest)
        output.write_text(json.dumps(report, indent=2))
        print('scored',name,flush=True)


if __name__ == '__main__':
    main()
