"""True runtime streams, not frozen 32-frame procedural volumes. NumPy only."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

directory = Path(sys.argv[1])
output = Path(sys.argv[2])
phases = np.arange(32)
frames = (1, 4, 32, 128)


def gaussian(a):
    k = np.exp(-.5 * np.arange(-3, 4)**2)
    k /= k.sum()
    for axis in (-1, -2):
        a = sum(w * np.roll(a, i-3, axis) for i, w in enumerate(k))
    return a


def integration(v, truth):
    result = {}
    for label, values in [('raw', v), ('blur', gaussian(v))]:
        target = gaussian(truth) if label == 'blur' and np.ndim(truth) else truth
        sums = np.concatenate((np.zeros((1, 128, 128)), values.cumsum(0)))
        for n in frames:
            errors = (sums[phases+n]-sums[phases])/n-target
            result[f'{label}/{n}'] = float(np.sqrt(np.mean(errors**2)))
        # One IIR prefix plus a correction for each phase's initialization is
        # algebraically equivalent to 32 independent recurrences.
        history = np.empty_like(values)
        h = np.zeros((128, 128))
        for n in range(len(values)):
            h = .9*h + .1*values[n]
            history[n] = h
        before = np.concatenate((np.zeros((1, 128, 128)), history[:31]))
        correction = values[:32].mean(0)[None]-before
        mse = []
        for n in range(96, 128):
            h = history[n+phases] + .9**(n+1)*correction
            mse.append(float(np.mean((h-target)**2)))
        result[f'{label}/ema'] = float(np.sqrt(np.mean(mse)))
    return result


report = {'protocol': 'runtime-v1', 'definitions': {
    'dimensions': '159 actual frames at 128x128, float32 shader/model output',
    'phases': 'All 32 starting phases; box means 1,4,32,128; EMA alpha .1 burn96 measure32',
    'metric': 'RMSE against analytic integral including bias, raw and wrapped sigma1 Gaussian radius3',
    'reference': 'Shipped FAST scalar B, midpoint decode; phase shifts (37,71) each32frames; same asset for both seed controls',
    'procedural': 'Continuous real frame index, no forced32frame repetition or quantization',
    'caution': 'Finite development suite, not rendering/motion/vector evidence. CPU floating math may differ from GPU; prefer exported actual GPU streams.'
}, 'methods': {}}
if output.exists():
    report = json.loads(output.read_text())
for file in sorted(directory.glob('*.f32')):
    if '-seed' not in file.stem:
        continue
    digest = hashlib.sha256(file.read_bytes()).hexdigest()
    prior = report['methods'].get(file.stem, {})
    if prior.get('sha256') == digest and 'image_metrics' in prior:
        continue
    v = np.fromfile(file, np.float32).reshape(159, 128, 128).astype(float)
    assert np.isfinite(v).all() and (v >= 0).all() and (v < 1).all(), file
    y, x = np.mgrid[:128, :128] / 128 - .5
    radius = .38
    z = radius**2-x*x-y*y
    normal = np.array([x/radius, -y/radius, np.sqrt(np.maximum(0, z))/radius])
    light = np.array([-.5, .6, .6])
    light /= np.linalg.norm(light)
    shade = np.maximum(0, (normal*light[:, None, None]).sum(0))*.9+.03
    sphere = np.where(z > 0, shade, (x+.5)*.5)
    image_metrics = {}
    for name, truth in [('ramp', np.broadcast_to((x+.5), (128, 128))), ('sphere', sphere)]:
        image_metrics.update({f'{name}/{k}': value for k, value in integration((v < truth).astype(float), truth).items()})
    if prior.get('sha256') == digest:
        prior['image_metrics'] = image_metrics
        output.write_text(json.dumps(report, indent=2)+'\n')
        continue
    functions = [('identity', v, .5), ('squared', v**2, 1/3),
                 ('fourth', v**4, 1/5), ('cosine', np.cos(2*np.pi*v), 0),
                 ('sine7', np.sin(14*np.pi*v), 0)]
    functions += [(f'threshold-{p}', (v < p).astype(float), p)
                  for p in (.1, .25, .5, .75, .9, .07, .37, .83, .94)]
    metrics = {}
    for name, values, truth in functions:
        metrics.update({f'{name}/{k}': value for k, value in integration(values, truth).items()})
    assert len(metrics) == 140
    # Spectral concentration exposes ordered spikes that a low-frequency ratio hides.
    field = (v[:32] < .5).astype(float)
    field -= field.mean(axis=(1, 2), keepdims=True)
    power = (np.abs(np.fft.fft2(field))**2).mean(0)
    spectral = {'max_bin_fraction': float(power.max()/power.sum())}
    report['methods'][file.stem] = {'sha256': digest, 'metrics': metrics, 'image_metrics': image_metrics, 'spectral': spectral}
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(file.stem, 'scored', flush=True)
output.write_text(json.dumps(report, indent=2)+'\n')
