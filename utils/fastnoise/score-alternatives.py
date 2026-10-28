"""Frozen-volume numerical diagnostics; no claim of rendered-scene validation.
Usage: python utils/fastnoise/score-alternatives.py OUTPUT_DIRECTORY
Requires NumPy and Pillow. All data is quantized to 8-bit bin centers like compare.py.
"""
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

SIZE, DEPTH = 128, 32
FRAMES = (1, 4, 8, 16, 32, 64)


def gauss(img):
    k = np.exp(-0.5 * np.arange(-3, 4) ** 2)
    k /= k.sum()
    for ax in (-1, -2):
        img = sum(w * np.roll(img, i - 3, axis=ax) for i, w in enumerate(k))
    return img


def fetch(v, n):
    return np.roll(v[n % 32], (-(71 * (n // 32)) % 128, -(37 * (n // 32)) % 128), axis=(0, 1))


def compare(v):
    # These ten definitions deliberately match the existing compare.py, including
    # its phase-zero means and its unshifted four-cycle EMA initialization.
    p = (np.abs(np.fft.fft(v - 0.5, axis=0)) ** 2).mean(axis=(1, 2))
    fy, fx = np.meshgrid(np.fft.fftfreq(128), np.fft.fftfreq(128), indexing='ij')
    r = np.hypot(fx, fy)
    ps = (np.abs(np.fft.fft2(v - 0.5)) ** 2).mean(axis=0)
    out = {'tmpLow': float(p[1:4].mean() / p[1:].mean()),
           'spLow': float(ps[(r < .1) & (r > 0)].mean() / ps[r > .3].mean())}
    acc = np.zeros((128, 128))
    for n in range(1, 257):
        acc += fetch(v, n - 1)
        if n in (1, 4, 32, 256):
            if n in (32, 256):
                out[f'pp{n}'] = float((acc / n).std())
            out[f'bl{n}'] = float(gauss(acc / n).std())
    h = v.mean(0)
    states = []
    for n in range(128):
        h = h + .1 * (v[n % 32] - h)
        if n >= 96:
            states.append(h.copy())
    out['ema'] = float(np.mean([x.std() for x in states]))
    out['ema+bl'] = float(np.mean([gauss(x).std() for x in states]))
    return out


def errors(estimates, truth):
    e = estimates - truth
    bias = e.mean(axis=(-1, -2))
    std = e.std(axis=(-1, -2))
    return {'rmse': float(np.sqrt(np.mean(e * e))),
            'bias': float(bias.mean()), 'bias_rms': float(np.sqrt(np.mean(bias * bias))),
            'spatial_std_rms': float(np.sqrt(np.mean(std * std)))}


def score_integrand(v, truth):
    # Start phases are 0..31 on one fixed timeline. Crossing the end of the
    # volume uses exactly compare.py's (71,37) cycle offsets, never fresh noise.
    timeline = np.array([fetch(v, n) for n in range(128)])
    result = {}
    for mode, sequence in [('raw', timeline), ('sigma1', gauss(timeline))]:
        sums = np.concatenate([np.zeros((1, 128, 128)), sequence.cumsum(axis=0)])
        phase = np.arange(32)
        result[mode] = {str(n): errors((sums[phase + n] - sums[phase]) / n, truth) for n in FRAMES}
        # All 32 start phases independently run four cycles, with the same
        # frozen-volume access and cycle offsets as their box averages.
        # Construct shifted extension for start phases through 31.
        extended = np.array([fetch(v, n) for n in range(159)])
        if mode == 'sigma1':
            extended = gauss(extended)
        h = np.repeat(v.mean(0)[None], 32, axis=0)
        if mode == 'sigma1':
            h = gauss(h)
        final = []
        for t in range(128):
            h += .1 * (extended[t + phase] - h)
            if t >= 96:
                # Accumulate moments instead of retaining 32*32 full images.
                final.append(errors(h, truth))
        result[mode]['ema01_4cycles'] = {
            'rmse': float(np.sqrt(np.mean([x['rmse'] ** 2 for x in final]))),
            'bias': float(np.mean([x['bias'] for x in final])),
            'bias_rms': float(np.sqrt(np.mean([x['bias_rms'] ** 2 for x in final]))),
            'spatial_std_rms': float(np.sqrt(np.mean([x['spatial_std_rms'] ** 2 for x in final])))}
    return result


def integrands(v):
    out = {}
    for threshold in (.1, .25, .5, .75, .9):
        out[f'indicator_lt_{threshold:g}'] = score_integrand((v < threshold).astype(float), threshold)
    out['x_squared'] = score_integrand(v ** 2, 1 / 3)
    out['x_fourth'] = score_integrand(v ** 4, 1 / 5)
    return out


def contact_sheet(volumes, target):
    canvas = Image.new('RGB', (3 * 256, len(volumes) * 284), 'white')
    draw = ImageDraw.Draw(canvas)
    for row, (label, v) in enumerate(volumes.items()):
        y = row * 284
        draw.text((4, y + 2), label + ' | frame / blurred threshold error (gain 4) / log power spectrum', fill='black')
        frame = np.uint8(np.clip(v[0] * 255, 0, 255))
        err = gauss((v[0] < .5).astype(float)) - .5
        error_rgb = np.stack([.5 + err * 4, .5 - np.abs(err) * 4, .5 - err * 4], axis=-1)
        spectrum = np.fft.fftshift((np.abs(np.fft.fft2(v - .5)) ** 2).mean(axis=0))
        logp = np.log1p(spectrum)
        logp /= max(float(np.percentile(logp, 99.5)), 1e-12)
        panels = [Image.fromarray(frame), Image.fromarray(np.uint8(np.clip(error_rgb, 0, 1) * 255)), Image.fromarray(np.uint8(np.clip(logp, 0, 1) * 255))]
        for col, panel in enumerate(panels):
            canvas.paste(panel.resize((256, 256), Image.Resampling.NEAREST).convert('RGB'), (col * 256, y + 24))
    canvas.save(target)


def main():
    output = Path(sys.argv[1])
    volumes = {}
    for name in ('hierarchy', 'lattice', 'relaxation', 'fast'):
        for seed in (1, 2):
            file = output / f'{name}-{seed}.f32'
            if file.exists():
                raw = np.fromfile(file, np.float32).reshape(32, 128, 128)
                volumes[f'{name}-{seed}'] = (np.floor(np.clip(raw, 0, .999999) * 256).astype(np.float64) + .5) / 256
    atlas = Path(__file__).resolve().parents[2] / 'examples/textures/noises/fast/gauss1-ema01-spatio-temporal.png'
    if atlas.exists():
        # make_atlas.py packs vector2 in RG and the independently optimized scalar in B.
        raw = np.asarray(Image.open(atlas).convert('RGB')).reshape(32, 128, 256, 3)[:, :, :128, 2]
        volumes['ea-shipped'] = (raw.astype(float) + .5) / 256
    for seed in (1, 2):
        raw = np.random.default_rng(seed).random((32, 128, 128))
        volumes[f'white-{seed}'] = (np.floor(raw * 256) + .5) / 256
    report = {'protocol': {'quantization': 'floor(x*256), then bin center (q+0.5)/256', 'frames': FRAMES,
              'phases': list(range(32)), 'cycle_shift_yx': [-71, -37], 'ema': 'alpha .1; 4 cycles; average final cycle over all 32 independent start phases; initialize volume mean',
              'note': 'Frozen scalar volumes, wrapped sigma=1 radius=3 blur. Numerical sampling diagnostics, not rendered-scene validation.'}, 'methods': {}}
    for label, v in volumes.items():
        ten = compare(v)
        print(label, json.dumps(ten), flush=True)
        report['methods'][label] = {'compare': ten}
        if label.endswith('-2'):
            continue
        report['methods'][label]['integrands'] = integrands(v)
        rmse = report['methods'][label]['integrands']['indicator_lt_0.5']['sigma1']
        print('  threshold .5 blurred RMSE:', {n: round(rmse[str(n)]['rmse'], 7) for n in FRAMES}, flush=True)
    (output / 'scores.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    contact_sheet({k: v for k, v in volumes.items() if not k.endswith('-2')}, output / 'contact-sheet.png')


if __name__ == '__main__':
    main()

