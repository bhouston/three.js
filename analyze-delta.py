"""Reference-relative error figures from matched, unmodified RGB screenshots.

Run: uv run --with pillow --with numpy python analyze-delta.py
Inputs and outputs are beside this script. All metrics use display-encoded RGB.
"""
from pathlib import Path
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont

p = Path(__file__).resolve().parent
font_path = '/System/Library/Fonts/Supplemental/Arial.ttf'
def font(size):
    try:
        return ImageFont.truetype(font_path, size)
    except OSError:
        return ImageFont.load_default(size=size)

images = [Image.open(p / (name + '.png')).convert('RGB') for name in ['linear', 'catmull-rom', 'reference']]
assert all(im.size == (962, 722) for im in images)
linear, catmull, reference = [np.asarray(im, dtype=np.float64) for im in images]
errors = [np.abs(im - reference).mean(axis=2) for im in [linear, catmull]]
gain = errors[0] - errors[1]
ERROR_GAIN = 4
DELTA_LIMIT = 32  # mean absolute RGB error, in 8-bit display code values

def residual(error):
    gray = np.clip(error * ERROR_GAIN, 0, 255).astype(np.uint8)
    return Image.fromarray(np.repeat(gray[:, :, None], 3, axis=2))

def diverging(delta):
    t = np.clip(delta / DELTA_LIMIT, -1, 1)
    neutral = np.array([22., 29., 38.])
    good = np.array([70., 235., 126.])
    bad = np.array([245., 91., 197.])
    strength = np.abs(t)[..., None]
    target = np.where((t >= 0)[..., None], good, bad)
    return Image.fromarray(np.rint(neutral + strength * (target - neutral)).astype(np.uint8))

err_images = [residual(e) for e in errors]
gain_image = diverging(gain)
regions = [('A: Siemens star', (0, 300, 180, 220)), ('B: Checkerboard', (370, 640, 180, 80))]
metrics = {}
for name, (x, y, w, h) in [('Full frame', (0, 0, 962, 722)), *regions]:
    diffs = [im[y:y+h, x:x+w] - reference[y:y+h, x:x+w] for im in [linear, catmull]]
    mse = [float(np.mean((d / 255) ** 2)) for d in diffs]
    metrics[name] = {
        'xywh': [x, y, w, h],
        'linearPSNRdB': float(-10 * np.log10(mse[0])),
        'catmullRomPSNRdB': float(-10 * np.log10(mse[1])),
        'mseReductionPercent': 100 * (1 - mse[1] / mse[0]),
        'linearMAE8bit': float(np.mean(np.abs(diffs[0]))),
        'catmullRomMAE8bit': float(np.mean(np.abs(diffs[1]))),
    }

bg = '#101a23'
overview = images[1].copy()
d = ImageDraw.Draw(overview)
for name, (x, y, w, h) in regions:
    d.rectangle((x, y, x+w-1, y+h-1), outline='#ffbd45', width=3)
    label = name.split(':')[0]
    d.rectangle((x+4, y+4, x+34, y+38), fill=bg)
    d.text((x+11, y+6), label, fill='#ffce6b', font=font(24))

fig = Image.new('RGB', (1996, 1812), bg)
d = ImageDraw.Draw(fig)
d.text((24, 20), 'Where does Catmull-Rom improve the result?', fill='white', font=font(38))
d.text((24, 70), 'Reference-relative errors for the same captured frame; A and B locate the enlarged regions below.', fill='#cfdae2', font=font(25))
for x, y, title, im in [
    (24, 124, 'Catmull-Rom result | regions A and B', overview),
    (1010, 124, 'Error reduction | green: better, magenta: worse', gain_image),
    (24, 946, 'Linear error against reference | brightness x4', err_images[0]),
    (1010, 946, 'Catmull-Rom error against reference | brightness x4', err_images[1]),
]:
    d.text((x, y), title, fill='white', font=font(28))
    fig.paste(im, (x, y+46))

legend = diverging(np.linspace(-32, 32, 640)[None, :].repeat(20, axis=0))
fig.paste(legend.resize((640, 12)), (1010, 926))
d.text((1010, 899), '-32 (worse)', fill='#cfdae2', font=font(20))
d.text((1322, 899), '0', fill='#cfdae2', font=font(20))
d.text((1537, 899), '+32 (better)', fill='#cfdae2', font=font(20))
d.text((24, 1730), 'Error = mean(abs(image RGB - reference RGB)); grayscale panels share one scale and clip at white.', fill='#cfdae2', font=font(24))
d.text((24, 1767), 'Reduction = linear error - Catmull-Rom error (8-bit display RGB); color saturates at +/-32. No spatial smoothing.', fill='#cfdae2', font=font(23))
fig.save(p / 'delta-analysis.png')

zoom = Image.new('RGB', (1680, 1330), bg)
d = ImageDraw.Draw(zoom)
d.text((18, 18), 'Error close-ups | same scales as the full-frame maps', fill='white', font=font(32))
for row, (name, (x, y, w, h)) in enumerate(regions):
    top = 86 if row == 0 else 864
    m = metrics[name]
    d.text((18, top), f'{name} | MSE reduced {m["mseReductionPercent"]:.1f}% | PSNR {m["linearPSNRdB"]:.2f} -> {m["catmullRomPSNRdB"]:.2f} dB', fill='#ffce6b', font=font(25))
    for col, (label, im) in enumerate([('Linear error x4', err_images[0]), ('Catmull-Rom error x4', err_images[1]), ('Green: better | Magenta: worse', gain_image)]):
        left = 18 + col * 554
        d.text((left, top+43), label, fill='white', font=font(24))
        crop = im.crop((x, y, x+w, y+h)).resize((w*3, h*3), Image.Resampling.NEAREST)
        zoom.paste(crop, (left, top+83))
d.text((18, 1240), '3x nearest-neighbor enlargement. Brighter residuals indicate larger error against the supersampled reference.', fill='#cfdae2', font=font(22))
d.text((18, 1278), 'MSE / PSNR use normalized display RGB. These are selected regions of one frame, not a temporal average.', fill='#cfdae2', font=font(22))
zoom.save(p / 'delta-closeups.png')

report = {'metricSpace': '8-bit display-encoded RGB, normalized to 0..1 for MSE and PSNR', 'residual': 'mean(abs(image-reference), RGB) * 4, clipped to 0..255', 'signedDelta': 'MAE(linear,reference) - MAE(catmullRom,reference), green positive, magenta negative, clipped to +/-32 code values', 'regions': metrics}
(p / 'delta-metrics.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(metrics, indent=2))
