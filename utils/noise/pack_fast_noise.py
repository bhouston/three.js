"""Pack EA FAST PNG slices without resampling. Usage: python3 utils/noise/pack_fast_noise.py /path/to/noise.zip

Requires Pillow. Archive: https://media.githubusercontent.com/media/electronicarts/fastnoise/main/noise.zip
Vector2 and scalar samples are stored in opaque RGB tiles to avoid browser alpha premultiplication.
"""
import io
import hashlib
from pathlib import Path
import sys
import zipfile
from PIL import Image, ImageChops

checksum = hashlib.sha256()
with open(sys.argv[1], 'rb') as source:
    for chunk in iter(lambda: source.read(1024 * 1024), b''):
        checksum.update(chunk)
if checksum.hexdigest() != '35fe2ce496a837186931acc5afedcdf53ef22d1aaf6d01ad1dcd416c5ff68cea':
    raise ValueError('Unexpected FAST archive version; update provenance before repacking.')

output = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parents[2] / 'examples/textures/noises/fast'
layout = sys.argv[3] if len(sys.argv) > 3 else 'vector2'
output.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(sys.argv[1]) as archive:
    for variant in ('product', 'separate05'):
        atlas = Image.new('RGB', (256, 128 * 32))
        for frame in range(32):
            suffix = f'uniform_gauss1_0_exp0101_{variant}_{frame}.png'
            rgb = Image.open(io.BytesIO(archive.read(f'noise/{layout}/temporal/exp/{layout}_{suffix}'))).convert('RGB')
            scalar = Image.open(io.BytesIO(archive.read(f'noise/real/temporal/exp/real_{suffix}'))).getchannel('R')
            assert rgb.size == scalar.size == (128, 128)
            if layout == 'vector2':
                rgb = Image.merge('RGB', (rgb.getchannel('R'), rgb.getchannel('G'), scalar))
                scalar = ImageChops.offset(scalar, 49, 83)
            atlas.paste(rgb, (0, 128 * frame))
            atlas.paste(scalar.convert('RGB'), (128, 128 * frame))
        atlas.save(output / f'gauss1-ema01-{variant}.png')
