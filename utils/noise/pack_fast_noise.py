"""Pack EA FAST PNG slices without resampling. Usage: python3 utils/noise/pack_fast_noise.py /path/to/noise.zip

Requires Pillow. Archive: https://media.githubusercontent.com/media/electronicarts/fastnoise/main/noise.zip
Vector3 and scalar tiles are stored side by side in opaque RGB to avoid browser alpha premultiplication.
"""
import io
from pathlib import Path
import sys
import zipfile
from PIL import Image

output = Path(__file__).resolve().parents[2] / 'examples/textures/noises/fast'
output.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(sys.argv[1]) as archive:
    for variant in ('product', 'separate05'):
        atlas = Image.new('RGB', (256, 128 * 32))
        for frame in range(32):
            suffix = f'uniform_gauss1_0_exp0101_{variant}_{frame}.png'
            rgb = Image.open(io.BytesIO(archive.read(f'noise/vector3/temporal/exp/vector3_{suffix}'))).convert('RGB')
            scalar = Image.open(io.BytesIO(archive.read(f'noise/real/temporal/exp/real_{suffix}'))).getchannel('R')
            assert rgb.size == scalar.size == (128, 128)
            atlas.paste(rgb, (0, 128 * frame))
            atlas.paste(scalar.convert('RGB'), (128, 128 * frame))
        atlas.save(output / f'gauss1-ema01-{variant}.png')
