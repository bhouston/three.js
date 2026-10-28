"""Cross-check all 88 original baseline points against the unchanged script."""
import json
from pathlib import Path
import subprocess
import sys

folder = Path(__file__).resolve().parent
root = folder.parents[1]
data = json.loads((folder / 'results.json').read_text())
original = json.loads(subprocess.check_output([
    sys.executable, str(folder / 'original-convergence.py'),
    str(root / 'examples/textures/noises/fast/gauss1-ema01-spatial.png'),
    str(root / 'examples/textures/noises/fast/gauss1-ema01-spatio-temporal.png')
]))
aliases = {'White': 'White', 'FAST Spatial': 'FAST Spatial (Product)',
           'FAST Spatio-Temporal': 'FAST Spatio-Temporal (Separate)', 'Golden Ratio': 'Golden Ratio'}
assert original['frames'] == data['frames']
count = 0
for field in ('raw', 'blur'):
    for old, new in aliases.items():
        assert original[field][old] == [round(v, 5) for v in data[field][new]], (field, new)
        count += len(original[field][old])
assert count == 88
print('All 88 baseline points agree exactly with the original script output.')
