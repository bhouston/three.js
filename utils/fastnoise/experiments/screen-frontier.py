"""Cheap screening of all-phase box-mean integrand RMSE; not final acceptance."""
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np

spec = importlib.util.spec_from_file_location('frontier', Path(__file__).with_name('score-frontier.py'))
frontier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(frontier)
directory = Path(sys.argv[1])
output = directory/'box-screen.json'
report = json.loads(output.read_text()) if output.exists() else {}
for name in sys.argv[2:]:
    raw = np.fromfile(directory/(name+'.f32'), np.float32).reshape(32,128,128)
    v = (np.floor(raw*256).astype(float)+.5)/256
    row = {'compare': frontier.score.compare(v), 'integrands': {}}
    phases = np.arange(32)
    for key,(f,truth) in frontier.integrands(v).items():
        planes = frontier.score.gauss(f)
        timeline = np.array([frontier.score.fetch(planes,n) for n in range(64)])
        sums = np.concatenate([np.zeros((1,128,128)), timeline.cumsum(0)])
        row['integrands'][key] = {str(n): float(np.sqrt(np.mean(((sums[phases+n]-sums[phases])/n-truth)**2))) for n in (1,4,16,32)}
    report[name] = row
    print(name,flush=True)
output.write_text(json.dumps(report,indent=2))
