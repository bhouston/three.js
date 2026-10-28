"""Spatial diagnostics explaining the short-history gap; no rendering timings."""
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import compare as c

rank_and_phase = c.hierarchy_phase(1)
tile_phase = c.pixel_hash(c.X >> c.U(3), c.Y >> c.U(3), 1 ^ 0x02e5be93)
rank = rank_and_phase-tile_phase
global_phase = c.hash32(c.U(1))
fields = {
    '3LHW': c.hierarchy(0),
    'Animated IGN': c.ign(0),
    '3LHW global instead of tile phase': ((rank+global_phase) >> c.U(8)).astype(float)/16777216
}
report = {'note': 'Seed1, frame0 spatial ablation. Global-phase variant is a diagnostic, not a selected general replacement.', 'fields': {}}
for name, v in fields.items():
    neighbors = np.sort(np.array([np.roll(v, (dy, dx), axis=(0, 1)) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]), axis=0)
    gaps = np.diff(np.concatenate((neighbors, neighbors[:1]+1)), axis=0)
    error = c.gauss(v)-.5
    near_edge = ((c.X % 8 < 2) | (c.X % 8 >= 6) | (c.Y % 8 < 2) | (c.Y % 8 >= 6))
    harmonic = np.exp(2j*np.pi*64*v)
    report['fields'][name] = {
        'sliding3x3_circular_gap_rms': float(np.sqrt(np.mean((gaps-1/9)**2))),
        'blurred_std': float(error.std()),
        'blurred_rmse': float(np.sqrt(np.mean(error**2))),
        'mean_bias': float(error.mean()),
        'boundary_rmse': float(np.sqrt(np.mean(error[near_edge]**2))),
        'interior_rmse': float(np.sqrt(np.mean(error[~near_edge]**2))),
        'harmonic64_filtered_rms': float(np.sqrt(np.mean(np.abs(c.gauss(harmonic))**2))),
        'harmonic64_max_within8x8_std': float(np.max(harmonic.reshape(16, 8, 16, 8).std(axis=(1, 3))))
    }
Path(__file__).with_name('diagnosis.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
