# Usage: uv run --with matplotlib chart.py <dir with benchmark-64.json and benchmark-96-occlusion.json> <out dir>
import json, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

src, out = sys.argv[1], sys.argv[2]
b64 = {e['example']: e for e in json.load(open(f'{src}/benchmark-64.json'))}
b96 = {e['example']: e for e in json.load(open(f'{src}/benchmark-96-occlusion.json'))}
ms = lambda e, name: next(m['impliedMs'] for m in e['modes'] if m['name'] == name)

rows = [
    ('Linear SSR,\ncomparable quality', '#64748b'),
    ('Linear SSR,\nexample defaults', '#a8b3c4'),
    ('Adaptive SSR,\n64 iterations', '#2563eb'),
    ('Adaptive SSR + occlusion\nskipping, 96 iterations', '#0891b2'),
    ('Adaptive SSR approximate,\n24 iterations', '#16a34a'),
]
iterations, steps = next(iter(b64.values()))['approximate'].split('/')
hiz_color = '#f59e0b'
creations = []

fig, axes = plt.subplots(1, 3, figsize=(13, 8), sharey=True)
fig.subplots_adjust(left=0.175, right=0.965, top=0.80, bottom=0.42, wspace=0.3)
for ax, example in zip(axes, ['mirror', 'stochastic', 'animation']):
    e64, e96 = b64[example], b96[example]
    values = [
        (ms(e64, 'Non-adaptive SSR'), 0),
        (ms(e64, 'Non-adaptive SSR (example defaults)'), 0),
        (ms(e64, 'Adaptive SSR (1 candidate)'), e64['creation']['impliedMs']),
        (ms(e96, 'Adaptive SSR + occlusion skipping (1 candidate)'), e96['creation']['impliedMs']),
        (ms(e64, 'Adaptive SSR, approximate (example defaults)'), e64['creation']['impliedMs']),
    ]
    creations += [e64['creation']['impliedMs'], e96['creation']['impliedMs']]
    for y, ((label, color), (trace, hiz)) in enumerate(zip(rows, values)):
        ax.barh(y, trace, height=0.55, color=color)
        if hiz: ax.barh(y, hiz, left=trace, height=0.55, color=hiz_color)
        ax.text(trace + hiz + 0.6, y, f'{trace + hiz:.2f}', va='center', fontweight='bold', fontsize=11)
    ax.set_title(example.capitalize(), fontweight='bold', fontsize=13, pad=14)
    ax.set_xlim(0, 36)
    ax.set_xticks([0, 10, 20, 30])
    ax.set_yticks(range(5), [r[0] for r in rows], fontsize=11)
    ax.invert_yaxis()
    ax.grid(axis='x', color='#eeeeee')
    ax.set_axisbelow(True)
    ax.tick_params(axis='y', length=0)
    for side in ('top', 'right', 'left'): ax.spines[side].set_visible(False)
    ax.set_xlabel('Implied milliseconds (1000 / FPS)', fontsize=11)

fig.suptitle('Hi-Z + adaptive SSR: before and after', fontweight='bold', fontsize=19, y=0.975)
fig.text(0.5, 0.905, '1920 × 1080 · Apple M3 (10 GPU cores) · hardware WebGPU / Chrome 157', ha='center', color='#475569', fontsize=11)
fig.legend(handles=[
    Patch(color=rows[0][1], label='Linear trace, comparable quality'),
    Patch(color=rows[1][1], label='Linear trace, example defaults'),
    Patch(color=rows[2][1], label='Adaptive trace, 64 iterations'),
    Patch(color=rows[3][1], label='Adaptive trace + occlusion skipping, 96 iterations'),
    Patch(color=rows[4][1], label=f'Adaptive trace, approximate, {iterations} iterations'),
    Patch(color=hiz_color, label=f'Hi-Z creation (~{min(creations):.2f}–{max(creations):.2f} ms, measured separately)'),
], loc='center', bbox_to_anchor=(0.53, 0.285), ncol=2, frameon=False, fontsize=10)
notes = [
    'Comparable quality and adaptive: maxDistance = 50. Linear: quality = 1, binaryRefine = true. Adaptive: hierarchical traversal, 1 candidate; labels include Hi-Z cost.',
    'Example defaults (as each example loads, binaryRefine off): mirror quality 0.5, maxDistance 1; stochastic quality 0.25, maxDistance 0.4; animation quality 0.5, maxDistance 20.',
    '5-second sequential runs after warm-up; vsync / caps off; completed GPU batches counted. Scene / denoising excluded.',
    f'Approximate adaptive ({iterations} iterations, {steps} approximate steps) point-samples coarse cells like a linear step and, like the example defaults, traces each example\'s own maxDistance.',
    'Occlusion skipping lets adaptive rays pass behind occluders, as linear SSR does, recovering reflections seen past them. It needs 96 iterations to avoid streaks.',
]
for i, note in enumerate(notes): fig.text(0.05, 0.19 - i * 0.031, note, fontsize=9)
fig.text(0.05, 0.19 - 5 * 0.031, 'Example defaults are faster because they take fewer steps and (mirror, stochastic) trace much shorter rays. Hit coverage differs between all five.', fontsize=9, color='#9a3412')
fig.savefig(f'{out}/performance.svg')
fig.savefig(f'{out}/performance.png', dpi=180)
