"""Audit finite dominance; a win count is not a statistical significance test.
Usage: python audit-frontier.py BOX_SCREEN_JSON FULL_SCORES_JSON OUTPUT_JSON
"""
import json
from pathlib import Path
import sys

TOLERANCE = 1e-12


def flatten(row, full):
    out = {'legacy/'+k: v for k,v in row['compare'].items()}
    if full:
        assert len(row['integrands']) == 22
        for function,modes in row['integrands'].items():
            assert set(modes) == {'raw','sigma1'}
            for mode,frames in modes.items():
                assert set(frames) == {'1','4','8','16','32','64','ema01_4cycles'}
                for frame,result in frames.items():
                    out[f'{function}/{mode}/{frame}'] = result['rmse']
        assert len(out) == 318
    else:
        for function,frames in row['integrands'].items():
            for frame,value in frames.items():
                out[f'{function}/sigma1/{frame}'] = value
        assert len(out) == 98
    return out


def audit(rows, full):
    result = {}
    for name,row in rows.items():
        reference = 'fast-10000-s2' if name.endswith('-s2') else 'fast-10000-s1'
        baseline = flatten(rows[reference], full)
        candidate = flatten(row, full)
        assert set(candidate) == set(baseline)
        assert all(v > 0 for v in baseline.values())
        ratios = {k: v/baseline[k] for k,v in candidate.items()}
        wins = [k for k,v in candidate.items() if v < baseline[k]-TOLERANCE]
        losses = [k for k,v in candidate.items() if v > baseline[k]+TOLERANCE]
        worst = max(ratios, key=ratios.get)
        result[name] = {'reference':reference, 'metrics':len(candidate), 'wins':len(wins), 'losses':len(losses), 'ties':len(candidate)-len(wins)-len(losses), 'legacy_wins':sum(k.startswith('legacy/') for k in wins), 'worst_ratio':ratios[worst], 'worst_metric':worst, 'strictly_better_all':len(wins)==len(candidate), 'no_regressions':not losses, 'ratios':ratios}
    return result


if __name__ == '__main__':
    screen=json.loads(Path(sys.argv[1]).read_text())
    full=json.loads(Path(sys.argv[2]).read_text())['methods']
    output={'absolute_tolerance':TOLERANCE, 'note':'Finite correlated development diagnostics, not independent statistical trials, novel held-out renderings, or all possible metrics.', 'screen':audit(screen,False), 'full':audit(full,True)}
    Path(sys.argv[3]).write_text(json.dumps(output,indent=2),encoding='utf-8')
    for name,row in output['full'].items():
        print(name,row['wins'],row['losses'],row['ties'],round(row['worst_ratio'],5),row['worst_metric'])
