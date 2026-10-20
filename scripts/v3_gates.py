#!/usr/bin/env python3
"""Apply PROTOCOL_V3 §5 gates G1-G3 to development (validation) results."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default='results/v3')
    p.add_argument('--subjects', nargs='+', default=['UTS01', 'UTS02', 'UTS03'])
    a = p.parse_args()
    val = {}
    for s in a.subjects:
        for path in (Path(a.root) / s / 'develop' / 'cells').glob('*.json'):
            cell = json.loads(path.read_text())
            if cell.get('estimable'):
                val[(s, cell['arm'], f"{cell['fraction']:g}")] = cell['val']['session_macro_feature_correlation']
    diff = lambda arm, f: np.array([val[(s, arm, f)] - val[(s, 'V0', f)] for s in a.subjects])
    def passes(arm):
        d1, d2 = diff(arm, '1'), diff(arm, '0.1')
        return dict(delta_100=d1.round(5).tolist(), mean_100=round(float(d1.mean()), 5),
                    delta_10=d2.round(5).tolist(), mean_10=round(float(d2.mean()), 5),
                    G1=bool(d1.mean() >= 0.005 and (d1 > 0).all()),
                    G2=bool(d2.mean() >= 0 and (d2 >= -0.005).all()))
    arms = {arm: passes(arm) for arm in ('V1', 'V2', 'V3', 'V4')}
    mean_val = {arm: float(np.mean([val[(s, arm, '1')] for s in a.subjects])) for arm in ('V0', 'V1', 'V2', 'V3', 'V4')}
    passing = [arm for arm in ('V2', 'V3', 'V4') if arms[arm]['G1'] and arms[arm]['G2']]
    if not passing:
        advanced, reason = 'V0', 'No structured arm passed G1 and G2: voxel ridge stays the decoder'
    elif 'V4' in passing:
        simpler = [arm for arm in ('V2', 'V3') if arm in passing and mean_val['V4'] - mean_val[arm] <= 0.002]
        advanced = simpler[0] if simpler else 'V4'
        reason = (f'{advanced} is within 0.002 of V4 and simpler' if simpler else
                  'V4 passes G1 and G2; no simpler arm passes and is within 0.002')
    else:
        advanced, reason = passing[0], f'{passing[0]} passes G1 and G2'
    v1 = arms['V1']
    gates = dict(G0='passed on the precision-aware decomposition (PROTOCOL_V3 §10)', arms=arms,
                 mean_validation_r_100=mean_val, advanced_arm=advanced, decision=reason,
                 output_wise_explanation=bool(v1['G1'] and v1['G2'] and abs(mean_val['V1'] - mean_val[advanced]) <= 0.002))
    print(json.dumps(gates, indent=1))


if __name__ == '__main__':
    main()
