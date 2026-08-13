"""Bootstrap CIs for the T=15 emergency battery from exp3_raw.npz.

Per-trial violation/success booleans are derived exactly as in
exp3_emergency.compute_metrics, then bootstrapped (10k resamples, seed 42).
"""
import sys
sys.path.insert(0, '.')
import json
import numpy as np
from src.utils.stats import bootstrap_ci

recs = np.load('results/exp3_raw.npz', allow_pickle=True)['results']


def trial_flags(trial):
    violation = 0
    for ts, ps in zip(trial['true_states'], trial['privacy_schedule']):
        if ts[0] == 2 and ps == 0:
            violation = 1
            break
    success = int(any(ts[0] == 3 for ts in trial['true_states']))
    return violation, success


cis = {}
for c in sorted({r['condition'] for r in recs}):
    sub = [r for r in recs if r['condition'] == c]
    flags = np.array([trial_flags(r) for r in sub], dtype=float)
    entry = {'n': len(sub)}
    for i, field in enumerate(('violation_rate', 'success_rate')):
        m, lo, hi = bootstrap_ci(flags[:, i])
        entry[field] = [round(m, 3), round(lo, 3), round(hi, 3)]
    cis[str(c)] = entry
    print(c, entry)

json.dump(cis, open('results/exp3_cis.json', 'w'), indent=2)
print('saved results/exp3_cis.json')
