"""Decisive check: ORIGINAL committed B + T=15 on C1/C2/C5/C7."""
import sys, json
sys.path.insert(0, '.')
import numpy as np
import src.experiments.exp3_emergency as e3

e3.T = 15

out = {}
for cond in [1, 2, 5, 7]:
    succ = viol = 0
    q_susp_final = []
    cross_times = []
    for k in range(12):
        r = e3.run_single_trial(cond, 999000 + cond * 1000 + k)
        ts, ps = r['true_states'], r['privacy_schedule']
        viol += any(s[0] == 2 and p == 0 for s, p in zip(ts, ps))
        succ += any(s[0] == 3 for s in ts)
        if cond == 5:
            q_susp_final.append(r['beliefs_privacy'][-1][1])
        ct = next((t for t, s2 in enumerate(ts) if s2[0] == 2), None)
        if ct is not None:
            cross_times.append(ct)
    out[f'C{cond}'] = {
        'success': round(succ / 12, 3), 'violation': round(viol / 12, 3),
        'q_susp_final': round(float(np.mean(q_susp_final)), 3) if q_susp_final else None,
        'mean_cross_t': round(float(np.mean(cross_times)), 2) if cross_times else None,
        'n_crossed': len(cross_times),
    }
    print(f'C{cond}', out[f'C{cond}'], flush=True)
json.dump(out, open('diag_horizon2.json', 'w'), indent=2)
print('DONE', flush=True)
