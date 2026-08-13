"""Check: asym-hazard B + T=15 on C1/C5/C7 (switches stay at t=7/t=4)."""
import sys, json
sys.path.insert(0, '.')
import jax.numpy as jnp
import numpy as np
import src.experiments.exp3_emergency as e3
from src.models.emergency_override import build_B_matrices as build_B_orig

def build_B_asym():
    B = build_B_orig()
    B1 = jnp.array([[0.9, 0.01], [0.1, 0.99]])[..., None]
    return [B[0], B1, B[2]]

e3.build_B_matrices = build_B_asym
e3.T = 15

out = {}
for cond in [1, 5, 7]:
    succ = viol = 0
    q_susp_final = []
    cross_times = []
    for k in range(12):
        r = e3.run_single_trial(cond, 888000 + cond * 1000 + k)
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
json.dump(out, open('diag_horizon.json', 'w'), indent=2)
print('DONE', flush=True)
