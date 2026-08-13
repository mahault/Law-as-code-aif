"""Mechanism diagnostic: is C5/C7 conservatism a transition-model artifact?

Compares the committed B (symmetric a_priv=0.01) against event-structure-matched
alternatives. Hazard values are set by the environment's true event statistics
(one ACTIVE->SUSPENDED switch per 10-step episode => 0.1; reversals never occur
within an episode => keep 0.01), not swept.
"""
import sys, json
sys.path.insert(0, '.')
import jax.numpy as jnp
import numpy as np
import src.experiments.exp3_emergency as e3
from src.models.emergency_override import build_B_matrices as build_B_orig


def build_B_asym(h_as=0.1, h_sa=0.01, a_urg=0.02):
    B = build_B_orig(a_priv=0.01, a_urg=a_urg)
    B1 = jnp.array([
        [1 - h_as, h_sa],
        [h_as, 1 - h_sa],
    ])[..., None]
    return [B[0], B1, B[2]]


CONFIGS = {
    'current': lambda: build_B_orig(),
    'asym_priv': lambda: build_B_asym(),
    'asym_priv_urg01': lambda: build_B_asym(a_urg=0.1),
}
CONDS = [1, 2, 5, 7]
NT = 12

out = {}
for cname, bfn in CONFIGS.items():
    e3.build_B_matrices = bfn
    for cond in CONDS:
        succ = viol = 0
        q_susp_final = []
        cross_times = []
        for k in range(NT):
            r = e3.run_single_trial(cond, 777000 + cond * 1000 + k)
            ts = r['true_states']
            ps = r['privacy_schedule']
            viol += any(s[0] == 2 and p == 0 for s, p in zip(ts, ps))
            succ += any(s[0] == 3 for s in ts)
            if cond == 5:
                q_susp_final.append(r['beliefs_privacy'][-1][1])
            ct = next((t for t, s2 in enumerate(ts) if s2[0] == 2), None)
            if ct is not None:
                cross_times.append(ct)
        key = f'{cname}|C{cond}'
        out[key] = {
            'success': succ / NT,
            'violation': viol / NT,
            'mean_q_susp_final_C5': float(np.mean(q_susp_final)) if q_susp_final else None,
            'mean_cross_t': float(np.mean(cross_times)) if cross_times else None,
            'n_crossed': len(cross_times),
        }
        print(key, out[key], flush=True)

json.dump(out, open('diag_results.json', 'w'), indent=2)
print('DONE', flush=True)
