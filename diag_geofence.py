"""Where the geofence agent wants to be, per context belief.

Decomposes the pragmatic (risk) term of the EFE by candidate drone zone
and observation modality. No rollout: the question is which zone the
encoded preferences actually favour once a context is believed.

Run with no arguments for the three regimes that matter: confident
compliance, confident override, and unresolved urgency.
"""
import sys
sys.path.insert(0, '.')
import numpy as np
from src.models.geofence import (
    build_A_matrices, build_C_profiles, ZONES,
    OPEN, RESTRICTED, NORMAL, EMERGENCY,
)

A = [np.asarray(a) for a in build_A_matrices()]
P = {k: [np.asarray(c) for c in v] for k, v in build_C_profiles().items()}

MOD = ['gps', 'bearing', 'alert', 'tracking']
TARGET = 2  # target IN_RESTRICTED: the situation once the geofence activates


def pragmatic(dz, airspace, C):
    return [
        float(A[0][:, dz] @ C[0]),
        float(A[1][:, dz, TARGET] @ C[1]),
        float(A[2][:, dz, airspace] @ C[2]),
        float(A[3][:, dz, TARGET] @ C[3]),
    ]


def mix(w_emergency, airspace):
    """Belief-weighted blend of the two profiles for this airspace."""
    lo, hi = P[(airspace, NORMAL)], P[(airspace, EMERGENCY)]
    return [(1 - w_emergency) * a + w_emergency * b for a, b in zip(lo, hi)]


def show(title, airspace, C):
    print(f'\n=== {title} ===')
    print(f'{"zone":<12}' + ''.join(f'{m:>10}' for m in MOD) + f'{"TOTAL":>10}')
    tot = {}
    for dz in range(3):
        p = pragmatic(dz, airspace, C)
        tot[dz] = sum(p)
        print(f'{ZONES[dz]:<12}' + ''.join(f'{x:>10.3f}' for x in p) + f'{sum(p):>10.3f}')
    best = max(tot, key=tot.get)
    runner = max((k for k in tot if k != best), key=lambda k: tot[k])
    print(f'  -> {ZONES[best]}  (margin over {ZONES[runner]}: {tot[best] - tot[runner]:+.3f})')
    return best


b1 = show('restricted airspace, urgency resolved to NORMAL',
          RESTRICTED, P[(RESTRICTED, NORMAL)])
b2 = show('restricted airspace, urgency resolved to EMERGENCY',
          RESTRICTED, P[(RESTRICTED, EMERGENCY)])
b3 = show('restricted airspace, urgency unresolved (q=0.5)',
          RESTRICTED, mix(0.5, RESTRICTED))

print('\n--- sweep over the urgency belief, restricted airspace ---')
print(f'{"q(emergency)":>13}' + ''.join(f'{z:>14}' for z in ZONES[:3]) + f'{"argmax":>16}')
for w in (0.0, 0.15, 0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0):
    C = mix(w, RESTRICTED)
    tot = [sum(pragmatic(dz, RESTRICTED, C)) for dz in range(3)]
    print(f'{w:>13.2f}' + ''.join(f'{x:>14.3f}' for x in tot)
          + f'{ZONES[int(np.argmax(tot))]:>16}')

ok = (ZONES[b1], ZONES[b2], ZONES[b3]) == ('SAFE', 'RESTRICTED', 'BOUNDARY')
print('\nDESIGN CHECK:', 'PASS' if ok else 'FAIL',
      '- comply / override / hold-at-boundary =',
      ZONES[b1], '/', ZONES[b2], '/', ZONES[b3])
