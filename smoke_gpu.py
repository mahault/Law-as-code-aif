import sys, time
sys.path.insert(0, '.')
import matplotlib
matplotlib.use('Agg')
import jax
print('devices:', jax.devices())
from src.experiments.exp3_emergency import run_experiment
t0 = time.time()
res, metrics = run_experiment(n_trials=4)
print('elapsed_s', round(time.time() - t0, 1))
for c, m in metrics.items():
    print(c, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items()})
