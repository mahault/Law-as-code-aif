"""Run one experiment + its figures in an isolated process.

Usage: python overnight_step.py <exp1|exp2|exp3|ablation|baselines|sensitivity|noise|learning>

exp4 (overhead) is deliberately excluded: the paper's timing figures are GPU
measurements; regenerating fig5 from CPU timings would corrupt them.
"""
import sys
sys.path.insert(0, '.')
import matplotlib
matplotlib.use('Agg')
from pathlib import Path

name = sys.argv[1]
rd = Path('results')
rd.mkdir(exist_ok=True)

if name == 'exp1':
    from src.experiments.exp1_minimization import run_experiment
    from src.plotting.figures import plot_fig1_minimization
    res = run_experiment(save_dir=rd, n_trials=100)
    plot_fig1_minimization(res, save_path=rd / 'fig1_minimization.pdf')
elif name == 'exp2':
    from src.experiments.exp2_geofence import run_experiment
    from src.plotting.figures import plot_fig2_geofence
    res = run_experiment(save_dir=rd, n_trials=50)
    plot_fig2_geofence(res, save_path=rd / 'fig2_geofence.pdf')
elif name == 'exp3':
    from src.experiments.exp3_emergency import run_experiment
    from src.plotting.figures import plot_fig3_emergency, plot_fig4_summary
    res, metrics = run_experiment(save_dir=rd, n_trials=50)
    plot_fig3_emergency(res, save_path=rd / 'fig3_emergency.pdf')
    plot_fig4_summary(res, save_path=rd / 'fig4_summary.pdf')
elif name == 'ablation':
    from src.experiments.exp_ablation import run_experiment
    from src.plotting.figures import plot_fig_ablation
    res, metrics = run_experiment(save_dir=rd, n_trials=100)
    plot_fig_ablation(res, save_path=rd / 'fig6_ablation.pdf')
elif name == 'baselines':
    from src.experiments.exp_baselines import run_experiment
    from src.plotting.figures import plot_fig_baselines
    res, metrics = run_experiment(save_dir=rd, n_trials=100)
    plot_fig_baselines(res, save_path=rd / 'fig7_baselines.pdf')
elif name == 'sensitivity':
    from src.experiments.exp_sensitivity import run_experiment
    from src.plotting.figures import plot_fig_sensitivity
    res = run_experiment(save_dir=rd, n_trials=50)
    plot_fig_sensitivity(res, save_path=rd / 'fig8_sensitivity.pdf')
elif name == 'noise':
    from src.experiments.exp_noise import run_experiment
    from src.plotting.figures import plot_fig_noise
    res = run_experiment(save_dir=rd, n_trials=100)
    plot_fig_noise(res, save_path=rd / 'fig9_noise.pdf')
elif name == 'learning':
    from src.experiments.exp_learning import run_experiment
    from src.plotting.figures import plot_fig_learning
    res, curve = run_experiment(save_dir=rd, n_trials=50)
    plot_fig_learning(res, curve, save_path=rd / 'fig10_learning.pdf')
else:
    raise SystemExit(f'unknown step: {name}')

print(f'STEP {name} DONE', flush=True)
