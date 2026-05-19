"""
Master runner for all experiments + figure generation.

Usage: python src/experiments/run_all.py [--quick]
       --quick: Reduced trial counts for fast validation
"""

import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use("Agg")

from src.experiments.exp1_minimization import run_experiment as run_exp1
from src.experiments.exp2_geofence import run_experiment as run_exp2
from src.experiments.exp3_emergency import run_experiment as run_exp3
from src.experiments.exp4_traceability import run_experiment as run_exp4
from src.experiments.exp_ablation import run_experiment as run_ablation
from src.experiments.exp_baselines import run_experiment as run_baselines
from src.experiments.exp_sensitivity import run_experiment as run_sensitivity
from src.experiments.exp_noise import run_experiment as run_noise
from src.experiments.exp_learning import run_experiment as run_learning
from src.plotting.figures import (
    plot_fig1_minimization, plot_fig2_geofence,
    plot_fig3_emergency, plot_fig4_summary, plot_fig5_overhead,
    plot_fig_ablation, plot_fig_baselines, plot_fig_sensitivity,
    plot_fig_noise, plot_fig_learning,
)


def main():
    quick = "--quick" in sys.argv

    results_dir = Path(__file__).resolve().parents[2] / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("  LAW AS CODE — Active Inference Simulation Suite")
    if quick:
        print("  MODE: Quick validation (reduced trials)")
    print("=" * 70)

    # Trial counts
    n1 = 3 if quick else 100
    n2 = 3 if quick else 50
    n3 = 3 if quick else 50
    n_abl = 3 if quick else 100
    n_base = 3 if quick else 100
    n_sens = 3 if quick else 50
    n_noise = 3 if quick else 100
    n_learn = 3 if quick else 50

    total_start = time.time()

    # ── Experiment 1: Data Minimization ──
    print(f"\n[1/9] Running Experiment 1: Data Minimization (n={n1})...")
    t0 = time.time()
    exp1_results = run_exp1(save_dir=results_dir, n_trials=n1)
    print(f"  Completed in {time.time() - t0:.1f}s")

    # ── Experiment 2: Geofence Compliance ──
    print(f"\n[2/9] Running Experiment 2: Geofence Compliance (n={n2})...")
    t0 = time.time()
    exp2_results = run_exp2(save_dir=results_dir, n_trials=n2)
    print(f"  Completed in {time.time() - t0:.1f}s")

    # ── Experiment 3: Emergency Override ──
    print(f"\n[3/9] Running Experiment 3: Emergency Override (n={n3})...")
    t0 = time.time()
    exp3_results, exp3_metrics = run_exp3(save_dir=results_dir, n_trials=n3)
    print(f"  Completed in {time.time() - t0:.1f}s")

    # ── Experiment 4: Traceability Overhead ──
    print(f"\n[4/9] Running Experiment 4: Traceability Overhead...")
    t0 = time.time()
    exp4_results = run_exp4(save_dir=results_dir)
    print(f"  Completed in {time.time() - t0:.1f}s")

    # ── Experiment 5: EFE Ablation ──
    print(f"\n[5/9] Running EFE Ablation (n={n_abl})...")
    t0 = time.time()
    ablation_results, ablation_metrics = run_ablation(save_dir=results_dir, n_trials=n_abl)
    print(f"  Completed in {time.time() - t0:.1f}s")

    # ── Experiment 6: Baselines ──
    print(f"\n[6/9] Running Baselines (n={n_base})...")
    t0 = time.time()
    baseline_results, baseline_metrics = run_baselines(save_dir=results_dir, n_trials=n_base)
    print(f"  Completed in {time.time() - t0:.1f}s")

    # ── Experiment 7: Sensitivity ──
    print(f"\n[7/9] Running Sensitivity Sweep (n={n_sens})...")
    t0 = time.time()
    sensitivity_results = run_sensitivity(save_dir=results_dir, n_trials=n_sens)
    print(f"  Completed in {time.time() - t0:.1f}s")

    # ── Experiment 8: Noise Robustness ──
    print(f"\n[8/9] Running Noise Robustness (n={n_noise})...")
    t0 = time.time()
    noise_results = run_noise(save_dir=results_dir, n_trials=n_noise)
    print(f"  Completed in {time.time() - t0:.1f}s")

    # ── Experiment 9: Learning ──
    print(f"\n[9/9] Running Learning Experiment (n={n_learn})...")
    t0 = time.time()
    learning_results, learning_curve = run_learning(save_dir=results_dir, n_trials=n_learn)
    print(f"  Completed in {time.time() - t0:.1f}s")

    # ── Generate Figures ──
    print("\n" + "=" * 70)
    print("  Generating Publication Figures")
    print("=" * 70)

    fig_dir = results_dir

    plot_fig1_minimization(exp1_results, save_path=fig_dir / "fig1_minimization.pdf")
    plot_fig2_geofence(exp2_results, save_path=fig_dir / "fig2_geofence.pdf")
    plot_fig3_emergency(exp3_results, save_path=fig_dir / "fig3_emergency.pdf")
    plot_fig4_summary(exp3_results, save_path=fig_dir / "fig4_summary.pdf")
    plot_fig5_overhead(exp4_results, save_path=fig_dir / "fig5_overhead.pdf")
    plot_fig_ablation(ablation_results, save_path=fig_dir / "fig6_ablation.pdf")
    plot_fig_baselines(baseline_results, save_path=fig_dir / "fig7_baselines.pdf")
    plot_fig_sensitivity(sensitivity_results, save_path=fig_dir / "fig8_sensitivity.pdf")
    plot_fig_noise(noise_results, save_path=fig_dir / "fig9_noise.pdf")
    plot_fig_learning(learning_results, learning_curve, save_path=fig_dir / "fig10_learning.pdf")

    total_time = time.time() - total_start
    print(f"\n{'=' * 70}")
    print(f"  All experiments complete in {total_time:.1f}s")
    print(f"  Results saved to: {results_dir}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
