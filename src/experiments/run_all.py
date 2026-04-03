"""
Master runner for all 4 experiments + figure generation.

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
from src.plotting.figures import (
    plot_fig1_minimization, plot_fig2_geofence,
    plot_fig3_emergency, plot_fig4_summary, plot_fig5_overhead,
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

    total_start = time.time()

    # ── Experiment 1: Data Minimization ──
    print(f"\n[1/4] Running Experiment 1: Data Minimization (n={n1})...")
    t0 = time.time()
    exp1_results = run_exp1(save_dir=results_dir, n_trials=n1)
    print(f"  Completed in {time.time() - t0:.1f}s")

    # ── Experiment 2: Geofence Compliance ──
    print(f"\n[2/4] Running Experiment 2: Geofence Compliance (n={n2})...")
    t0 = time.time()
    exp2_results = run_exp2(save_dir=results_dir, n_trials=n2)
    print(f"  Completed in {time.time() - t0:.1f}s")

    # ── Experiment 3: Emergency Override ──
    print(f"\n[3/4] Running Experiment 3: Emergency Override (n={n3})...")
    t0 = time.time()
    exp3_results, exp3_metrics = run_exp3(save_dir=results_dir, n_trials=n3)
    print(f"  Completed in {time.time() - t0:.1f}s")

    # ── Experiment 4: Traceability Overhead ──
    print(f"\n[4/4] Running Experiment 4: Traceability Overhead...")
    t0 = time.time()
    exp4_results = run_exp4(save_dir=results_dir)
    print(f"  Completed in {time.time() - t0:.1f}s")

    # ── Generate Figures ──
    print("\n" + "=" * 70)
    print("  Generating Publication Figures")
    print("=" * 70)

    fig1 = plot_fig1_minimization(
        exp1_results,
        save_path=results_dir / "fig1_minimization.pdf",
    )

    fig2 = plot_fig2_geofence(
        exp2_results,
        save_path=results_dir / "fig2_geofence.pdf",
    )

    fig3 = plot_fig3_emergency(
        exp3_results,
        save_path=results_dir / "fig3_emergency.pdf",
    )

    fig4 = plot_fig4_summary(
        exp3_results,
        save_path=results_dir / "fig4_summary.pdf",
    )

    fig5 = plot_fig5_overhead(
        exp4_results,
        save_path=results_dir / "fig5_overhead.pdf",
    )

    total_time = time.time() - total_start
    print(f"\n{'=' * 70}")
    print(f"  All experiments complete in {total_time:.1f}s")
    print(f"  Results saved to: {results_dir}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
