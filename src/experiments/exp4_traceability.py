"""
Experiment 4: Traceability Overhead Benchmark

1000 decision cycles, wall-clock timing of PID / AIF / CIL.
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.traceability import run_benchmark


def run_experiment(save_dir=None):
    """Run the overhead benchmark."""
    print("=" * 60)
    print("Experiment 4: Traceability Overhead (EU AI Act)")
    print("=" * 60)

    results = run_benchmark(n_cycles=1000)

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / "exp4_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {save_dir}")

    return results


if __name__ == "__main__":
    results_dir = Path(__file__).resolve().parents[2] / "results"
    run_experiment(save_dir=results_dir)
