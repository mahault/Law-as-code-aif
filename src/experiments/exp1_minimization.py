"""
Experiment 1: GDPR Data Minimization

100 trials × 20 timesteps per condition.
3 conditions: Baseline (always RAW), Rule-based (always ANONYMIZED), AIF-LAL.
Vary bystander density: [0, 2, 5, 10].
"""

import sys
import json
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pymdp.agent import Agent
from src.models.data_minimization import (
    build_A_matrices, build_B_matrices, build_C_vectors, build_D_priors,
    get_A_dependencies, get_B_dependencies, build_scene_schedule,
    PIPELINE, SCENE, EXPOSURE,
)
from src.environments.drone_env import DroneEnv

T = 20
N_TRIALS = 100
BYSTANDER_DENSITIES = [0, 2, 5, 10]


def run_condition(condition, bystander_density, n_trials, seed=42):
    """Run one condition across multiple trials.

    condition: 'baseline' | 'rule_based' | 'aif_lal'
    """
    A = build_A_matrices()
    B = build_B_matrices()
    C = build_C_vectors()
    D = build_D_priors()
    A_deps = get_A_dependencies()
    B_deps = get_B_dependencies()

    # For non-AIF conditions, we still simulate the env for metrics
    exposure_counts = []
    tracking_counts = []

    for trial in range(n_trials):
        trial_seed = seed * 10000 + bystander_density * 1000 + trial
        rng = jr.PRNGKey(trial_seed)

        scene_schedule = build_scene_schedule(T, bystander_density)
        consent_schedule = [0] * T  # no consent throughout

        env = DroneEnv(
            A=A, B=B,
            schedules={1: scene_schedule, 2: consent_schedule},
            num_states=[2, 3, 2],
            control_fac_idx=[0],
        )
        true_state = env.reset(D)
        true_state[1] = scene_schedule[0]
        true_state[2] = consent_schedule[0]

        # AIF agent (only used for aif_lal condition)
        if condition == "aif_lal":
            agent = Agent(
                A=A, B=B, C=C, D=D,
                A_dependencies=A_deps,
                B_dependencies=B_deps,
                control_fac_idx=[0],
                policy_len=2,
                gamma=16.0,
                action_selection="stochastic",
                sampling_mode="marginal",
            )
            qs = [jnp.expand_dims(d, -2) for d in agent.D]
            action = -jnp.ones((1, 3), dtype=jnp.int32)

        n_exposure = 0
        n_tracked = 0

        for t in range(T):
            rng, obs_key, act_key = jr.split(rng, 3)

            if condition == "baseline":
                # Always RAW pipeline
                pipeline_action = 0
            elif condition == "rule_based":
                # Always ANONYMIZED pipeline
                pipeline_action = 1
            else:
                # AIF-LAL: agent decides
                obs_list = env.generate_observation(true_state, A_deps, obs_key)
                obs_batch = [jnp.array([[int(o)]]) for o in obs_list]

                if jnp.any(action < 0):
                    emp_prior = agent.D
                else:
                    emp_prior, qs = agent.update_empirical_prior(action, qs)

                qs = agent.infer_states(obs_batch, emp_prior)
                q_pi, G = agent.infer_policies(qs)
                action = agent.sample_action(q_pi, rng_key=jr.split(act_key, 1))
                qs = [q[:, -1:, :] for q in qs]

                pipeline_action = int(action[0, 0])

            # Set pipeline mode in true state
            true_state[0] = pipeline_action

            # Compute exposure from A matrix
            scene = true_state[1]
            mode = true_state[0]
            exposure_probs = np.array(A[0][:, mode, scene])
            # FULL exposure = index 2
            n_exposure += exposure_probs[2]

            # Tracking accuracy: target present and pipeline works
            if scene in [0, 1]:  # target visible
                if mode == 0:  # RAW: perfect tracking
                    n_tracked += 1.0
                else:  # ANONYMIZED: slightly degraded
                    n_tracked += 0.85

            # Step env
            next_state = env.step(true_state, [pipeline_action, 0, 0])
            true_state = next_state

        exposure_ratio = n_exposure / T
        tracking_acc = n_tracked / T

        exposure_counts.append(float(exposure_ratio))
        tracking_counts.append(float(tracking_acc))

    return {
        "condition": condition,
        "bystander_density": bystander_density,
        "exposure_ratio_mean": float(np.mean(exposure_counts)),
        "exposure_ratio_std": float(np.std(exposure_counts)),
        "tracking_acc_mean": float(np.mean(tracking_counts)),
        "tracking_acc_std": float(np.std(tracking_counts)),
        "n_trials": n_trials,
    }


def run_experiment(seed=42, n_trials=N_TRIALS, save_dir=None):
    """Run all conditions across all bystander densities."""
    print("=" * 60)
    print("Experiment 1: Data Minimization (GDPR Art 5)")
    print(f"  Densities: {BYSTANDER_DENSITIES}")
    print(f"  Trials per condition: {n_trials}")
    print(f"  Timesteps: {T}")
    print("=" * 60)

    all_results = []
    conditions = ["baseline", "rule_based", "aif_lal"]

    for density in BYSTANDER_DENSITIES:
        print(f"\n--- Bystander density: {density} ---")
        for cond in conditions:
            result = run_condition(cond, density, n_trials, seed)
            all_results.append(result)
            print(f"  {cond:>12}: exposure={result['exposure_ratio_mean']:.3f} "
                  f"tracking={result['tracking_acc_mean']:.3f}")

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / "exp1_results.json", "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {save_dir}")

    return all_results


if __name__ == "__main__":
    results_dir = Path(__file__).resolve().parents[2] / "results"
    run_experiment(save_dir=results_dir)
