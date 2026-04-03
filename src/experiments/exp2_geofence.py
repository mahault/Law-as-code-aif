"""
Experiment 2: EASA Geofence Compliance

50 trials × 30 timesteps per condition.
3 conditions: PID-only, Rule-based geofence, AIF-LAL.
Target walks toward restricted zone.
"""

import sys
import json
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import equinox as eqx
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pymdp.agent import Agent
from src.models.geofence import (
    build_A_matrices, build_B_matrices, build_C_vectors, build_D_priors,
    get_A_dependencies, get_B_dependencies, build_target_schedule,
    ZONES, TARGET_POS, STAY, ADVANCE, RETREAT,
)
from src.environments.drone_env import DroneEnv

T = 30
N_TRIALS = 50


def run_condition(condition, n_trials, seed=42):
    """Run one condition across multiple trials.

    condition: 'pid_only' | 'rule_based' | 'aif_lal'
    """
    A = build_A_matrices()
    B = build_B_matrices()
    D = build_D_priors()
    A_deps = get_A_dependencies()

    # Per-trial metrics
    all_violations = []
    all_tracking = []
    all_abort = []

    # Time-series for averaging
    violations_over_time = np.zeros(T)
    tracking_over_time = np.zeros(T)

    for trial in range(n_trials):
        trial_seed = seed * 10000 + trial
        rng = jr.PRNGKey(trial_seed)

        target_schedule = build_target_schedule(T)
        airspace_schedule = [0] * (T // 2) + [1] * (T - T // 2)  # OPEN then RESTRICTED

        env = DroneEnv(
            A=A, B=B,
            schedules={1: target_schedule, 2: airspace_schedule},
            num_states=[4, 3, 2],
            control_fac_idx=[0],
        )
        true_state = env.reset(D)
        true_state[1] = target_schedule[0]
        true_state[2] = airspace_schedule[0]

        # AIF agent for aif_lal
        if condition == "aif_lal":
            C = build_C_vectors("open")
            agent = Agent(
                A=A, B=B, C=C, D=D,
                A_dependencies=A_deps,
                B_dependencies=get_B_dependencies(),
                control_fac_idx=[0],
                policy_len=2,
                gamma=16.0,
                action_selection="stochastic",
                sampling_mode="marginal",
            )
            qs = [jnp.expand_dims(d, -2) for d in agent.D]
            action = -jnp.ones((1, 3), dtype=jnp.int32)

        trial_violations = 0
        trial_tracked = 0
        aborted = False

        for t in range(T):
            rng, obs_key, act_key = jr.split(rng, 3)

            if condition == "pid_only":
                # PID follows target blindly
                target_pos = target_schedule[min(t, len(target_schedule) - 1)]
                drone_zone = true_state[0]
                if target_pos > drone_zone:
                    zone_action = ADVANCE
                elif target_pos < drone_zone:
                    zone_action = RETREAT
                else:
                    zone_action = STAY

            elif condition == "rule_based":
                # Hard geofence: NEVER enter RESTRICTED (zone >= 2)
                drone_zone = true_state[0]
                target_pos = target_schedule[min(t, len(target_schedule) - 1)]
                if drone_zone >= 1:  # at boundary or beyond
                    zone_action = RETREAT
                elif target_pos > drone_zone:
                    zone_action = ADVANCE
                else:
                    zone_action = STAY

            else:
                # AIF-LAL: context-dependent
                obs_list = env.generate_observation(true_state, A_deps, obs_key)
                obs_batch = [jnp.array([[int(o)]]) for o in obs_list]

                # Context-dependent C: check airspace belief
                qs_latest = [q[:, -1, :] for q in qs]
                airspace_belief = qs_latest[2][0]  # (2,)
                if airspace_belief[1] > 0.5:
                    new_C = build_C_vectors("restricted")
                else:
                    new_C = build_C_vectors("open")
                batched_C = [c[None, ...] for c in new_C]
                agent = eqx.tree_at(lambda a: a.C, agent, batched_C)

                if jnp.any(action < 0):
                    emp_prior = agent.D
                else:
                    emp_prior, qs = agent.update_empirical_prior(action, qs)

                qs = agent.infer_states(obs_batch, emp_prior)
                q_pi, G = agent.infer_policies(qs)
                action = agent.sample_action(q_pi, rng_key=jr.split(act_key, 1))
                qs = [q[:, -1:, :] for q in qs]

                zone_action = int(action[0, 0])

            # Apply action and step
            full_action = [zone_action, 0, 0]
            next_state = env.step(true_state, full_action)

            # Check violation: drone in RESTRICTED or EMERGENCY zone
            if next_state[0] >= 2:
                trial_violations += 1
                violations_over_time[t] += 1

            # Check tracking: drone near target
            dist = abs(next_state[0] - next_state[1])
            if dist <= 1:
                trial_tracked += 1
                tracking_over_time[t] += 1

            true_state = next_state

        all_violations.append(trial_violations)
        all_tracking.append(trial_tracked / T)
        all_abort.append(1 if trial_tracked / T < 0.1 else 0)

    return {
        "condition": condition,
        "violations_mean": float(np.mean(all_violations)),
        "violations_std": float(np.std(all_violations)),
        "tracking_pct_mean": float(np.mean(all_tracking)),
        "tracking_pct_std": float(np.std(all_tracking)),
        "abort_rate": float(np.mean(all_abort)),
        "violations_over_time": (violations_over_time / n_trials).tolist(),
        "tracking_over_time": (tracking_over_time / n_trials).tolist(),
        "n_trials": n_trials,
    }


def run_experiment(seed=42, n_trials=N_TRIALS, save_dir=None):
    """Run all 3 conditions."""
    print("=" * 60)
    print("Experiment 2: Geofence Compliance (EASA)")
    print(f"  Trials per condition: {n_trials}")
    print(f"  Timesteps: {T}")
    print("=" * 60)

    conditions = ["pid_only", "rule_based", "aif_lal"]
    all_results = []

    for cond in conditions:
        print(f"\n--- {cond} ---")
        result = run_condition(cond, n_trials, seed)
        all_results.append(result)
        print(f"  Violations: {result['violations_mean']:.1f} ± {result['violations_std']:.1f}")
        print(f"  Tracking:   {result['tracking_pct_mean']:.3f} ± {result['tracking_pct_std']:.3f}")
        print(f"  Abort rate: {result['abort_rate']:.3f}")

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / "exp2_results.json", "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {save_dir}")

    return all_results


if __name__ == "__main__":
    results_dir = Path(__file__).resolve().parents[2] / "results"
    run_experiment(save_dir=results_dir)
