"""
Sensitivity Analysis — Profile Magnitude Sweep

Sweep emergency target drive × normal complaint aversion.
35 cells × C1 + C2 × 50 trials.

Metric: Correct override score = P(cross | emergency) - P(cross | normal)

Validates that defeasibility emerges across a wide parameter range.
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
import equinox as eqx
from src.models.emergency_override import (
    build_A_matrices, build_B_matrices, build_D_priors,
    get_A_dependencies, get_B_dependencies,
    get_condition_schedule, NORMAL, EMERGENCY, ACTIVE, SUSPENDED,
)
from src.environments.drone_env import DroneEnv
from src.utils.profile_mixing import compute_C_effective

T = 10
N_TRIALS = 50
POLICY_LEN = 4
GAMMA = 16.0

# Sweep parameters
EMERGENCY_TARGET_DRIVES = [0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0]
NORMAL_COMPLAINT_AVERSIONS = [2.0, 4.0, 6.0, 8.0, 10.0]


def build_custom_profiles(emerg_target_drive, normal_complaint_aversion):
    """Build C profiles with custom parameter values."""
    profiles = {}

    profiles[(NORMAL, ACTIVE)] = [
        jnp.array([0.0, 0.0, -2.0, 0.5]),
        jnp.array([0.0, 0.0]),
        jnp.array([0.0, 0.0]),
        jnp.array([normal_complaint_aversion, -normal_complaint_aversion]),
    ]

    profiles[(NORMAL, SUSPENDED)] = [
        jnp.array([0.0, 0.0, 0.0, 3.0]),
        jnp.array([0.0, 0.0]),
        jnp.array([0.0, 0.0]),
        jnp.array([0.5, -0.5]),
    ]

    profiles[(EMERGENCY, ACTIVE)] = [
        jnp.array([0.0, 0.0, 0.0, emerg_target_drive]),
        jnp.array([0.0, 0.0]),
        jnp.array([0.0, 0.0]),
        jnp.array([1.0, -1.0]),
    ]

    profiles[(EMERGENCY, SUSPENDED)] = [
        jnp.array([0.0, 0.0, 0.0, emerg_target_drive]),
        jnp.array([0.0, 0.0]),
        jnp.array([0.0, 0.0]),
        jnp.array([0.5, -0.5]),
    ]

    return profiles


def swap_C_on_agent(agent, new_C):
    batched_C = [c[None, ...] if c.ndim == 1 else c for c in new_C]
    return eqx.tree_at(lambda a: a.C, agent, batched_C)


def run_single_trial(profiles, condition_id, trial_seed):
    """Run one trial with custom profiles."""
    rng = jr.PRNGKey(trial_seed)

    privacy_schedule, urgency_schedule = get_condition_schedule(condition_id, T)
    A = build_A_matrices()
    B = build_B_matrices()
    D = build_D_priors()
    A_deps = get_A_dependencies()

    # Use first profile for initialization
    first_key = next(iter(profiles))
    C_init = profiles[first_key]

    agent = Agent(
        A=A, B=B, C=C_init, D=D,
        A_dependencies=A_deps,
        B_dependencies=get_B_dependencies(),
        control_fac_idx=[0],
        policy_len=POLICY_LEN,
        inference_algo="fpi",
        num_iter=16,
        action_selection="stochastic",
        sampling_mode="marginal",
        use_utility=True,
        use_states_info_gain=True,
        gamma=GAMMA,
        alpha=16.0,
    )

    rng, env_key = jr.split(rng)
    env = DroneEnv(
        A=A, B=B,
        schedules={1: privacy_schedule, 2: urgency_schedule},
        num_states=[4, 2, 2],
        control_fac_idx=[0],
    )

    true_state = env.reset(D, rng_key=env_key)
    true_state[1] = privacy_schedule[0]
    true_state[2] = urgency_schedule[0]

    action = -jnp.ones((1, 3), dtype=jnp.int32)
    qs = [jnp.expand_dims(d, -2) for d in agent.D]

    positions = []

    for t in range(T):
        rng, obs_key, act_key, step_key = jr.split(rng, 4)

        obs_list = env.generate_observation(true_state, A_deps, obs_key)
        obs_batch = [jnp.array([[int(o)]]) for o in obs_list]

        qs_latest = [q[:, -1, :] for q in qs]
        q_urgency = qs_latest[2][0]
        q_privacy = qs_latest[1][0]

        C_eff = compute_C_effective(
            profiles,
            {"urgency": q_urgency, "privacy": q_privacy},
        )
        agent = swap_C_on_agent(agent, C_eff)

        if jnp.any(action < 0):
            empirical_prior = agent.D
        else:
            empirical_prior, qs = agent.update_empirical_prior(action, qs)

        qs = agent.infer_states(obs_batch, empirical_prior)
        q_pi, G = agent.infer_policies(qs)
        action = agent.sample_action(q_pi, rng_key=jr.split(act_key, 1))
        qs = [q[:, -1:, :] for q in qs]

        positions.append(int(true_state[0]))
        next_state = env.step(true_state, action[0], rng_key=step_key)
        true_state = next_state

    crossed_privacy = any(p >= 2 for p in positions)
    return crossed_privacy


def run_experiment(seed=42, n_trials=N_TRIALS, save_dir=None):
    """Run sensitivity sweep."""
    print("=" * 60)
    print("Sensitivity Analysis")
    print(f"  Emergency target drives: {EMERGENCY_TARGET_DRIVES}")
    print(f"  Normal complaint aversions: {NORMAL_COMPLAINT_AVERSIONS}")
    print(f"  Trials per cell: {n_trials}")
    print("=" * 60)

    results = {}

    for etd in EMERGENCY_TARGET_DRIVES:
        for nca in NORMAL_COMPLAINT_AVERSIONS:
            profiles = build_custom_profiles(etd, nca)
            key = f"etd={etd}_nca={nca}"

            # Run C1 (normal) and C2 (emergency) conditions
            p_cross_normal = 0
            p_cross_emergency = 0

            for trial in range(n_trials):
                trial_seed_c1 = seed * 100000 + int(etd * 100) + int(nca * 10) + trial
                trial_seed_c2 = trial_seed_c1 + 50000

                crossed_c1 = run_single_trial(profiles, 1, trial_seed_c1)
                crossed_c2 = run_single_trial(profiles, 2, trial_seed_c2)

                p_cross_normal += int(crossed_c1)
                p_cross_emergency += int(crossed_c2)

            p_cross_normal /= n_trials
            p_cross_emergency /= n_trials
            override_score = p_cross_emergency - p_cross_normal

            results[key] = {
                "emergency_target_drive": etd,
                "normal_complaint_aversion": nca,
                "p_cross_normal": p_cross_normal,
                "p_cross_emergency": p_cross_emergency,
                "override_score": override_score,
            }

            print(f"  ETD={etd:.1f} NCA={nca:.1f}: "
                  f"P(cross|norm)={p_cross_normal:.2f}  "
                  f"P(cross|emerg)={p_cross_emergency:.2f}  "
                  f"Score={override_score:.2f}")

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / "exp_sensitivity_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {save_dir}")

    return results


if __name__ == "__main__":
    results_dir = Path(__file__).resolve().parents[2] / "results"
    run_experiment(save_dir=results_dir)
