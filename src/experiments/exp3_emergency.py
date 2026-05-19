"""
Experiment 3: Emergency Override — Context-Dependent Legal Compliance

Runs 7 conditions × N_TRIALS trials, T timesteps each.
For each trial: create agent, run simulation, record trajectories.

Uses belief-weighted preference profile mixing (C subtensors):
at each timestep, compute C_eff from posterior beliefs about urgency
and privacy context, producing smooth preference transitions rather
than discontinuous threshold-based switching.
"""

import sys
import os
import json
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pymdp.agent import Agent
from src.models.emergency_override import (
    build_A_matrices, build_B_matrices, build_C_profiles,
    build_C_vectors_default, build_D_priors,
    get_A_dependencies, get_B_dependencies,
    get_condition_schedule, POSITIONS,
)
from src.environments.drone_env import DroneEnv
from src.utils.profile_mixing import compute_C_effective, compute_C_eff_tracking_error

# ── Experiment Parameters ──
T = 10             # timesteps per trial
N_TRIALS = 50      # trials per condition
POLICY_LEN = 4     # planning horizon (N=4 as in DEM_laws.m)
GAMMA = 16.0       # policy precision
CONDITIONS = [1, 2, 3, 4, 5, 6, 7]


def create_agent(C_vectors):
    """Create a pymdp Agent with the emergency override generative model."""
    A = build_A_matrices()
    B = build_B_matrices()
    D = build_D_priors()

    agent = Agent(
        A=A,
        B=B,
        C=C_vectors,
        D=D,
        A_dependencies=get_A_dependencies(),
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
    return agent


def swap_C_on_agent(agent, new_C):
    """Return a new agent with updated C vectors.

    pymdp 1.0.0 Agent is an equinox Module (immutable). We use eqx.tree_at
    to produce a new agent with modified C.
    """
    import equinox as eqx

    batched_C = [c[None, ...] if c.ndim == 1 else c for c in new_C]
    return eqx.tree_at(lambda a: a.C, agent, batched_C)


def run_single_trial(condition_id, trial_seed):
    """Run one trial for a given condition.

    Returns dict with trajectory data.
    """
    rng = jr.PRNGKey(trial_seed)

    privacy_schedule, urgency_schedule = get_condition_schedule(condition_id, T)
    A = build_A_matrices()
    B = build_B_matrices()
    D = build_D_priors()

    # Build C profiles for belief-weighted mixing
    profiles = build_C_profiles()

    # Start with default C
    agent = create_agent(build_C_vectors_default())

    rng, env_key = jr.split(rng)
    env = DroneEnv(
        A=A, B=B,
        schedules={1: privacy_schedule, 2: urgency_schedule},
        num_states=[4, 2, 2],
        control_fac_idx=[0],
    )

    true_state = env.reset(D, rng_key=env_key)
    # Override uncontrollable initial states from schedule
    true_state[1] = privacy_schedule[0]
    true_state[2] = urgency_schedule[0]

    A_deps = get_A_dependencies()

    # Storage
    positions = []
    actions_taken = []
    observations_all = []
    beliefs_position = []
    beliefs_privacy = []
    beliefs_urgency = []
    gamma_trajectory = []
    efe_values = []
    true_states_log = []
    c_eff_tracking_errors = []

    # Initialize beliefs
    action = -jnp.ones((1, 3), dtype=jnp.int32)
    qs = [jnp.expand_dims(d, -2) for d in agent.D]

    for t in range(T):
        rng, obs_key, act_key, step_key = jr.split(rng, 4)

        # Generate observation from true state
        obs_list = env.generate_observation(true_state, A_deps, obs_key)

        # Format observations for pymdp
        obs_batch = [jnp.array([[int(o)]]) for o in obs_list]

        # Extract current beliefs for profile mixing
        qs_latest = [q[:, -1, :] for q in qs]
        q_urgency = qs_latest[2][0]   # (2,)
        q_privacy = qs_latest[1][0]   # (2,)

        # Belief-weighted C mixing
        C_eff = compute_C_effective(
            profiles,
            {"urgency": q_urgency, "privacy": q_privacy},
        )
        agent = swap_C_on_agent(agent, C_eff)

        # Compute C_eff tracking error against oracle
        true_urg = urgency_schedule[min(t, len(urgency_schedule) - 1)]
        true_priv = privacy_schedule[min(t, len(privacy_schedule) - 1)]
        C_oracle = profiles[(true_urg, true_priv)]
        tracking_error = compute_C_eff_tracking_error(C_eff, C_oracle)
        c_eff_tracking_errors.append(tracking_error)

        # Compute empirical prior
        if jnp.any(action < 0):
            empirical_prior = agent.D
        else:
            empirical_prior, qs = agent.update_empirical_prior(action, qs)

        # Infer hidden states
        qs = agent.infer_states(
            observations=obs_batch,
            empirical_prior=empirical_prior,
        )

        # Infer policies
        q_pi, G = agent.infer_policies(qs)

        # Sample action
        action = agent.sample_action(q_pi, rng_key=jr.split(act_key, 1))

        # Extract info for logging
        qs_latest = [q[:, -1, :] for q in qs]

        pos_action = int(action[0, 0])
        pos_belief = np.array(qs_latest[0][0])
        priv_belief = np.array(qs_latest[1][0])
        urg_belief = np.array(qs_latest[2][0])

        # Approximate gamma (policy precision) from q_pi entropy
        q_pi_np = np.array(q_pi[0])
        q_pi_np = np.clip(q_pi_np, 1e-16, 1.0)
        policy_entropy = -np.sum(q_pi_np * np.log(q_pi_np))
        max_entropy = np.log(len(q_pi_np)) if len(q_pi_np) > 1 else 1.0
        precision = 1.0 - (policy_entropy / max_entropy)

        # Log
        positions.append(int(true_state[0]))
        actions_taken.append(pos_action)
        observations_all.append([int(o) for o in obs_list])
        beliefs_position.append(pos_belief.tolist())
        beliefs_privacy.append(priv_belief.tolist())
        beliefs_urgency.append(urg_belief.tolist())
        gamma_trajectory.append(float(precision))
        efe_values.append(float(np.min(np.array(G[0]))))
        true_states_log.append(list(true_state))

        # Keep only latest belief for next step
        qs = [q[:, -1:, :] for q in qs]

        # Step environment
        next_state = env.step(true_state, action[0], rng_key=step_key)
        true_state = next_state

    return {
        "condition": condition_id,
        "trial_seed": trial_seed,
        "positions": positions,
        "actions": actions_taken,
        "observations": observations_all,
        "beliefs_position": beliefs_position,
        "beliefs_privacy": beliefs_privacy,
        "beliefs_urgency": beliefs_urgency,
        "gamma": gamma_trajectory,
        "efe_min": efe_values,
        "true_states": true_states_log,
        "privacy_schedule": privacy_schedule,
        "urgency_schedule": urgency_schedule,
        "c_eff_tracking_errors": c_eff_tracking_errors,
    }


def compute_metrics(results):
    """Compute summary metrics across all trials for each condition.

    Returns dict: condition_id -> {violation_rate, success_rate, mean_gamma, ...}
    """
    metrics = {}
    for cond in CONDITIONS:
        cond_trials = [r for r in results if r["condition"] == cond]
        n = len(cond_trials)
        if n == 0:
            continue

        violations = 0
        successes = 0
        gammas = []
        tracking_errors = []

        for trial in cond_trials:
            for t_idx, (ts, ps) in enumerate(
                zip(trial["true_states"], trial["privacy_schedule"])
            ):
                if ts[0] == 2 and ps == 0:
                    violations += 1
                    break

            if any(ts[0] == 3 for ts in trial["true_states"]):
                successes += 1

            gammas.extend(trial["gamma"])
            tracking_errors.extend(trial["c_eff_tracking_errors"])

        metrics[cond] = {
            "violation_rate": violations / n,
            "success_rate": successes / n,
            "mean_gamma": float(np.mean(gammas)),
            "mean_tracking_error": float(np.mean(tracking_errors)),
            "n_trials": n,
        }

    return metrics


def run_experiment(seed=42, n_trials=N_TRIALS, save_dir=None):
    """Run all 7 conditions.

    Returns:
        results: list of trial dicts
        metrics: dict of condition -> metric dict
    """
    print("=" * 60)
    print("Experiment 3: Emergency Override")
    print(f"  Conditions: {CONDITIONS}")
    print(f"  Trials per condition: {n_trials}")
    print(f"  Timesteps: {T}, Policy depth: {POLICY_LEN}")
    print("=" * 60)

    all_results = []
    trial_counter = 0

    for cond in CONDITIONS:
        print(f"\n--- Condition {cond} ---")
        for trial in range(n_trials):
            trial_seed = seed * 1000 + cond * 100 + trial
            result = run_single_trial(cond, trial_seed)
            all_results.append(result)
            trial_counter += 1

            if trial == 0:
                print(f"  Trial 0 positions:  {result['positions']}")
                print(f"  Trial 0 actions:    {result['actions']}")
                print(f"  Trial 0 gamma:      {[f'{g:.2f}' for g in result['gamma']]}")

        print(f"  Completed {n_trials} trials for condition {cond}")

    metrics = compute_metrics(all_results)

    print("\n" + "=" * 60)
    print("Summary Metrics:")
    print(f"{'Cond':>6} {'Violations':>12} {'Success':>10} {'Mean_gam':>10} {'C_err':>10}")
    print("-" * 50)
    for cond in CONDITIONS:
        m = metrics[cond]
        print(
            f"{cond:>6} {m['violation_rate']:>12.2f} "
            f"{m['success_rate']:>10.2f} {m['mean_gamma']:>10.3f} "
            f"{m['mean_tracking_error']:>10.3f}"
        )

    # Save results
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        with open(save_dir / "exp3_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        np.savez(
            save_dir / "exp3_raw.npz",
            results=np.array(all_results, dtype=object),
        )
        print(f"\nResults saved to {save_dir}")

    return all_results, metrics


if __name__ == "__main__":
    results_dir = Path(__file__).resolve().parents[2] / "results"
    run_experiment(save_dir=results_dir)
