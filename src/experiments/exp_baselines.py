"""
Baselines Experiment — AIF vs Rule-Based Agents

4 agents: AIF, HPM_ORACLE, HPM_NOISY, BAYES_RULES
All 7 conditions, 100 trials, stochastic transitions.

Key comparison: AIF uses continuous belief-weighted preference mixing.
BAYES_RULES uses the same beliefs but applies hard threshold rules.
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
    build_A_matrices, build_B_matrices, build_C_profiles,
    build_C_vectors_default, build_D_priors,
    get_A_dependencies, get_B_dependencies,
    get_condition_schedule, POSITIONS,
)
from src.environments.drone_env import DroneEnv
from src.baselines.agents import HPMOracleAgent, HPMNoisyAgent, BayesRulesAgent
from src.utils.profile_mixing import compute_C_effective
from src.utils.stats import bootstrap_ci, mann_whitney_u, cohens_d

T = 10
N_TRIALS = 100
POLICY_LEN = 4
GAMMA = 16.0
CONDITIONS = [1, 2, 3, 4, 5, 6, 7]
AGENT_TYPES = ["AIF", "HPM_ORACLE", "HPM_NOISY", "BAYES_RULES"]


def swap_C_on_agent(agent, new_C):
    batched_C = [c[None, ...] if c.ndim == 1 else c for c in new_C]
    return eqx.tree_at(lambda a: a.C, agent, batched_C)


def run_single_trial(agent_type, condition_id, trial_seed):
    """Run one trial with specified agent type and condition."""
    rng = jr.PRNGKey(trial_seed)

    privacy_schedule, urgency_schedule = get_condition_schedule(condition_id, T)
    A = build_A_matrices()
    B = build_B_matrices()
    D = build_D_priors()
    profiles = build_C_profiles()
    A_deps = get_A_dependencies()

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

    # Initialize agent
    if agent_type == "AIF":
        agent = Agent(
            A=A, B=B, C=build_C_vectors_default(), D=D,
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
        qs = [jnp.expand_dims(d, -2) for d in agent.D]
        action = -jnp.ones((1, 3), dtype=jnp.int32)
    elif agent_type == "HPM_ORACLE":
        baseline_agent = HPMOracleAgent()
    elif agent_type == "HPM_NOISY":
        baseline_agent = HPMNoisyAgent()
    elif agent_type == "BAYES_RULES":
        baseline_agent = BayesRulesAgent(A=A, B=B, D=D, A_deps=A_deps)

    positions = []
    actions_taken = []

    for t in range(T):
        rng, obs_key, act_key, step_key = jr.split(rng, 4)

        obs_list = env.generate_observation(true_state, A_deps, obs_key)

        if agent_type == "AIF":
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

            pos_action = int(action[0, 0])

        elif agent_type == "HPM_ORACLE":
            pos_action = baseline_agent.select_action(true_state)

        elif agent_type == "HPM_NOISY":
            pos_action = baseline_agent.select_action(obs_list, true_state[0])

        elif agent_type == "BAYES_RULES":
            pos_action = baseline_agent.select_action(obs_list, true_state[0])

        positions.append(int(true_state[0]))
        actions_taken.append(pos_action)

        full_action = [pos_action, 0, 0]
        next_state = env.step(true_state, full_action, rng_key=step_key)
        true_state = next_state

    # Metrics
    reached_target = any(p == 3 for p in positions)
    violated_privacy = False
    for t_idx, (pos, ps) in enumerate(zip(positions, privacy_schedule)):
        if pos == 2 and ps == 0:
            violated_privacy = True
            break

    return {
        "agent_type": agent_type,
        "condition": condition_id,
        "reached_target": reached_target,
        "violated_privacy": violated_privacy,
        "positions": positions,
        "actions": actions_taken,
    }


def run_experiment(seed=42, n_trials=N_TRIALS, save_dir=None):
    """Run all agent types across all conditions."""
    print("=" * 60)
    print("Baselines Experiment")
    print(f"  Agents: {AGENT_TYPES}")
    print(f"  Conditions: {CONDITIONS}")
    print(f"  Trials: {n_trials}")
    print("=" * 60)

    all_results = []

    for agent_type in AGENT_TYPES:
        for cond in CONDITIONS:
            print(f"\n--- {agent_type} × C{cond} ---", end="")
            for trial in range(n_trials):
                trial_seed = seed * 10000 + hash(agent_type) % 1000 * 100 + cond * 10 + trial
                result = run_single_trial(agent_type, cond, trial_seed)
                all_results.append(result)

            cond_results = [r for r in all_results
                            if r["agent_type"] == agent_type and r["condition"] == cond]
            s = np.mean([r["reached_target"] for r in cond_results])
            v = np.mean([r["violated_privacy"] for r in cond_results])
            print(f"  success={s:.2f}  violation={v:.2f}")

    # Compute metrics
    metrics = {}
    for cond in CONDITIONS:
        metrics[cond] = {}
        for agent_type in AGENT_TYPES:
            cr = [r for r in all_results
                  if r["agent_type"] == agent_type and r["condition"] == cond]
            successes = [r["reached_target"] for r in cr]
            violations = [r["violated_privacy"] for r in cr]
            metrics[cond][agent_type] = {
                "success_rate": bootstrap_ci(successes),
                "violation_rate": bootstrap_ci(violations),
            }

    # Summary table
    print("\n" + "=" * 80)
    print("Baselines Summary (success / violation):")
    header = f"{'Agent':<16}"
    for cond in CONDITIONS:
        header += f" {'C' + str(cond):>8}"
    print(header)
    print("-" * 80)
    for agent_type in AGENT_TYPES:
        row = f"{agent_type:<16}"
        for cond in CONDITIONS:
            m = metrics[cond][agent_type]
            s = m["success_rate"][0]
            v = m["violation_rate"][0]
            row += f" {s:.1f}/{v:.1f}"
        print(row)

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        metrics_json = {}
        for cond in metrics:
            metrics_json[str(cond)] = {}
            for agent_type, val in metrics[cond].items():
                metrics_json[str(cond)][agent_type] = {
                    k: list(v) if isinstance(v, tuple) else v
                    for k, v in val.items()
                }
        with open(save_dir / "exp_baselines_metrics.json", "w") as f:
            json.dump(metrics_json, f, indent=2)
        np.savez(
            save_dir / "exp_baselines_raw.npz",
            results=np.array(all_results, dtype=object),
        )
        print(f"\nResults saved to {save_dir}")

    return all_results, metrics


if __name__ == "__main__":
    results_dir = Path(__file__).resolve().parents[2] / "results"
    run_experiment(save_dir=results_dir)
