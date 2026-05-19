"""
Noise Robustness Experiment

Sweep A-matrix noise: {0.01, 0.05, 0.125, 0.2, 0.3, 0.4, 0.49}
3 agents: AIF, HPM_NOISY, BAYES_RULES
Conditions C1 + C7, 100 trials.

AIF with belief-weighted mixing should be more robust because under
high noise, beliefs are uncertain → C_eff is conservative (blended)
→ fewer reckless decisions.
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
    get_condition_schedule,
)
from src.environments.drone_env import DroneEnv
from src.baselines.agents import HPMNoisyAgent, BayesRulesAgent
from src.utils.profile_mixing import compute_C_effective
from src.utils.stats import bootstrap_ci

T = 10
N_TRIALS = 100
POLICY_LEN = 4
GAMMA = 16.0
NOISE_LEVELS = [0.01, 0.05, 0.125, 0.2, 0.3, 0.4, 0.49]
AGENT_TYPES = ["AIF", "HPM_NOISY", "BAYES_RULES"]
CONDITIONS = [1, 7]


def swap_C_on_agent(agent, new_C):
    batched_C = [c[None, ...] if c.ndim == 1 else c for c in new_C]
    return eqx.tree_at(lambda a: a.C, agent, batched_C)


def run_single_trial(agent_type, condition_id, noise_level, trial_seed):
    """Run one trial with specified noise level."""
    rng = jr.PRNGKey(trial_seed)

    privacy_schedule, urgency_schedule = get_condition_schedule(condition_id, T)
    A = build_A_matrices(noise=noise_level)
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
    elif agent_type == "HPM_NOISY":
        baseline_agent = HPMNoisyAgent()
    elif agent_type == "BAYES_RULES":
        baseline_agent = BayesRulesAgent(A=A, B=B, D=D, A_deps=A_deps)

    positions = []

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
        else:
            pos_action = baseline_agent.select_action(obs_list, true_state[0])

        positions.append(int(true_state[0]))
        full_action = [pos_action, 0, 0]
        next_state = env.step(true_state, full_action, rng_key=step_key)
        true_state = next_state

    reached_target = any(p == 3 for p in positions)
    violated_privacy = any(
        pos == 2 and privacy_schedule[t_idx] == 0
        for t_idx, pos in enumerate(positions)
    )
    # False positive: violated privacy in C1 (should not cross)
    # Success: reached target in C7 (should eventually cross)

    return {
        "reached_target": reached_target,
        "violated_privacy": violated_privacy,
    }


def run_experiment(seed=42, n_trials=N_TRIALS, save_dir=None):
    """Run noise robustness sweep."""
    print("=" * 60)
    print("Noise Robustness Experiment")
    print(f"  Noise levels: {NOISE_LEVELS}")
    print(f"  Agents: {AGENT_TYPES}")
    print(f"  Conditions: {CONDITIONS}")
    print(f"  Trials: {n_trials}")
    print("=" * 60)

    results = {}

    for noise in NOISE_LEVELS:
        results[noise] = {}
        for agent_type in AGENT_TYPES:
            results[noise][agent_type] = {}
            for cond in CONDITIONS:
                successes = []
                violations = []

                for trial in range(n_trials):
                    trial_seed = (seed * 100000 + int(noise * 1000) * 1000
                                  + hash(agent_type) % 100 * 100 + cond * 10 + trial)
                    result = run_single_trial(agent_type, cond, noise, trial_seed)
                    successes.append(result["reached_target"])
                    violations.append(result["violated_privacy"])

                success_ci = bootstrap_ci(successes)
                violation_ci = bootstrap_ci(violations)

                results[noise][agent_type][cond] = {
                    "success_rate": success_ci,
                    "violation_rate": violation_ci,
                }

                print(f"  noise={noise:.2f} {agent_type:<12} C{cond}: "
                      f"success={success_ci[0]:.2f} [{success_ci[1]:.2f}-{success_ci[2]:.2f}] "
                      f"violation={violation_ci[0]:.2f}")

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        # Convert for JSON serialization
        results_json = {}
        for noise in results:
            results_json[str(noise)] = {}
            for agent_type in results[noise]:
                results_json[str(noise)][agent_type] = {}
                for cond in results[noise][agent_type]:
                    entry = results[noise][agent_type][cond]
                    results_json[str(noise)][agent_type][str(cond)] = {
                        k: list(v) if isinstance(v, tuple) else v
                        for k, v in entry.items()
                    }
        with open(save_dir / "exp_noise_results.json", "w") as f:
            json.dump(results_json, f, indent=2)
        print(f"\nResults saved to {save_dir}")

    return results


if __name__ == "__main__":
    results_dir = Path(__file__).resolve().parents[2] / "results"
    run_experiment(save_dir=results_dir)
