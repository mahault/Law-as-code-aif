"""
Learning Experiment — Observation Model Acquisition

Training phase (200 steps): learn_A=True, stochastic environment,
uniform A priors. Agent discovers observation model from experience.

Test phase: Frozen learned A. Run C1-C7.
Compare LEARNED vs ORACLE vs MISSPECIFIED.
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
from src.utils.profile_mixing import compute_C_effective
from src.utils.stats import bootstrap_ci

T_TRAIN = 200
T_TEST = 10
N_TRIALS_TEST = 50
POLICY_LEN = 4
GAMMA = 16.0
CONDITIONS = [1, 2, 3, 4, 5, 6, 7]


def swap_C_on_agent(agent, new_C):
    batched_C = [c[None, ...] if c.ndim == 1 else c for c in new_C]
    return eqx.tree_at(lambda a: a.C, agent, batched_C)


def build_uniform_A_priors():
    """Build weakly informative (near-uniform) A priors for learning."""
    # Small concentration parameters — agent starts with little knowledge
    # of the observation model. Shape matches A matrices.
    a0 = jnp.ones((4, 4)) * 1.0           # position obs (4x4)
    a1 = jnp.ones((2, 4, 2)) * 1.0        # privacy cue (2x4x2)
    a2 = jnp.ones((2, 2)) * 1.0           # emergency signal (2x2)
    a3 = jnp.ones((2, 4, 2)) * 1.0        # complaint (2x4x2)
    return [a0, a1, a2, a3]


def build_misspecified_A():
    """Build deliberately misspecified A matrices.

    Swaps privacy cue polarity: ACTIVE shows as SUSPENDED and vice versa.
    """
    A = build_A_matrices()
    # Flip M1 privacy cue along privacy axis
    A[1] = A[1][:, :, ::-1]
    return A


def train_agent(seed=42):
    """Train agent by learning A matrices from stochastic experience.

    Returns the learned A matrices (as concentration parameters).
    """
    rng = jr.PRNGKey(seed)

    A_true = build_A_matrices()
    B = build_B_matrices()
    D = build_D_priors()
    profiles = build_C_profiles()
    A_deps = get_A_dependencies()
    pA = build_uniform_A_priors()

    # Create agent with learning enabled
    agent = Agent(
        A=A_true,  # True A used only for shape; pA drives learning
        B=B,
        C=build_C_vectors_default(),
        D=D,
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
        learn_A=True,
        pA=pA,
    )

    # Stochastic environment with cycling conditions
    # Alternate between different privacy/urgency contexts during training
    train_privacy = [0] * 50 + [1] * 50 + [0] * 50 + [1] * 50
    train_urgency = [0] * 100 + [1] * 100

    rng, env_key = jr.split(rng)
    env = DroneEnv(
        A=A_true, B=B,
        schedules={1: train_privacy, 2: train_urgency},
        num_states=[4, 2, 2],
        control_fac_idx=[0],
        stochastic_uncontrollable=True,
    )

    true_state = env.reset(D, rng_key=env_key)
    true_state[1] = train_privacy[0]
    true_state[2] = train_urgency[0]

    action = -jnp.ones((1, 3), dtype=jnp.int32)
    qs = [jnp.expand_dims(d, -2) for d in agent.D]

    learning_curve = []

    for t in range(T_TRAIN):
        rng, obs_key, act_key, step_key = jr.split(rng, 4)

        obs_list = env.generate_observation(true_state, A_deps, obs_key)
        obs_batch = [jnp.array([[int(o)]]) for o in obs_list]

        # Profile mixing
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

        # Learn A matrices via infer_parameters
        agent = agent.infer_parameters(
            beliefs_A=qs,
            outcomes=obs_batch,
            actions=action,
        )

        # Track learning progress: L1 distance between learned and true A for M2
        pA2 = np.array(agent.pA[2])
        # Remove batch dim if present
        if pA2.ndim > 2:
            pA2 = pA2[0] if pA2.shape[0] == 1 else pA2
        learned_A2_norm = pA2 / (pA2.sum(axis=0, keepdims=True) + 1e-16)
        true_A2 = np.array(A_true[2])
        l1 = float(np.mean(np.abs(learned_A2_norm - true_A2)))
        learning_curve.append(l1)

        qs = [q[:, -1:, :] for q in qs]
        next_state = env.step(true_state, action[0], rng_key=step_key)
        true_state = next_state

    # Extract learned A matrices (normalize concentration params)
    learned_A = []
    for pA_m in agent.pA:
        pA_np = np.array(pA_m)
        # Remove batch dimension if present
        if pA_np.ndim > len(A_true[0].shape):
            pA_np = pA_np[0]
        A_m = pA_np / pA_np.sum(axis=0, keepdims=True)
        learned_A.append(jnp.array(A_m))

    return learned_A, learning_curve


def run_test_phase(A_matrices, agent_label, n_trials, seed=42):
    """Run test phase with given A matrices."""
    B = build_B_matrices()
    D = build_D_priors()
    profiles = build_C_profiles()
    A_deps = get_A_dependencies()
    A_true = build_A_matrices()  # true A for env

    results = {}

    for cond in CONDITIONS:
        successes = []
        violations = []

        for trial in range(n_trials):
            trial_seed = seed * 10000 + cond * 100 + trial
            rng = jr.PRNGKey(trial_seed)

            privacy_schedule, urgency_schedule = get_condition_schedule(cond, T_TEST)

            agent = Agent(
                A=A_matrices, B=B, C=build_C_vectors_default(), D=D,
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
                A=A_true, B=B,
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

            for t in range(T_TEST):
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

            reached = any(p == 3 for p in positions)
            violated = any(
                pos == 2 and privacy_schedule[t_idx] == 0
                for t_idx, pos in enumerate(positions)
            )
            successes.append(reached)
            violations.append(violated)

        results[cond] = {
            "success_rate": bootstrap_ci(successes),
            "violation_rate": bootstrap_ci(violations),
        }

    return results


def run_experiment(seed=42, n_trials=N_TRIALS_TEST, save_dir=None):
    """Run learning experiment."""
    print("=" * 60)
    print("Learning Experiment")
    print(f"  Training: {T_TRAIN} steps")
    print(f"  Test: {T_TEST} steps × {n_trials} trials × 7 conditions")
    print("=" * 60)

    # Phase 1: Train
    print("\n--- Training phase ---")
    learned_A, learning_curve = train_agent(seed=seed)
    print(f"  Training complete. Final L1(A2): {learning_curve[-1]:.4f}")

    # Phase 2: Test with 3 configurations
    oracle_A = build_A_matrices()
    misspecified_A = build_misspecified_A()

    configs = {
        "ORACLE": oracle_A,
        "LEARNED": [jnp.array(a) for a in learned_A],
        "MISSPECIFIED": misspecified_A,
    }

    all_results = {}
    for label, A_mats in configs.items():
        print(f"\n--- Testing: {label} ---")
        all_results[label] = run_test_phase(A_mats, label, n_trials, seed)

        for cond in CONDITIONS:
            m = all_results[label][cond]
            s = m["success_rate"][0]
            v = m["violation_rate"][0]
            print(f"  C{cond}: success={s:.2f}  violation={v:.2f}")

    # Summary table
    print("\n" + "=" * 70)
    print("Learning Experiment Summary (success / violation):")
    header = f"{'Config':<16}"
    for cond in CONDITIONS:
        header += f" {'C' + str(cond):>8}"
    print(header)
    print("-" * 70)
    for label in configs:
        row = f"{label:<16}"
        for cond in CONDITIONS:
            m = all_results[label][cond]
            s = m["success_rate"][0]
            v = m["violation_rate"][0]
            row += f" {s:.1f}/{v:.1f}"
        print(row)

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        results_json = {
            "learning_curve": learning_curve,
        }
        for label in all_results:
            results_json[label] = {}
            for cond in all_results[label]:
                entry = all_results[label][cond]
                results_json[label][str(cond)] = {
                    k: list(v) if isinstance(v, tuple) else v
                    for k, v in entry.items()
                }
        with open(save_dir / "exp_learning_results.json", "w") as f:
            json.dump(results_json, f, indent=2)
        print(f"\nResults saved to {save_dir}")

    return all_results, learning_curve


if __name__ == "__main__":
    results_dir = Path(__file__).resolve().parents[2] / "results"
    run_experiment(save_dir=results_dir)
