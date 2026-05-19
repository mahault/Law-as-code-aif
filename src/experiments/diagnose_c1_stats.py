"""Quick C1 violation rate estimate across many seeds."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import jax.numpy as jnp
import jax.random as jr
import numpy as np

from pymdp.agent import Agent
import equinox as eqx
from src.models.emergency_override import (
    build_A_matrices, build_B_matrices, build_C_profiles,
    build_C_vectors_default, build_D_priors,
    get_A_dependencies, get_B_dependencies,
    get_condition_schedule, POSITIONS,
)
from src.environments.drone_env import DroneEnv
from src.utils.profile_mixing import compute_C_effective

T = 10
POLICY_LEN = 4
GAMMA = 16.0


def run_trial(condition_id, seed, B_override=None, D_override=None):
    """Run one trial, return (violated, reached_target)."""
    rng = jr.PRNGKey(seed)
    privacy_schedule, urgency_schedule = get_condition_schedule(condition_id, T)
    A = build_A_matrices()
    B = B_override if B_override is not None else build_B_matrices()
    D = D_override if D_override is not None else build_D_priors()
    profiles = build_C_profiles()

    agent = Agent(
        A=A, B=B, C=build_C_vectors_default(), D=D,
        A_dependencies=get_A_dependencies(),
        B_dependencies=get_B_dependencies(),
        control_fac_idx=[0],
        policy_len=POLICY_LEN,
        inference_algo="fpi", num_iter=16,
        action_selection="stochastic",
        sampling_mode="marginal",
        use_utility=True, use_states_info_gain=True,
        gamma=GAMMA, alpha=16.0,
    )

    rng, env_key = jr.split(rng)
    env = DroneEnv(A=A, B=B, schedules={1: privacy_schedule, 2: urgency_schedule},
                   num_states=[4, 2, 2], control_fac_idx=[0])

    true_state = env.reset(D, rng_key=env_key)
    true_state[1] = privacy_schedule[0]
    true_state[2] = urgency_schedule[0]

    A_deps = get_A_dependencies()
    action = -jnp.ones((1, 3), dtype=jnp.int32)
    qs = [jnp.expand_dims(d, -2) for d in agent.D]

    violated = False
    reached_target = False

    for t in range(T):
        rng, obs_key, act_key, step_key = jr.split(rng, 4)
        obs_list = env.generate_observation(true_state, A_deps, obs_key)
        obs_batch = [jnp.array([[int(o)]]) for o in obs_list]

        qs_latest = [q[:, -1, :] for q in qs]
        q_urgency = qs_latest[2][0]
        q_privacy = qs_latest[1][0]

        C_eff = compute_C_effective(profiles, {"urgency": q_urgency, "privacy": q_privacy})
        agent = eqx.tree_at(lambda a: a.C, agent,
                            [c[None, ...] if c.ndim == 1 else c for c in C_eff])

        if jnp.any(action < 0):
            empirical_prior = agent.D
        else:
            empirical_prior, qs = agent.update_empirical_prior(action, qs)

        qs = agent.infer_states(observations=obs_batch, empirical_prior=empirical_prior)
        q_pi, G = agent.infer_policies(qs)
        action = agent.sample_action(q_pi, rng_key=jr.split(act_key, 1))
        qs = [q[:, -1:, :] for q in qs]

        next_state = env.step(true_state, action[0], rng_key=step_key)
        true_state = next_state

        if true_state[0] == 2 and privacy_schedule[min(t+1, T-1)] == 0:
            violated = True
        if true_state[0] == 3:
            reached_target = True

    return violated, reached_target


def sweep(label, n_trials, conditions, B_override=None, D_override=None):
    """Run n_trials for each condition and report."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  {n_trials} trials per condition")
    print(f"{'='*60}")

    for cond in conditions:
        violations = 0
        successes = 0
        for trial in range(n_trials):
            seed = 10000 * cond + trial
            v, s = run_trial(cond, seed, B_override, D_override)
            violations += v
            successes += s
        print(f"  C{cond}: violations={violations}/{n_trials} ({violations/n_trials:.1%}), "
              f"success={successes}/{n_trials} ({successes/n_trials:.1%})")


if __name__ == "__main__":
    N = 30
    CONDITIONS = [1, 2, 3, 4, 5, 7]

    # Current: B with a_priv=0.01, a_urg=0.02 (just changed)
    print("Testing with NEW B matrix (a_priv=0.01, a_urg=0.02)")
    sweep("New B, D1=[0.5, 0.5]", N, CONDITIONS)

    # With informative D prior for privacy
    D_informative = build_D_priors()
    D_informative[1] = jnp.array([0.75, 0.25])
    sweep("New B, D1=[0.75, 0.25]", N, CONDITIONS, D_override=D_informative)

    # Old B for comparison
    B_old = build_B_matrices(a_priv=0.125, a_urg=0.125)
    sweep("Old B (a=0.125), D1=[0.5, 0.5]", N, CONDITIONS, B_override=B_old)

    # Combined: new B + informative D
    D_informative2 = build_D_priors()
    D_informative2[1] = jnp.array([0.65, 0.35])
    sweep("New B, D1=[0.65, 0.35]", N, CONDITIONS, D_override=D_informative2)
