"""Final validation: all 7 conditions, 30 trials each, with fixed model."""

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
    get_condition_schedule,
)
from src.environments.drone_env import DroneEnv
from src.utils.profile_mixing import compute_C_effective

T = 10
POLICY_LEN = 4
GAMMA = 16.0
N = 30


def run_trial(condition_id, seed):
    rng = jr.PRNGKey(seed)
    privacy_schedule, urgency_schedule = get_condition_schedule(condition_id, T)
    A = build_A_matrices()
    B = build_B_matrices()
    D = build_D_priors()
    profiles = build_C_profiles()

    agent = Agent(
        A=A, B=B, C=build_C_vectors_default(), D=D,
        A_dependencies=get_A_dependencies(),
        B_dependencies=get_B_dependencies(),
        control_fac_idx=[0], policy_len=POLICY_LEN,
        inference_algo="fpi", num_iter=16,
        action_selection="stochastic", sampling_mode="marginal",
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
        C_eff = compute_C_effective(
            profiles,
            {"urgency": qs_latest[2][0], "privacy": qs_latest[1][0]},
        )
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


if __name__ == "__main__":
    print(f"Model parameters:")
    B = build_B_matrices()
    D = build_D_priors()
    print(f"  B1 (privacy): a_priv = {1 - float(B[1][0,0,0]):.3f}")
    print(f"  B2 (urgency): a_urg  = {float(B[2][1,0,0]):.3f}")
    print(f"  D1 (privacy prior): {np.array(D[1])}")
    print(f"  D2 (urgency prior): {np.array(D[2])}")
    print(f"  Observation noise: 0.125")
    print(f"  Trials: {N}")
    print()

    print(f"{'Cond':>4} | {'Description':30} | {'Violations':>12} | {'Success':>10}")
    print("-" * 70)

    for cond in [1, 2, 3, 4, 5, 6, 7]:
        violations = 0
        successes = 0
        for trial in range(N):
            seed = 10000 * cond + trial
            v, s = run_trial(cond, seed)
            violations += v
            successes += s

        desc = {
            1: "Normal, Active -> Stay",
            2: "Emergency, Active -> Cross",
            3: "Normal, Suspended -> Cross",
            4: "Emergency, Suspended -> Cross",
            5: "Normal, A->S@t7 -> Stay then cross",
            6: "Emergency, A->S@t7 -> Cross",
            7: "N->E@t4, A->S@t7 -> Override",
        }[cond]

        print(f"  C{cond} | {desc:30} | {violations:>3}/{N} ({violations/N:>5.1%}) | "
              f"{successes:>3}/{N} ({successes/N:>5.1%})")
