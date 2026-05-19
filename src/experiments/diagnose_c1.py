"""Diagnostic: trace a C1 trial step-by-step to understand why violations occur."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from pymdp.agent import Agent
from src.models.emergency_override import (
    build_A_matrices, build_B_matrices, build_C_profiles,
    build_C_vectors_default, build_D_priors,
    get_A_dependencies, get_B_dependencies,
    get_condition_schedule, POSITIONS, PRIVACY, URGENCY,
)
from src.environments.drone_env import DroneEnv
from src.utils.profile_mixing import compute_C_effective

T = 10
POLICY_LEN = 4
GAMMA = 16.0

def run_diagnostic(condition_id=1, seed=42000):
    """Run a single trial with full diagnostic output."""
    rng = jr.PRNGKey(seed)

    privacy_schedule, urgency_schedule = get_condition_schedule(condition_id, T)
    A = build_A_matrices()
    B = build_B_matrices()
    D = build_D_priors()
    profiles = build_C_profiles()

    print(f"=== Condition {condition_id} ===")
    print(f"Privacy schedule: {[PRIVACY[p] for p in privacy_schedule]}")
    print(f"Urgency schedule: {[URGENCY[u] for u in urgency_schedule]}")
    print()

    # Print B matrix transition probs
    print("B1 (privacy transitions):")
    print(f"  P(stay ACTIVE|ACTIVE) = {float(B[1][0,0,0]):.3f}")
    print(f"  P(flip to SUSPENDED|ACTIVE) = {float(B[1][1,0,0]):.3f}")
    print(f"  P(flip to ACTIVE|SUSPENDED) = {float(B[1][0,1,0]):.3f}")
    print(f"  P(stay SUSPENDED|SUSPENDED) = {float(B[1][1,1,0]):.3f}")
    print()

    import equinox as eqx

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
    env = DroneEnv(
        A=A, B=B,
        schedules={1: privacy_schedule, 2: urgency_schedule},
        num_states=[4, 2, 2],
        control_fac_idx=[0],
    )

    true_state = env.reset(D, rng_key=env_key)
    true_state[1] = privacy_schedule[0]
    true_state[2] = urgency_schedule[0]

    A_deps = get_A_dependencies()
    action = -jnp.ones((1, 3), dtype=jnp.int32)
    qs = [jnp.expand_dims(d, -2) for d in agent.D]

    for t in range(T):
        rng, obs_key, act_key, step_key = jr.split(rng, 4)

        # True state
        true_pos = POSITIONS[true_state[0]]
        true_priv = PRIVACY[true_state[1]]
        true_urg = URGENCY[true_state[2]]

        # Generate observation
        obs_list = env.generate_observation(true_state, A_deps, obs_key)
        obs_names = [
            POSITIONS[obs_list[0]],
            PRIVACY[obs_list[1]],
            ["OFF", "ON"][obs_list[2]],
            ["OFF", "ON"][obs_list[3]],
        ]

        # Current beliefs (BEFORE this observation)
        qs_latest = [q[:, -1, :] for q in qs]
        q_urgency = qs_latest[2][0]
        q_privacy = qs_latest[1][0]
        q_position = qs_latest[0][0]

        # C_eff computation
        C_eff = compute_C_effective(profiles, {"urgency": q_urgency, "privacy": q_privacy})
        agent = eqx.tree_at(lambda a: a.C, agent,
                            [c[None, ...] if c.ndim == 1 else c for c in C_eff])

        # Profile weights
        w_na = float(q_urgency[0] * q_privacy[0])
        w_ns = float(q_urgency[0] * q_privacy[1])
        w_ea = float(q_urgency[1] * q_privacy[0])
        w_es = float(q_urgency[1] * q_privacy[1])

        print(f"--- t={t} ---")
        print(f"  True state: pos={true_pos}, priv={true_priv}, urg={true_urg}")
        print(f"  Observation: pos={obs_names[0]}, priv_cue={obs_names[1]}, "
              f"emerg={obs_names[2]}, complaint={obs_names[3]}")
        print(f"  Beliefs (pre-obs): q_priv={np.array(q_privacy).round(3)}, "
              f"q_urg={np.array(q_urgency).round(3)}")
        print(f"  Profile weights: NA={w_na:.3f}, NS={w_ns:.3f}, "
              f"EA={w_ea:.3f}, ES={w_es:.3f}")
        print(f"  C_eff[pos]: {np.array(C_eff[0]).round(3)}")
        print(f"  C_eff[complaint]: {np.array(C_eff[3]).round(3)}")

        # Process observation
        obs_batch = [jnp.array([[int(o)]]) for o in obs_list]

        if jnp.any(action < 0):
            empirical_prior = agent.D
        else:
            empirical_prior, qs = agent.update_empirical_prior(action, qs)

        qs = agent.infer_states(observations=obs_batch, empirical_prior=empirical_prior)

        # Post-observation beliefs
        qs_post = [q[:, -1, :] for q in qs]
        q_priv_post = qs_post[1][0]
        q_urg_post = qs_post[2][0]
        print(f"  Beliefs (post-obs): q_priv={np.array(q_priv_post).round(3)}, "
              f"q_urg={np.array(q_urg_post).round(3)}")

        # Infer policies
        q_pi, G = agent.infer_policies(qs)

        # Show top policies
        q_pi_np = np.array(q_pi[0])
        G_np = np.array(G[0])
        top_idx = np.argsort(q_pi_np)[::-1][:3]

        # Get policy table from agent
        policies = np.array(agent.policies)  # (n_policies, policy_len, n_control_factors)
        print(f"  Top policies (prob | EFE | actions):")
        for idx in top_idx:
            acts = policies[idx, :, 0].tolist()
            act_names = ["HOLD" if a == 0 else "ADV" for a in acts]
            print(f"    p={q_pi_np[idx]:.3f} G={G_np[idx]:.2f} -> {act_names}")

        # Select action
        action = agent.sample_action(q_pi, rng_key=jr.split(act_key, 1))
        act_name = "HOLD" if int(action[0, 0]) == 0 else "ADVANCE"
        print(f"  => Action: {act_name}")

        qs = [q[:, -1:, :] for q in qs]

        # Step environment
        next_state = env.step(true_state, action[0], rng_key=step_key)
        true_state = next_state
        print()

    # Final position
    print(f"Final position: {POSITIONS[true_state[0]]}")
    violated = any(true_state[0] == 2 for _ in [0])  # check all recorded positions
    print(f"Entered privacy zone: check output above for position=PRIVACY_ZONE")


if __name__ == "__main__":
    # Run multiple seeds to see the pattern
    for seed in [42000, 42001, 42002, 42003, 42004]:
        print(f"\n{'='*70}")
        print(f"SEED: {seed}")
        print(f"{'='*70}")
        run_diagnostic(condition_id=1, seed=seed)
        print()
