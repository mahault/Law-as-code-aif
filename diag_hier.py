"""Diagnose the hierarchical-config oracle gap.

Replicates train_agent(seed=42, pA_builder=build_hierarchical_A_priors)
exactly, but additionally records position-visit and (position, privacy)
occupancy histograms, then compares the learned privacy-cue (A1) and
complaint (A3) channels to their true values.
"""
import sys
sys.path.insert(0, '.')
import numpy as np
import jax.numpy as jnp
import jax.random as jr

from src.experiments.exp_learning import (
    build_hierarchical_A_priors, T_TRAIN, POLICY_LEN, GAMMA,
    swap_C_on_agent,
)
from src.models.emergency_override import (
    build_A_matrices, build_B_matrices, build_C_profiles,
    build_C_vectors_default, build_D_priors,
    get_A_dependencies, get_B_dependencies,
)
from src.environments.drone_env import DroneEnv
from src.utils.profile_mixing import compute_C_effective
from pymdp.agent import Agent

seed = 42
rng = jr.PRNGKey(seed)
A_true = build_A_matrices()
B = build_B_matrices()
D = build_D_priors()
profiles = build_C_profiles()
A_deps = get_A_dependencies()
pA = build_hierarchical_A_priors()

agent = Agent(
    A=A_true, B=B, C=build_C_vectors_default(), D=D,
    A_dependencies=A_deps, B_dependencies=get_B_dependencies(),
    control_fac_idx=[0], policy_len=POLICY_LEN, inference_algo="fpi",
    num_iter=16, action_selection="stochastic", sampling_mode="marginal",
    use_utility=True, use_states_info_gain=True, gamma=GAMMA, alpha=16.0,
    learn_A=True, pA=pA,
)

train_privacy = [0] * 50 + [1] * 50 + [0] * 50 + [1] * 50
train_urgency = [0] * 100 + [1] * 100

rng, env_key = jr.split(rng)
env = DroneEnv(
    A=A_true, B=B, schedules={1: train_privacy, 2: train_urgency},
    num_states=[4, 2, 2], control_fac_idx=[0],
    stochastic_uncontrollable=True,
)
true_state = env.reset(D, rng_key=env_key)
true_state[1] = train_privacy[0]
true_state[2] = train_urgency[0]

action = -jnp.ones((1, 3), dtype=jnp.int32)
qs = [jnp.expand_dims(d, -2) for d in agent.D]

pos_hist = np.zeros(4, dtype=int)
pos_priv_hist = np.zeros((4, 2), dtype=int)

for t in range(T_TRAIN):
    rng, obs_key, act_key, step_key = jr.split(rng, 4)
    pos_hist[int(true_state[0])] += 1
    pos_priv_hist[int(true_state[0]), int(true_state[1])] += 1

    obs_list = env.generate_observation(true_state, A_deps, obs_key)
    obs_batch = [jnp.array([[int(o)]]) for o in obs_list]

    qs_latest = [q[:, -1, :] for q in qs]
    C_eff = compute_C_effective(
        profiles,
        {"urgency": qs_latest[2][0], "privacy": qs_latest[1][0]},
    )
    agent = swap_C_on_agent(agent, C_eff)

    if jnp.any(action < 0):
        empirical_prior = agent.D
    else:
        empirical_prior = agent.update_empirical_prior(action, qs)

    qs = agent.infer_states(obs_batch, empirical_prior)
    q_pi, G = agent.infer_policies(qs)
    action = agent.sample_action(q_pi, rng_key=jr.split(act_key, 1))
    agent = agent.infer_parameters(
        beliefs_A=qs, observations=obs_batch, actions=action)
    qs = [q[:, -1:, :] for q in qs]
    true_state = env.step(true_state, action[0], rng_key=step_key)

print("position visits (PATROL, APPROACH, ZONE, TARGET):", pos_hist.tolist())
print("position x privacy visits (rows=pos, cols=[ACTIVE, SUSP]):")
print(pos_priv_hist)

POS = ["PATROL", "APPROACH", "ZONE", "TARGET"]


def norm(pA_m):
    p = np.array(pA_m)
    if p.ndim > 3:
        p = p[0]
    return p / p.sum(axis=0, keepdims=True)


A1_learned, A1_true = norm(agent.pA[1]), np.array(A_true[1])
A3_learned, A3_true = norm(agent.pA[3]), np.array(A_true[3])

print("\nA1 privacy cue: P(cue=ACTIVE_sig | pos, privacy=ACTIVE) learned vs true")
for p in range(4):
    print(f"  {POS[p]:>9}: {A1_learned[0, p, 0]:.3f} vs {A1_true[0, p, 0]:.3f}"
          f"   | privacy=SUSP: {A1_learned[0, p, 1]:.3f} vs {A1_true[0, p, 1]:.3f}")

print("\nA3 complaint: P(complaint=ON | pos, privacy) learned vs true")
for p in range(4):
    print(f"  {POS[p]:>9}: ACTIVE {A3_learned[1, p, 0]:.3f} vs {A3_true[1, p, 0]:.3f}"
          f"   | SUSP {A3_learned[1, p, 1]:.3f} vs {A3_true[1, p, 1]:.3f}")
