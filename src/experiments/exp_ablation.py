"""
EFE Ablation Experiment — Tests the Core Claim

Hypothesis: The epistemic term drives information-gathering behavior
that is necessary for correct context inference, which in turn is
necessary for correct preference mixing.

4 configs: FULL, PRAGMATIC_ONLY, EPISTEMIC_ONLY, RANDOM
Conditions: C1, C5, C7 — 100 trials, stochastic transitions.

New metric: C_eff tracking error — how fast preferences converge
to the correct profile for the true context.
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
from src.utils.profile_mixing import compute_C_effective, compute_C_eff_tracking_error
from src.utils.stats import bootstrap_ci, mann_whitney_u, cohens_d

T = 10
N_TRIALS = 100
POLICY_LEN = 4
GAMMA = 16.0
CONDITIONS = [1, 5, 7]

# EFE ablation configs
CONFIGS = {
    "FULL": {"use_utility": True, "use_states_info_gain": True},
    "PRAGMATIC_ONLY": {"use_utility": True, "use_states_info_gain": False},
    "EPISTEMIC_ONLY": {"use_utility": False, "use_states_info_gain": True},
    "RANDOM": {"use_utility": False, "use_states_info_gain": False},
}


def create_agent(config_name):
    """Create agent with specified EFE configuration."""
    A = build_A_matrices()
    B = build_B_matrices()
    D = build_D_priors()
    C = build_C_vectors_default()
    cfg = CONFIGS[config_name]

    agent = Agent(
        A=A, B=B, C=C, D=D,
        A_dependencies=get_A_dependencies(),
        B_dependencies=get_B_dependencies(),
        control_fac_idx=[0],
        policy_len=POLICY_LEN,
        inference_algo="fpi",
        num_iter=16,
        action_selection="stochastic",
        sampling_mode="marginal",
        use_utility=cfg["use_utility"],
        use_states_info_gain=cfg["use_states_info_gain"],
        gamma=GAMMA,
        alpha=16.0,
    )
    return agent


def swap_C_on_agent(agent, new_C):
    batched_C = [c[None, ...] if c.ndim == 1 else c for c in new_C]
    return eqx.tree_at(lambda a: a.C, agent, batched_C)


def run_single_trial(config_name, condition_id, trial_seed):
    """Run one trial with specified EFE config and condition."""
    rng = jr.PRNGKey(trial_seed)

    privacy_schedule, urgency_schedule = get_condition_schedule(condition_id, T)
    A = build_A_matrices()
    B = build_B_matrices()
    D = build_D_priors()
    profiles = build_C_profiles()

    agent = create_agent(config_name)

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

    positions = []
    c_eff_tracking_errors = []
    beliefs_urgency = []
    beliefs_privacy = []
    timesteps_at_approach = 0

    for t in range(T):
        rng, obs_key, act_key, step_key = jr.split(rng, 4)

        obs_list = env.generate_observation(true_state, A_deps, obs_key)
        obs_batch = [jnp.array([[int(o)]]) for o in obs_list]

        # Belief-weighted C mixing
        qs_latest = [q[:, -1, :] for q in qs]
        q_urgency = qs_latest[2][0]
        q_privacy = qs_latest[1][0]

        C_eff = compute_C_effective(
            profiles,
            {"urgency": q_urgency, "privacy": q_privacy},
        )
        agent = swap_C_on_agent(agent, C_eff)

        # Tracking error
        true_urg = urgency_schedule[min(t, len(urgency_schedule) - 1)]
        true_priv = privacy_schedule[min(t, len(privacy_schedule) - 1)]
        C_oracle = profiles[(true_urg, true_priv)]
        tracking_error = compute_C_eff_tracking_error(C_eff, C_oracle)
        c_eff_tracking_errors.append(tracking_error)

        # Inference
        if jnp.any(action < 0):
            empirical_prior = agent.D
        else:
            empirical_prior, qs = agent.update_empirical_prior(action, qs)

        qs = agent.infer_states(obs_batch, empirical_prior)
        q_pi, G = agent.infer_policies(qs)
        action = agent.sample_action(q_pi, rng_key=jr.split(act_key, 1))

        qs_latest = [q[:, -1, :] for q in qs]
        positions.append(int(true_state[0]))
        beliefs_urgency.append(float(qs_latest[2][0][1]))
        beliefs_privacy.append(float(qs_latest[1][0][0]))

        if true_state[0] == 1:  # APPROACH
            timesteps_at_approach += 1

        qs = [q[:, -1:, :] for q in qs]
        next_state = env.step(true_state, action[0], rng_key=step_key)
        true_state = next_state

    # Compute metrics
    reached_target = any(p == 3 for p in positions)
    violated_privacy = False
    for t_idx, (pos, ps) in enumerate(zip(positions, privacy_schedule)):
        if pos == 2 and ps == 0:
            violated_privacy = True
            break

    return {
        "config": config_name,
        "condition": condition_id,
        "reached_target": reached_target,
        "violated_privacy": violated_privacy,
        "c_eff_tracking_errors": c_eff_tracking_errors,
        "timesteps_at_approach": timesteps_at_approach,
        "positions": positions,
        "beliefs_urgency": beliefs_urgency,
        "beliefs_privacy": beliefs_privacy,
    }


def run_experiment(seed=42, n_trials=N_TRIALS, save_dir=None):
    """Run ablation across all configs and conditions."""
    print("=" * 60)
    print("EFE Ablation Experiment")
    print(f"  Configs: {list(CONFIGS.keys())}")
    print(f"  Conditions: {CONDITIONS}")
    print(f"  Trials: {n_trials}")
    print("=" * 60)

    all_results = []

    for config_name in CONFIGS:
        for cond in CONDITIONS:
            print(f"\n--- {config_name} × C{cond} ---")
            for trial in range(n_trials):
                trial_seed = seed * 10000 + hash(config_name) % 1000 * 100 + cond * 10 + trial
                result = run_single_trial(config_name, cond, trial_seed)
                all_results.append(result)

            # Quick summary
            cond_results = [r for r in all_results
                            if r["config"] == config_name and r["condition"] == cond]
            success_rate = np.mean([r["reached_target"] for r in cond_results])
            violation_rate = np.mean([r["violated_privacy"] for r in cond_results])
            mean_c_err = np.mean([np.mean(r["c_eff_tracking_errors"]) for r in cond_results])
            print(f"  Success: {success_rate:.2f}  Violations: {violation_rate:.2f}  "
                  f"C_err: {mean_c_err:.3f}")

    # Compute statistical comparisons
    metrics = {}
    for cond in CONDITIONS:
        metrics[cond] = {}
        for config_name in CONFIGS:
            cr = [r for r in all_results
                  if r["config"] == config_name and r["condition"] == cond]
            successes = [r["reached_target"] for r in cr]
            violations = [r["violated_privacy"] for r in cr]
            c_errs = [np.mean(r["c_eff_tracking_errors"]) for r in cr]
            approach_times = [r["timesteps_at_approach"] for r in cr]

            metrics[cond][config_name] = {
                "success_rate": bootstrap_ci(successes),
                "violation_rate": bootstrap_ci(violations),
                "mean_c_eff_error": bootstrap_ci(c_errs),
                "mean_approach_time": bootstrap_ci(approach_times),
            }

        # Statistical tests: FULL vs PRAGMATIC_ONLY
        full_results = [r for r in all_results
                        if r["config"] == "FULL" and r["condition"] == cond]
        prag_results = [r for r in all_results
                        if r["config"] == "PRAGMATIC_ONLY" and r["condition"] == cond]
        if full_results and prag_results:
            full_c_errs = [np.mean(r["c_eff_tracking_errors"]) for r in full_results]
            prag_c_errs = [np.mean(r["c_eff_tracking_errors"]) for r in prag_results]
            U, p = mann_whitney_u(full_c_errs, prag_c_errs)
            d = cohens_d(full_c_errs, prag_c_errs)
            metrics[cond]["full_vs_pragmatic"] = {
                "U": U, "p_value": p, "cohens_d": d,
            }

    # Print summary table
    print("\n" + "=" * 70)
    print("Ablation Summary (success_rate / violation_rate / c_eff_error):")
    print(f"{'Config':<20} {'C1':>18} {'C5':>18} {'C7':>18}")
    print("-" * 76)
    for cfg in CONFIGS:
        row = f"{cfg:<20}"
        for cond in CONDITIONS:
            m = metrics[cond][cfg]
            s = m["success_rate"][0]
            v = m["violation_rate"][0]
            c = m["mean_c_eff_error"][0]
            row += f" {s:.2f}/{v:.2f}/{c:.2f}"
        print(row)

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        # Save metrics (convert tuples to lists for JSON)
        metrics_json = {}
        for cond in metrics:
            metrics_json[str(cond)] = {}
            for key, val in metrics[cond].items():
                if isinstance(val, dict):
                    metrics_json[str(cond)][key] = {
                        k: list(v) if isinstance(v, tuple) else v
                        for k, v in val.items()
                    }
        with open(save_dir / "exp_ablation_metrics.json", "w") as f:
            json.dump(metrics_json, f, indent=2)
        np.savez(
            save_dir / "exp_ablation_raw.npz",
            results=np.array(all_results, dtype=object),
        )
        print(f"\nResults saved to {save_dir}")

    return all_results, metrics


if __name__ == "__main__":
    results_dir = Path(__file__).resolve().parents[2] / "results"
    run_experiment(save_dir=results_dir)
