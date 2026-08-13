"""
Experiment 1: GDPR Data Minimization

100 trials × 20 timesteps per condition.
3 conditions: Baseline (always RAW), Rule-based (always ANONYMIZED), AIF-LAL.
Vary bystander density: [0, 2, 5, 10].

Uses belief-weighted preference profile mixing (C subtensors):
the agent infers scene composition from noisy observations and blends
preference profiles accordingly, choosing the appropriate pipeline mode
under genuine uncertainty.
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
from src.models.data_minimization import (
    build_A_matrices, build_B_matrices, build_C_profiles,
    build_C_vectors_default, build_D_priors,
    get_A_dependencies, get_B_dependencies, build_scene_schedule,
    PIPELINE, SCENE, EXPOSURE, TARGET_ONLY,
)
from src.environments.drone_env import DroneEnv
from src.utils.profile_mixing import compute_C_effective

T = 20
N_TRIALS = 100
BYSTANDER_DENSITIES = [0, 2, 5, 10]


def run_condition(condition, bystander_density, n_trials, seed=42):
    """Run one condition across multiple trials.

    condition: 'baseline' | 'rule_based' | 'aif_lal'
    """
    A = build_A_matrices()
    B = build_B_matrices()
    D = build_D_priors()
    A_deps = get_A_dependencies()
    B_deps = get_B_dependencies()

    # C profiles for belief-weighted mixing
    profiles = build_C_profiles()

    exposure_counts = []
    nontarget_exposure_counts = []
    tracking_counts = []

    for trial in range(n_trials):
        trial_seed = seed * 10000 + bystander_density * 1000 + trial
        rng = jr.PRNGKey(trial_seed)

        scene_schedule = build_scene_schedule(T, bystander_density)
        consent_schedule = [0] * T

        rng, env_key = jr.split(rng)
        env = DroneEnv(
            A=A, B=B,
            schedules={1: scene_schedule, 2: consent_schedule},
            num_states=[2, 3, 2],
            control_fac_idx=[0],
        )
        true_state = env.reset(D, rng_key=env_key)
        true_state[1] = scene_schedule[0]
        true_state[2] = consent_schedule[0]

        # AIF agent
        if condition == "aif_lal":
            C_default = build_C_vectors_default()
            agent = Agent(
                A=A, B=B, C=C_default, D=D,
                A_dependencies=A_deps,
                B_dependencies=B_deps,
                control_fac_idx=[0],
                policy_len=2,
                gamma=16.0,
                action_selection="stochastic",
                sampling_mode="marginal",
                use_utility=True,
                use_states_info_gain=True,
            )
            qs = [jnp.expand_dims(d, -2) for d in agent.D]
            action = -jnp.ones((1, 3), dtype=jnp.int32)

        n_exposure = 0
        n_nontarget_exposure = 0
        n_tracked = 0

        for t in range(T):
            rng, obs_key, act_key, step_key = jr.split(rng, 4)

            if condition == "baseline":
                pipeline_action = 0
            elif condition == "rule_based":
                pipeline_action = 1
            else:
                # AIF-LAL: agent decides with belief-weighted C mixing
                obs_list = env.generate_observation(true_state, A_deps, obs_key)
                obs_batch = [jnp.array([[int(o)]]) for o in obs_list]

                # Extract scene belief for profile mixing
                qs_latest = [q[:, -1, :] for q in qs]
                q_scene = qs_latest[1][0]  # (3,)

                # Belief-weighted C mixing
                C_eff = compute_C_effective(
                    profiles,
                    {"scene": q_scene},
                )
                batched_C = [c[None, ...] for c in C_eff]
                agent = eqx.tree_at(lambda a: a.C, agent, batched_C)

                if jnp.any(action < 0):
                    emp_prior = agent.D
                else:
                    emp_prior = agent.update_empirical_prior(action, qs)

                qs = agent.infer_states(obs_batch, emp_prior)
                q_pi, G = agent.infer_policies(qs)
                action = agent.sample_action(q_pi, rng_key=jr.split(act_key, 1))
                qs = [q[:, -1:, :] for q in qs]

                pipeline_action = int(action[0, 0])

            # Set pipeline mode in true state
            true_state[0] = pipeline_action

            # Compute exposure from A matrix.
            #
            # Two metrics are recorded. `n_exposure` is the raw rate of FULL
            # exposure events. `n_nontarget_exposure` counts them only when a
            # non-target person is actually in frame, which is what GDPR
            # Art. 5(1)(c) is about: capturing the tracking subject is the
            # lawful purpose of the mission and is not a minimisation
            # violation. The two diverge because the model assigns RAW
            # capture a non-zero FULL-exposure probability even in
            # target-only scenes, so the raw rate is non-zero at zero
            # bystander density.
            scene = true_state[1]
            mode = true_state[0]
            exposure_probs = np.array(A[0][:, mode, scene])
            n_exposure += exposure_probs[2]
            if scene != TARGET_ONLY:
                n_nontarget_exposure += exposure_probs[2]

            # Tracking accuracy
            if scene in [0, 1]:
                if mode == 0:
                    n_tracked += 1.0
                else:
                    n_tracked += 0.85

            # Step env
            next_state = env.step(true_state, [pipeline_action, 0, 0], rng_key=step_key)
            true_state = next_state

        exposure_ratio = n_exposure / T
        tracking_acc = n_tracked / T

        exposure_counts.append(float(exposure_ratio))
        nontarget_exposure_counts.append(float(n_nontarget_exposure / T))
        tracking_counts.append(float(tracking_acc))

    return {
        "condition": condition,
        "bystander_density": bystander_density,
        "exposure_ratio_mean": float(np.mean(exposure_counts)),
        "exposure_ratio_std": float(np.std(exposure_counts)),
        "nontarget_exposure_mean": float(np.mean(nontarget_exposure_counts)),
        "nontarget_exposure_std": float(np.std(nontarget_exposure_counts)),
        "tracking_acc_mean": float(np.mean(tracking_counts)),
        "tracking_acc_std": float(np.std(tracking_counts)),
        "n_trials": n_trials,
    }


def run_experiment(seed=42, n_trials=N_TRIALS, save_dir=None):
    """Run all conditions across all bystander densities."""
    print("=" * 60)
    print("Experiment 1: Data Minimization (GDPR Art 5)")
    print(f"  Densities: {BYSTANDER_DENSITIES}")
    print(f"  Trials per condition: {n_trials}")
    print(f"  Timesteps: {T}")
    print("=" * 60)

    all_results = []
    conditions = ["baseline", "rule_based", "aif_lal"]

    for density in BYSTANDER_DENSITIES:
        print(f"\n--- Bystander density: {density} ---")
        for cond in conditions:
            result = run_condition(cond, density, n_trials, seed)
            all_results.append(result)
            print(f"  {cond:>12}: exposure={result['exposure_ratio_mean']:.3f} "
                  f"tracking={result['tracking_acc_mean']:.3f}")

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / "exp1_results.json", "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {save_dir}")

    return all_results


if __name__ == "__main__":
    results_dir = Path(__file__).resolve().parents[2] / "results"
    run_experiment(save_dir=results_dir)
