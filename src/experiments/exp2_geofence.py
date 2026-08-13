"""
Experiment 2: EASA Geofence Compliance with an emergency exception

50 trials x 30 timesteps per (scenario, controller) cell.

Two scenarios, both with the airspace closing at t=15:
  normal    — urgency stays NORMAL. Entering restricted airspace is a violation.
  emergency — an override is authorised at t=20. Entering is then lawful.

Four controllers: PID-only, a rigid geofence rule, that rule plus an
emergency clause, and the AIF-LAL with belief-weighted profile mixing.

The emergency scenario is what makes this a conflict rather than a
compliance demo: the same action (entering restricted airspace) is a
violation before t=20 and the correct response after it, and the agent
only sees the switch through a noisy dispatch channel.
"""

import sys
import json
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import equinox as eqx
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pymdp.agent import Agent
from src.models.geofence import (
    build_A_matrices, build_B_matrices, build_C_profiles,
    build_C_vectors_default, build_D_priors,
    get_A_dependencies, get_B_dependencies, build_target_schedule,
    ZONES, STAY, ADVANCE, RETREAT,
)
from src.environments.drone_env import DroneEnv
from src.utils.profile_mixing import compute_C_effective

T = 30
N_TRIALS = 50
AIRSPACE_SWITCH = 15     # airspace OPEN -> RESTRICTED
EMERGENCY_ONSET = 20     # override authorised (emergency scenario only)

SCENARIOS = ("normal", "emergency")
CONTROLLERS = ("pid_only", "rule_based", "rule_based_emergency", "aif_lal")


def build_schedules(scenario):
    """Return (target, airspace, urgency) schedules for a scenario."""
    target = build_target_schedule(T)
    airspace = [0] * AIRSPACE_SWITCH + [1] * (T - AIRSPACE_SWITCH)
    if scenario == "emergency":
        urgency = [0] * EMERGENCY_ONSET + [1] * (T - EMERGENCY_ONSET)
    else:
        urgency = [0] * T
    return target, airspace, urgency


def run_cell(scenario, controller, n_trials, seed=42):
    """Run one (scenario, controller) cell across trials."""
    A = build_A_matrices()
    B = build_B_matrices()
    D = build_D_priors()
    A_deps = get_A_dependencies()
    profiles = build_C_profiles()

    incursions, unjustified, tracking, boundary_frac = [], [], [], []
    overrode = []
    zone_over_time = np.zeros((T, 4))
    q_emerg_over_time = np.zeros(T)
    unjust_over_time = np.zeros(T)
    tracking_over_time = np.zeros(T)

    for trial in range(n_trials):
        rng = jr.PRNGKey(seed * 10000 + trial)
        target_schedule, airspace_schedule, urgency_schedule = build_schedules(scenario)

        rng, env_key = jr.split(rng)
        env = DroneEnv(
            A=A, B=B,
            schedules={1: target_schedule, 2: airspace_schedule, 3: urgency_schedule},
            num_states=[4, 3, 2, 2],
            control_fac_idx=[0],
        )
        true_state = env.reset(D, rng_key=env_key)
        true_state[1] = target_schedule[0]
        true_state[2] = airspace_schedule[0]
        true_state[3] = urgency_schedule[0]

        if controller == "aif_lal":
            agent = Agent(
                A=A, B=B, C=build_C_vectors_default(), D=D,
                A_dependencies=A_deps,
                B_dependencies=get_B_dependencies(),
                control_fac_idx=[0],
                policy_len=2,
                gamma=16.0,
                action_selection="stochastic",
                sampling_mode="marginal",
                use_utility=True,
                use_states_info_gain=True,
            )
            qs = [jnp.expand_dims(d, -2) for d in agent.D]
            action = -jnp.ones((1, 4), dtype=jnp.int32)

        n_incur = n_unjust = n_tracked = n_boundary = 0
        did_override = False

        for t in range(T):
            rng, obs_key, act_key, step_key = jr.split(rng, 4)
            drone_zone = true_state[0]
            target_pos = target_schedule[min(t, T - 1)]

            if controller == "pid_only":
                zone_action = (ADVANCE if target_pos > drone_zone
                               else RETREAT if target_pos < drone_zone else STAY)

            elif controller == "rule_based":
                if drone_zone >= 1:
                    zone_action = RETREAT
                elif target_pos > drone_zone:
                    zone_action = ADVANCE
                else:
                    zone_action = STAY

            elif controller == "rule_based_emergency":
                # Same rigid rule, with a dispatch-signal exception. The
                # signal is the same noisy channel the AIF agent sees, so
                # this is a fair comparison rather than an oracle.
                obs = env.generate_observation(true_state, A_deps, obs_key)
                dispatch_on = int(obs[4]) == 1
                if dispatch_on:
                    zone_action = (ADVANCE if target_pos > drone_zone
                                   else RETREAT if target_pos < drone_zone else STAY)
                elif drone_zone >= 1:
                    zone_action = RETREAT
                elif target_pos > drone_zone:
                    zone_action = ADVANCE
                else:
                    zone_action = STAY

            else:
                obs = env.generate_observation(true_state, A_deps, obs_key)
                obs_batch = [jnp.array([[int(o)]]) for o in obs]

                qs_latest = [q[:, -1, :] for q in qs]
                C_eff = compute_C_effective(
                    profiles,
                    {"airspace": qs_latest[2][0], "urgency": qs_latest[3][0]},
                )
                agent = eqx.tree_at(lambda a: a.C, agent,
                                    [c[None, ...] for c in C_eff])

                if jnp.any(action < 0):
                    emp_prior = agent.D
                else:
                    emp_prior = agent.update_empirical_prior(action, qs)

                qs = agent.infer_states(obs_batch, emp_prior)
                q_pi, _ = agent.infer_policies(qs)
                action = agent.sample_action(q_pi, rng_key=jr.split(act_key, 1))
                q_emerg_over_time[t] += float(qs[3][0, -1, 1])
                qs = [q[:, -1:, :] for q in qs]
                zone_action = int(action[0, 0])

            next_state = env.step(true_state, [zone_action, 0, 0, 0], rng_key=step_key)

            zone, airspace, urgency = next_state[0], next_state[2], next_state[3]
            zone_over_time[t, zone] += 1
            if zone == 1:
                n_boundary += 1
            if airspace == 1 and zone >= 2:
                n_incur += 1
                if urgency == 0:
                    n_unjust += 1
                    unjust_over_time[t] += 1
                else:
                    did_override = True
            if abs(zone - next_state[1]) <= 1:
                n_tracked += 1
                tracking_over_time[t] += 1

            true_state = next_state

        incursions.append(n_incur)
        unjustified.append(n_unjust)
        tracking.append(n_tracked / T)
        boundary_frac.append(n_boundary / T)
        overrode.append(1 if did_override else 0)

    return {
        "scenario": scenario,
        "controller": controller,
        "incursions_mean": float(np.mean(incursions)),
        "unjustified_mean": float(np.mean(unjustified)),
        "unjustified_rate": float(np.mean([u > 0 for u in unjustified])),
        "tracking_pct_mean": float(np.mean(tracking)),
        "tracking_pct_std": float(np.std(tracking)),
        "boundary_pct_mean": float(np.mean(boundary_frac)),
        "override_rate": float(np.mean(overrode)),
        "zone_over_time": (zone_over_time / n_trials).tolist(),
        "q_emergency_over_time": (q_emerg_over_time / n_trials).tolist(),
        "unjustified_over_time": (unjust_over_time / n_trials).tolist(),
        "tracking_over_time": (tracking_over_time / n_trials).tolist(),
        "n_trials": n_trials,
    }


def run_experiment(seed=42, n_trials=N_TRIALS, save_dir=None):
    print("=" * 68)
    print("Experiment 2: Geofence Compliance with emergency exception")
    print(f"  Trials per cell: {n_trials}   Timesteps: {T}")
    print(f"  Airspace closes at t={AIRSPACE_SWITCH}, "
          f"override authorised at t={EMERGENCY_ONSET} (emergency scenario)")
    print("=" * 68)

    all_results = []
    for scenario in SCENARIOS:
        print(f"\n########## scenario: {scenario} ##########")
        for controller in CONTROLLERS:
            print(f"\n--- {controller} ---", flush=True)
            r = run_cell(scenario, controller, n_trials, seed)
            all_results.append(r)
            print(f"  unjustified incursions: {r['unjustified_mean']:.2f} "
                  f"(any: {r['unjustified_rate']:.2f})")
            print(f"  total incursions:       {r['incursions_mean']:.2f}")
            print(f"  tracking:               {r['tracking_pct_mean']:.3f}")
            print(f"  time at BOUNDARY:       {r['boundary_pct_mean']:.3f}")
            print(f"  override achieved:      {r['override_rate']:.2f}")

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / "exp2_results.json", "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {save_dir}")

    return all_results


if __name__ == "__main__":
    run_experiment(save_dir=Path(__file__).resolve().parents[2] / "results")
