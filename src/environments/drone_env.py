"""
Simulated drone environment — lightweight state-transition wrapper.

No physics, just prescribes uncontrollable context factors and applies
controllable actions via B matrices. Shared across all 4 experiments.
"""

import jax.numpy as jnp
import jax.random as jr


class DroneEnv:
    """Simulated drone environment with legal context.

    The environment manages "true" hidden states. Controllable factors
    are updated via the agent's chosen action through B matrices;
    uncontrollable factors follow a prescribed schedule.
    """

    def __init__(self, A, B, schedules, num_states, control_fac_idx):
        """
        Args:
            A: list of A matrices (no batch dim)
            B: list of B matrices (no batch dim)
            schedules: dict mapping uncontrollable factor index -> list of states per timestep
            num_states: list of int, number of states per factor
            control_fac_idx: list of controllable factor indices
        """
        self.A = A
        self.B = B
        self.schedules = schedules
        self.num_states = num_states
        self.num_factors = len(num_states)
        self.control_fac_idx = control_fac_idx
        self.uncontrol_fac_idx = [
            f for f in range(self.num_factors) if f not in control_fac_idx
        ]

    def reset(self, D):
        """Reset environment to initial state sampled from D priors.

        Args:
            D: list of D arrays (no batch dim), priors over initial states

        Returns:
            true_state: list of int, one per factor
        """
        # Deterministic: take the argmax of each prior
        true_state = [int(jnp.argmax(d)) for d in D]
        self.t = 0
        return true_state

    def step(self, true_state, action):
        """Apply action and advance environment one timestep.

        Controllable factors transition via B[f][:, current, action].
        Uncontrollable factors follow their prescribed schedule.

        Args:
            true_state: list of int, current state per factor
            action: array of shape (num_factors,) with action indices

        Returns:
            next_state: list of int
        """
        self.t += 1
        next_state = list(true_state)

        # Controllable factors: transition via B matrix (deterministic argmax)
        for f in self.control_fac_idx:
            a_f = int(action[f]) if hasattr(action, '__getitem__') else int(action)
            # B[f] shape: (ns, ns, na) — B[f][:, current_state, action]
            transition_probs = self.B[f][:, true_state[f], a_f]
            next_state[f] = int(jnp.argmax(transition_probs))

        # Uncontrollable factors: follow schedule
        for f in self.uncontrol_fac_idx:
            if f in self.schedules and self.t < len(self.schedules[f]):
                next_state[f] = self.schedules[f][self.t]
            # else: stay at current state

        return next_state

    def generate_observation(self, true_state, A_dependencies, rng_key):
        """Sample observations from likelihood given true state.

        Args:
            true_state: list of int, one per factor
            A_dependencies: list of lists, factor deps per modality
            rng_key: JAX PRNG key

        Returns:
            obs: list of int, one observation index per modality
        """
        keys = jr.split(rng_key, len(self.A))
        obs = []
        for m, (A_m, deps) in enumerate(zip(self.A, A_dependencies)):
            # Index into A_m along the factor dimensions
            idx = tuple(true_state[f] for f in deps)
            probs = A_m[(slice(None),) + idx]  # shape (num_obs_m,)
            o = jr.categorical(keys[m], jnp.log(probs + 1e-16))
            obs.append(o)
        return obs
