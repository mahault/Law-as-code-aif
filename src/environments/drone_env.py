"""
Simulated drone environment — lightweight state-transition wrapper.

No physics, just prescribes uncontrollable context factors and applies
controllable actions via B matrices. Shared across all experiments.
"""

import jax.numpy as jnp
import jax.random as jr


class DroneEnv:
    """Simulated drone environment with legal context.

    The environment manages "true" hidden states. Controllable factors
    are updated via the agent's chosen action through B matrices;
    uncontrollable factors follow a prescribed schedule or evolve
    stochastically via their B matrices.
    """

    def __init__(self, A, B, schedules, num_states, control_fac_idx,
                 stochastic_uncontrollable=False):
        """
        Args:
            A: list of A matrices (no batch dim)
            B: list of B matrices (no batch dim)
            schedules: dict mapping uncontrollable factor index -> list of states per timestep
            num_states: list of int, number of states per factor
            control_fac_idx: list of controllable factor indices
            stochastic_uncontrollable: if True, uncontrollable factors evolve via
                B matrices instead of following fixed schedules
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
        self.stochastic_uncontrollable = stochastic_uncontrollable

    def reset(self, D, rng_key=None):
        """Reset environment to initial state sampled from D priors.

        Args:
            D: list of D arrays (no batch dim), priors over initial states
            rng_key: JAX PRNG key. If None, uses deterministic argmax (legacy).

        Returns:
            true_state: list of int, one per factor
        """
        if rng_key is not None:
            keys = jr.split(rng_key, len(D))
            true_state = [
                int(jr.categorical(keys[f], jnp.log(d + 1e-16)))
                for f, d in enumerate(D)
            ]
        else:
            true_state = [int(jnp.argmax(d)) for d in D]
        self.t = 0
        return true_state

    def step(self, true_state, action, rng_key=None):
        """Apply action and advance environment one timestep.

        Controllable factors transition via B[f][:, current, action].
        Uncontrollable factors follow their prescribed schedule or evolve
        stochastically (if stochastic_uncontrollable=True).

        Args:
            true_state: list of int, current state per factor
            action: array of shape (num_factors,) with action indices
            rng_key: JAX PRNG key. If None, uses deterministic argmax (legacy).

        Returns:
            next_state: list of int
        """
        self.t += 1
        next_state = list(true_state)

        if rng_key is not None:
            keys = jr.split(rng_key, self.num_factors)
        else:
            keys = [None] * self.num_factors

        # Controllable factors: transition via B matrix
        for f in self.control_fac_idx:
            a_f = int(action[f]) if hasattr(action, '__getitem__') else int(action)
            transition_probs = self.B[f][:, true_state[f], a_f]
            if keys[f] is not None:
                next_state[f] = int(jr.categorical(keys[f], jnp.log(transition_probs + 1e-16)))
            else:
                next_state[f] = int(jnp.argmax(transition_probs))

        # Uncontrollable factors
        for f in self.uncontrol_fac_idx:
            if self.stochastic_uncontrollable and keys[f] is not None:
                # Evolve via B matrix stochastically
                transition_probs = self.B[f][:, true_state[f], 0]
                next_state[f] = int(jr.categorical(keys[f], jnp.log(transition_probs + 1e-16)))
            elif f in self.schedules and self.t < len(self.schedules[f]):
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
            idx = tuple(true_state[f] for f in deps)
            probs = A_m[(slice(None),) + idx]
            o = jr.categorical(keys[m], jnp.log(probs + 1e-16))
            obs.append(o)
        return obs
