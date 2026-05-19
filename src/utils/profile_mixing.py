"""
Belief-weighted preference profile mixing for Active Inference agents.

Implements the principled C subtensor approach: define preference profiles
indexed by context states, then compute effective preferences via
belief-weighted mixing at each timestep.

C_eff[m] = sum_k w_k * profiles[k][m]

where w_k = product of marginal beliefs q(factor_value) for each
context factor in the profile key.
"""

import jax.numpy as jnp


def compute_C_effective(profiles, belief_dict):
    """Compute belief-weighted effective preference vectors.

    Args:
        profiles: dict mapping tuples of context-factor indices to C vectors.
            Example: {(0, 0): [C0, C1, C2, C3], (0, 1): [...], ...}
            Keys are tuples of state indices for each context factor.
        belief_dict: dict mapping factor names to belief arrays.
            Example: {"urgency": jnp.array([0.7, 0.3]), "privacy": jnp.array([0.6, 0.4])}
            Order of keys determines order of tuple indices in profiles.

    Returns:
        C_eff: list of jnp arrays, one per modality — belief-weighted blend
            of all profile preferences.
    """
    factor_names = list(belief_dict.keys())
    beliefs = [belief_dict[name] for name in factor_names]

    # Determine number of modalities from first profile
    first_key = next(iter(profiles))
    num_modalities = len(profiles[first_key])

    # Initialize C_eff as zeros matching modality shapes
    C_eff = [jnp.zeros_like(profiles[first_key][m]) for m in range(num_modalities)]

    # Weighted sum over all profiles
    for key, C_profile in profiles.items():
        # Compute weight as product of marginal beliefs
        w = 1.0
        for i, factor_idx in enumerate(key):
            w = w * beliefs[i][factor_idx]

        # Accumulate weighted profile
        for m in range(num_modalities):
            C_eff[m] = C_eff[m] + w * C_profile[m]

    return C_eff


def compute_C_eff_tracking_error(C_eff, C_oracle):
    """Compute L2 norm between effective and oracle C vectors.

    Args:
        C_eff: list of jnp arrays — current effective preferences
        C_oracle: list of jnp arrays — preferences matching true context

    Returns:
        error: float — sum of L2 norms across modalities
    """
    error = 0.0
    for c_eff_m, c_oracle_m in zip(C_eff, C_oracle):
        error += float(jnp.sqrt(jnp.sum((c_eff_m - c_oracle_m) ** 2)))
    return error
