"""
Baseline agents for comparison against AIF with C subtensors.

HPM_ORACLE:  HierarchicalPrecedenceMatrix with TRUE context state.
HPM_NOISY:   HPM with context from thresholded observations.
BAYES_RULES: Bayesian filtering (same A matrices) + if-then rules on posterior.
"""

import jax.numpy as jnp
import jax.random as jr
import numpy as np

from src.models.emergency_override import (
    build_A_matrices, build_B_matrices, build_D_priors,
    get_A_dependencies, HOLD, ADVANCE,
    NORMAL, EMERGENCY, ACTIVE, SUSPENDED,
)


class HPMOracleAgent:
    """HPM agent with access to TRUE context state.

    Uses hard-coded priority rules with perfect information.
    Strongest possible rule baseline.
    """

    def select_action(self, true_state, **kwargs):
        """Select action given true hidden state.

        Args:
            true_state: list [position, privacy, urgency]

        Returns:
            action: int (HOLD=0 or ADVANCE=1)
        """
        pos, priv, urg = true_state

        # Safety: if at TARGET, stay (mission complete)
        if pos == 3:
            return HOLD

        # Emergency override: if emergency, advance regardless of privacy
        if urg == EMERGENCY:
            return ADVANCE

        # Privacy: if privacy is active and next position would be PRIVACY_ZONE
        if priv == ACTIVE and pos == 1:
            return HOLD  # don't enter privacy zone

        # If privacy is active and in privacy zone, hold (shouldn't be here)
        if priv == ACTIVE and pos == 2:
            return HOLD

        # Default: advance toward target
        return ADVANCE


class HPMNoisyAgent:
    """HPM agent with context inferred from thresholded observations.

    Applies the same priority rules as HPM_ORACLE but uses noisy
    observations rather than true state. Single-observation thresholding.
    """

    def __init__(self):
        self.last_privacy_obs = 0  # ACTIVE
        self.last_urgency_obs = 0  # NORMAL

    def select_action(self, obs_list, true_position, **kwargs):
        """Select action given observations.

        Args:
            obs_list: list of observation indices [pos_obs, priv_cue, emerg_sig, complaint]
            true_position: int — we assume position is directly observable

        Returns:
            action: int (HOLD=0 or ADVANCE=1)
        """
        pos = true_position
        priv_cue = int(obs_list[1])    # 0=ACTIVE, 1=SUSPENDED
        emerg_sig = int(obs_list[2])   # 0=OFF, 1=ON

        # Update context from observations (single-step, no filtering)
        self.last_privacy_obs = priv_cue
        self.last_urgency_obs = emerg_sig

        # Infer context from thresholded observations
        is_emergency = (emerg_sig == 1)
        privacy_active = (priv_cue == 0)

        # Apply same HPM rules
        if pos == 3:
            return HOLD

        if is_emergency:
            return ADVANCE

        if privacy_active and pos == 1:
            return HOLD

        if privacy_active and pos == 2:
            return HOLD

        return ADVANCE


class BayesRulesAgent:
    """Bayesian filtering + if-then rules on posterior beliefs.

    Uses the same A matrices as AIF for Bayesian belief updating,
    but applies hard threshold rules for action selection instead
    of EFE-based policy selection with preference mixing.
    """

    def __init__(self, A=None, B=None, D=None, A_deps=None):
        if A is None:
            A = build_A_matrices()
        if B is None:
            B = build_B_matrices()
        if D is None:
            D = build_D_priors()
        if A_deps is None:
            A_deps = get_A_dependencies()

        self.A = A
        self.B = B
        self.D = D
        self.A_deps = A_deps

        # Initialize beliefs from priors
        self.q_privacy = np.array(D[1])
        self.q_urgency = np.array(D[2])

    def reset(self):
        """Reset beliefs to priors."""
        self.q_privacy = np.array(self.D[1])
        self.q_urgency = np.array(self.D[2])

    def update_beliefs(self, obs_list, position):
        """Bayesian belief update using observation likelihoods.

        Args:
            obs_list: list of observation indices
            position: int — current position (assumed known)
        """
        # Update privacy belief using privacy cue (M1: depends on position, privacy)
        priv_cue = int(obs_list[1])
        likelihood_priv = np.array(self.A[1][priv_cue, position, :])
        self.q_privacy = self.q_privacy * likelihood_priv
        self.q_privacy = self.q_privacy / (self.q_privacy.sum() + 1e-16)

        # Update urgency belief using emergency signal (M2: depends on urgency)
        emerg_sig = int(obs_list[2])
        likelihood_urg = np.array(self.A[2][emerg_sig, :])
        self.q_urgency = self.q_urgency * likelihood_urg
        self.q_urgency = self.q_urgency / (self.q_urgency.sum() + 1e-16)

        # Predict forward using B matrices (time update)
        self.q_privacy = np.array(self.B[1][:, :, 0]) @ self.q_privacy
        self.q_urgency = np.array(self.B[2][:, :, 0]) @ self.q_urgency

    def select_action(self, obs_list, true_position, **kwargs):
        """Bayesian filtering + threshold rules.

        Args:
            obs_list: list of observation indices
            true_position: int — current position

        Returns:
            action: int (HOLD=0 or ADVANCE=1)
        """
        self.update_beliefs(obs_list, true_position)

        pos = true_position
        is_emergency = (self.q_urgency[1] > 0.5)
        privacy_active = (self.q_privacy[0] > 0.5)

        if pos == 3:
            return HOLD

        if is_emergency:
            return ADVANCE

        if privacy_active and pos == 1:
            return HOLD

        if privacy_active and pos == 2:
            return HOLD

        return ADVANCE
