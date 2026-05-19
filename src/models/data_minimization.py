"""
Experiment 1: Data Minimization (GDPR Art 5)

State factors (3):
  F0: pipeline_mode      [2] — RAW, ANONYMIZED  (controllable)
  F1: scene_composition  [3] — TARGET_ONLY, TARGET+BYSTANDERS, BYSTANDERS_ONLY
  F2: consent_context    [2] — NO_CONSENT, CONSENT_OBTAINED

Observation modalities (3):
  M0: data_exposure  [3] — NONE, PARTIAL, FULL
  M1: scene_cue      [3] — TARGET_ONLY, MIXED, CROWD
  M2: consent_signal [2] — NO_CONSENT, CONSENT

Actions for F0 (2): SET_RAW=0, SET_ANONYMIZED=1

C Subtensors: 3 preference profiles indexed by scene composition,
blended at runtime via belief-weighted mixing.
"""

import jax.numpy as jnp

# Labels
PIPELINE = ["RAW", "ANONYMIZED"]
SCENE = ["TARGET_ONLY", "TARGET+BYSTANDERS", "BYSTANDERS_ONLY"]
CONSENT = ["NO_CONSENT", "CONSENT_OBTAINED"]
EXPOSURE = ["NONE", "PARTIAL", "FULL"]

SET_RAW = 0
SET_ANONYMIZED = 1

# Scene state indices
TARGET_ONLY = 0
TARGET_BYSTANDERS = 1
BYSTANDERS_ONLY = 2


def build_A_matrices(noise=0.3):
    """Build observation likelihood matrices.

    A[0]: data_exposure | pipeline_mode, scene — shape (3, 2, 3)
    A[1]: scene_cue | scene — shape (3, 3)
    A[2]: consent_signal | consent — shape (2, 2)

    Args:
        noise: ambiguity for scene_cue (default 0.3 for ~70% accuracy,
               requiring genuine inference of scene composition)
    """
    # M0: data_exposure depends on pipeline AND scene
    A0 = jnp.zeros((3, 2, 3))

    # RAW pipeline (mode=0):
    A0 = A0.at[:, 0, 0].set(jnp.array([0.1, 0.7, 0.2]))
    A0 = A0.at[:, 0, 1].set(jnp.array([0.0, 0.1, 0.9]))
    A0 = A0.at[:, 0, 2].set(jnp.array([0.0, 0.05, 0.95]))

    # ANONYMIZED pipeline (mode=1):
    A0 = A0.at[:, 1, 0].set(jnp.array([0.85, 0.14, 0.01]))
    A0 = A0.at[:, 1, 1].set(jnp.array([0.3, 0.65, 0.05]))
    A0 = A0.at[:, 1, 2].set(jnp.array([0.9, 0.09, 0.01]))

    # M1: scene_cue — reduced accuracy so agent must infer under uncertainty
    off_diag = noise / 2
    on_diag = 1.0 - noise
    A1 = jnp.array([
        [on_diag, off_diag, off_diag],
        [off_diag, on_diag, off_diag],
        [off_diag, off_diag, on_diag],
    ])

    # M2: consent_signal — clear observation of consent
    A2 = jnp.array([
        [0.95, 0.05],
        [0.05, 0.95],
    ])

    return [A0, A1, A2]


def build_B_matrices():
    """Build transition matrices.

    B[0]: pipeline_mode — shape (2, 2, 2): directly set by action
    B[1]: scene_composition — shape (3, 3, 1): uncontrollable, stochastic
    B[2]: consent_context — shape (2, 2, 1): uncontrollable, mostly stable
    """
    # B[0]: Pipeline mode is directly set by action
    B0 = jnp.zeros((2, 2, 2))
    B0 = B0.at[0, :, 0].set(1.0)  # action=0 → RAW
    B0 = B0.at[1, :, 1].set(1.0)  # action=1 → ANONYMIZED

    # B[1]: Scene composition changes stochastically (uncontrollable)
    B1 = jnp.array([
        [0.6, 0.2, 0.1],
        [0.3, 0.6, 0.3],
        [0.1, 0.2, 0.6],
    ])[..., None]

    # B[2]: Consent is mostly stable
    B2 = jnp.array([
        [0.95, 0.05],
        [0.05, 0.95],
    ])[..., None]

    return [B0, B1, B2]


def build_C_profiles():
    """Build preference profiles indexed by scene composition.

    Returns dict mapping (scene_idx,) -> [C0, C1, C2].

    3 profiles:
      TARGET_ONLY:       prefer RAW pipeline, strong tracking preference
      TARGET+BYSTANDERS: prefer ANONYMIZED, balance tracking and privacy
      BYSTANDERS_ONLY:   strongly prefer ANONYMIZED, no tracking preference
    """
    profiles = {}

    # TARGET_ONLY: RAW is fine, prioritize tracking
    profiles[(TARGET_ONLY,)] = [
        jnp.array([1.0, 1.5, -1.0]),     # mild exposure tolerance (target data ok)
        jnp.array([1.5, 0.5, -0.5]),     # strong tracking preference
        jnp.array([0.0, 0.0]),           # no consent preference
    ]

    # TARGET+BYSTANDERS: must anonymize, but still track
    profiles[(TARGET_BYSTANDERS,)] = [
        jnp.array([3.0, 0.5, -4.0]),     # strong aversion to FULL exposure
        jnp.array([0.5, 0.5, -0.5]),     # mild tracking preference
        jnp.array([0.0, 0.0]),
    ]

    # BYSTANDERS_ONLY: strongly anonymize, no tracking needed
    profiles[(BYSTANDERS_ONLY,)] = [
        jnp.array([4.0, 0.0, -5.0]),     # very strong aversion to exposure
        jnp.array([0.0, 0.0, 0.0]),      # no tracking preference
        jnp.array([0.0, 0.0]),
    ]

    return profiles


def build_C_vectors_default():
    """Build default C vectors for agent initialization.

    Returns the TARGET+BYSTANDERS profile as a conservative starting point.
    """
    profiles = build_C_profiles()
    return profiles[(TARGET_BYSTANDERS,)]


def build_D_priors():
    D0 = jnp.array([0.5, 0.5])          # uncertain about current pipeline
    D1 = jnp.array([0.33, 0.34, 0.33])  # uncertain about scene
    D2 = jnp.array([0.7, 0.3])          # mostly no consent
    return [D0, D1, D2]


def get_A_dependencies():
    return [
        [0, 1],  # M0: data_exposure depends on pipeline AND scene
        [1],     # M1: scene_cue depends on scene only
        [2],     # M2: consent_signal depends on consent only
    ]


def get_B_dependencies():
    return [[0], [1], [2]]


def build_scene_schedule(T, bystander_density):
    """Build scene composition schedule based on bystander density.

    Higher density → more timesteps with bystanders present.
    """
    p_bystanders = min(bystander_density / 12.0, 0.9)

    schedule = []
    for t in range(T):
        if bystander_density == 0:
            schedule.append(0)
        else:
            r = (t * 7 + bystander_density * 13) % 100 / 100.0
            if r < (1 - p_bystanders):
                schedule.append(0)
            elif r < (1 - p_bystanders * 0.2):
                schedule.append(1)
            else:
                schedule.append(2)
    return schedule
