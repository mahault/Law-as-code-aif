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
"""

import jax.numpy as jnp

# Labels
PIPELINE = ["RAW", "ANONYMIZED"]
SCENE = ["TARGET_ONLY", "TARGET+BYSTANDERS", "BYSTANDERS_ONLY"]
CONSENT = ["NO_CONSENT", "CONSENT_OBTAINED"]
EXPOSURE = ["NONE", "PARTIAL", "FULL"]

SET_RAW = 0
SET_ANONYMIZED = 1


def build_A_matrices():
    """Build observation likelihood matrices.

    A[0]: data_exposure | pipeline_mode, scene — shape (3, 2, 3)
    A[1]: scene_cue | scene — shape (3, 3)
    A[2]: consent_signal | consent — shape (2, 2)
    """
    # M0: data_exposure depends on pipeline AND scene
    # RAW pipeline: exposure depends on scene composition
    # ANONYMIZED pipeline: exposure is always NONE or PARTIAL
    A0 = jnp.zeros((3, 2, 3))

    # RAW pipeline (mode=0):
    #   TARGET_ONLY: mostly PARTIAL (some target data)
    A0 = A0.at[:, 0, 0].set(jnp.array([0.1, 0.7, 0.2]))
    #   TARGET+BYSTANDERS: FULL exposure (bystander biometrics transmitted)
    A0 = A0.at[:, 0, 1].set(jnp.array([0.0, 0.1, 0.9]))
    #   BYSTANDERS_ONLY: FULL exposure
    A0 = A0.at[:, 0, 2].set(jnp.array([0.0, 0.05, 0.95]))

    # ANONYMIZED pipeline (mode=1):
    #   TARGET_ONLY: NONE (no non-target data to leak)
    A0 = A0.at[:, 1, 0].set(jnp.array([0.85, 0.14, 0.01]))
    #   TARGET+BYSTANDERS: PARTIAL (target tracked, bystanders blurred)
    A0 = A0.at[:, 1, 1].set(jnp.array([0.3, 0.65, 0.05]))
    #   BYSTANDERS_ONLY: NONE (all faces blurred)
    A0 = A0.at[:, 1, 2].set(jnp.array([0.9, 0.09, 0.01]))

    # M1: scene_cue — near-identity observation of scene
    A1 = jnp.array([
        [0.9, 0.05, 0.05],
        [0.05, 0.9, 0.05],
        [0.05, 0.05, 0.9],
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
    ])[..., None]  # (3, 3, 1)

    # B[2]: Consent is mostly stable
    B2 = jnp.array([
        [0.95, 0.05],
        [0.05, 0.95],
    ])[..., None]  # (2, 2, 1)

    return [B0, B1, B2]


def build_C_vectors():
    """Preferences: minimize exposure while maintaining tracking.

    C[0]: Strong preference for NONE/PARTIAL exposure, aversion to FULL
    C[1]: Mild preference for TARGET_ONLY (good tracking, no bystanders)
    C[2]: No preference on consent signal
    """
    C0 = jnp.array([3.0, 1.0, -4.0])   # NONE > PARTIAL >> FULL
    C1 = jnp.array([1.0, 0.5, -0.5])    # mild tracking preference
    C2 = jnp.zeros(2)
    return [C0, C1, C2]


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
    import jax.random as jr

    # Probability of bystanders present increases with density
    p_bystanders = min(bystander_density / 12.0, 0.9)

    # Generate schedule: 0=TARGET_ONLY, 1=TARGET+BYSTANDERS, 2=BYSTANDERS_ONLY
    schedule = []
    for t in range(T):
        if bystander_density == 0:
            schedule.append(0)  # always target only
        else:
            # Stochastic assignment based on density
            r = (t * 7 + bystander_density * 13) % 100 / 100.0
            if r < (1 - p_bystanders):
                schedule.append(0)
            elif r < (1 - p_bystanders * 0.2):
                schedule.append(1)
            else:
                schedule.append(2)
    return schedule
