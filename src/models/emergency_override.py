"""
Experiment 3: Emergency Override — Context-Dependent Legal Compliance

Direct port of DEM_laws.m to autonomous drone domain using pymdp JAX 1.0.0.

State factors (3):
  F0: drone_position [4] — PATROL, APPROACH, PRIVACY_ZONE, TARGET  (controllable)
  F1: privacy_context [2] — ACTIVE, SUSPENDED                      (uncontrollable)
  F2: urgency_context [2] — NORMAL, EMERGENCY                      (uncontrollable)

Observation modalities (4):
  M0: position_obs   [4] — precise drone location
  M1: privacy_cue    [2] — PRIVACY_ACTIVE, PRIVACY_SUSPENDED
  M2: emergency_signal[2] — OFF, ON
  M3: complaint_signal[2] — OFF, ON

Actions for F0 (2): HOLD=0, ADVANCE=1

C Subtensors: 4 preference profiles indexed by (urgency, privacy),
blended at runtime via belief-weighted mixing.
"""

import jax.numpy as jnp
from jax.nn import softmax

# ── State factor labels ──
POSITIONS = ["PATROL", "APPROACH", "PRIVACY_ZONE", "TARGET"]
PRIVACY = ["ACTIVE", "SUSPENDED"]
URGENCY = ["NORMAL", "EMERGENCY"]

# Actions
HOLD = 0
ADVANCE = 1

# Context state indices
NORMAL = 0
EMERGENCY = 1
ACTIVE = 0
SUSPENDED = 1


def build_A_matrices(noise=0.125):
    """Build observation likelihood matrices A[m].

    Returns list of 4 arrays (no batch dimension — pymdp adds it).

    A[0]: position_obs | position        — shape (4, 4)
    A[1]: privacy_cue  | position, privacy — shape (2, 4, 2)
    A[2]: emergency_signal | urgency     — shape (2, 2)
    A[3]: complaint_signal | position, privacy — shape (2, 4, 2)

    Args:
        noise: ambiguity parameter (default 0.125 = 1/8, matching DEM_laws.m)
    """
    a = noise

    # M0: position_obs — 1-to-1 mapping (identity)
    A0 = jnp.eye(4)

    # M1: privacy_cue depends on (position, privacy)
    # At positions 1,2: precise about privacy state
    # At positions 0,3: ambiguous
    A1_full = jnp.zeros((2, 4, 2))
    # Position 0 (PATROL): ambiguous about privacy
    A1_full = A1_full.at[:, 0, 0].set(jnp.array([0.5, 0.5]))
    A1_full = A1_full.at[:, 0, 1].set(jnp.array([0.5, 0.5]))
    # Position 1 (APPROACH): precise about privacy
    A1_full = A1_full.at[:, 1, 0].set(jnp.array([1 - a, a]))
    A1_full = A1_full.at[:, 1, 1].set(jnp.array([a, 1 - a]))
    # Position 2 (PRIVACY_ZONE): precise about privacy
    A1_full = A1_full.at[:, 2, 0].set(jnp.array([1 - a, a]))
    A1_full = A1_full.at[:, 2, 1].set(jnp.array([a, 1 - a]))
    # Position 3 (TARGET): ambiguous
    A1_full = A1_full.at[:, 3, 0].set(jnp.array([0.5, 0.5]))
    A1_full = A1_full.at[:, 3, 1].set(jnp.array([0.5, 0.5]))

    # M2: emergency_signal — heard everywhere with slight ambiguity
    A2 = jnp.array([
        [1 - a, a],
        [a, 1 - a],
    ])

    # M3: complaint_signal — ON when at PRIVACY_ZONE (pos=2) AND privacy=ACTIVE
    A3 = jnp.zeros((2, 4, 2))
    for pos in range(4):
        for priv in range(2):
            if pos == 2 and priv == 0:  # PRIVACY_ZONE + ACTIVE
                A3 = A3.at[0, pos, priv].set(a)
                A3 = A3.at[1, pos, priv].set(1 - a)
            else:
                A3 = A3.at[0, pos, priv].set(1 - a)
                A3 = A3.at[1, pos, priv].set(a)

    return [A0, A1_full, A2, A3]


def build_B_matrices(a_priv=0.01, a_urg=0.02):
    """Build transition matrices B[f].

    B[0]: position — shape (4, 4, 2): HOLD=identity, ADVANCE=shift forward
    B[1]: privacy  — shape (2, 2, 1): slow switching (uncontrollable)
    B[2]: urgency  — shape (2, 2, 1): absorbing once emergency

    Args:
        a_priv: privacy context transition rate. Low values (0.01) model
            privacy zones as stable legal designations that rarely change.
            Higher values make beliefs volatile and noise-sensitive.
        a_urg: urgency context transition rate (NORMAL→EMERGENCY only).
            Low values (0.02) model emergencies as rare events. Once in
            EMERGENCY, the state is absorbing (cannot return to NORMAL).
    """
    # B[0]: Position transitions
    B0_hold = jnp.eye(4)
    B0_advance = jnp.zeros((4, 4))
    B0_advance = B0_advance.at[1, 0].set(1.0)  # PATROL → APPROACH
    B0_advance = B0_advance.at[2, 1].set(1.0)  # APPROACH → PRIVACY_ZONE
    B0_advance = B0_advance.at[3, 2].set(1.0)  # PRIVACY_ZONE → TARGET
    B0_advance = B0_advance.at[2, 3].set(1.0)  # TARGET → PRIVACY_ZONE (wrap back)

    B0 = jnp.stack([B0_hold, B0_advance], axis=-1)

    # B[1]: Privacy context — stable, rare switching
    B1 = jnp.array([
        [1 - a_priv, a_priv],
        [a_priv, 1 - a_priv],
    ])[..., None]

    # B[2]: Urgency context — absorbing once emergency
    B2 = jnp.array([
        [1 - a_urg, 0],
        [a_urg, 1],
    ])[..., None]

    return [B0, B1, B2]


def build_C_profiles():
    """Build preference profiles indexed by (urgency, privacy).

    Returns dict mapping (urgency_idx, privacy_idx) -> [C0, C1, C2, C3].

    4 profiles:
      (NORMAL, ACTIVE):     weak target drive, strong complaint aversion
      (NORMAL, SUSPENDED):  moderate target drive, mild complaint aversion
      (EMERGENCY, ACTIVE):  strong target drive, reduced complaint aversion
      (EMERGENCY, SUSPENDED): strong target drive, minimal complaint aversion
    """
    profiles = {}

    # Normal urgency, Active privacy
    # C0: Strong privacy-zone aversion (-6.0) and negative target drive (-2.0)
    # makes the agent prefer staying at APPROACH or PATROL rather than
    # advancing through the privacy zone. Combined with strong complaint
    # aversion (8.0), the agent avoids the privacy zone entirely.
    # The position preferences encode: "under active privacy + normal urgency,
    # do NOT advance past APPROACH."
    profiles[(NORMAL, ACTIVE)] = [
        jnp.array([0.0, 1.0, -6.0, -2.0]),    # prefer APPROACH, strong privacy-zone aversion
        jnp.array([0.0, 0.0]),                  # no privacy-cue preference
        jnp.array([0.0, 0.0]),                  # no emergency-signal preference
        jnp.array([8.0, -8.0]),                 # very strong complaint aversion
    ]

    # Normal urgency, Suspended privacy
    # With privacy suspended, the agent can freely advance to TARGET.
    profiles[(NORMAL, SUSPENDED)] = [
        jnp.array([0.0, 0.0, 0.0, 3.0]),      # strong target drive
        jnp.array([0.0, 0.0]),
        jnp.array([0.0, 0.0]),
        jnp.array([0.5, -0.5]),                # mild complaint aversion
    ]

    # Emergency, Active privacy
    # Emergency overrides privacy: strong target drive (6.0) dominates
    # even with some complaint aversion (1.0). The agent advances through
    # the privacy zone because the emergency justifies the override.
    profiles[(EMERGENCY, ACTIVE)] = [
        jnp.array([0.0, 0.0, 0.0, 6.0]),      # very strong target drive
        jnp.array([0.0, 0.0]),
        jnp.array([0.0, 0.0]),
        jnp.array([1.0, -1.0]),                # reduced complaint aversion
    ]

    # Emergency, Suspended privacy
    profiles[(EMERGENCY, SUSPENDED)] = [
        jnp.array([0.0, 0.0, 0.0, 6.0]),      # very strong target drive
        jnp.array([0.0, 0.0]),
        jnp.array([0.0, 0.0]),
        jnp.array([0.5, -0.5]),                # minimal complaint aversion
    ]

    return profiles


def build_C_vectors_default():
    """Build default C vectors for agent initialization.

    Returns the normal-active profile as a reasonable starting point.
    The actual effective C is computed via profile mixing at each timestep.
    """
    profiles = build_C_profiles()
    return profiles[(NORMAL, ACTIVE)]


def build_D_priors():
    """Build initial state priors D[f].

    D1 uses a modest prior favouring active privacy [0.75, 0.25].
    This reflects that privacy restrictions are the default in most
    drone operating areas, making the agent appropriately cautious
    before it gathers direct evidence about privacy status.
    """
    D0 = jnp.array([1.0, 0.0, 0.0, 0.0])  # start at PATROL
    D1 = jnp.array([0.75, 0.25])           # prior: privacy likely active
    D2 = jnp.array([7 / 8, 1 / 8])         # mostly expect normal
    return [D0, D1, D2]


def get_A_dependencies():
    """Sparse dependency structure for A matrices."""
    return [
        [0],      # M0: position_obs depends on position only
        [0, 1],   # M1: privacy_cue depends on position AND privacy
        [2],      # M2: emergency_signal depends on urgency only
        [0, 1],   # M3: complaint depends on position AND privacy
    ]


def get_B_dependencies():
    """Each factor's dynamics depend only on itself."""
    return [[0], [1], [2]]


def get_condition_schedule(condition_id, T=10):
    """Return (privacy_schedule, urgency_schedule) for the given condition.

    Each schedule is a list of T state indices.

    7 Conditions (matching DEM_laws.m):
      C1: All ACTIVE,    All NORMAL       — Stay at approach
      C2: All ACTIVE,    All EMERGENCY    — Cross despite complaint
      C3: All SUSPENDED, All NORMAL       — Cross freely
      C4: All SUSPENDED, All EMERGENCY    — Cross immediately
      C5: ACTIVE→SUSPENDED at t=7, All NORMAL    — Wait, then cross
      C6: ACTIVE→SUSPENDED at t=7, All EMERGENCY — Cross early
      C7: ACTIVE→SUSPENDED at t=7, NORMAL→EMERGENCY at t=4 — Key test
    """
    switch_privacy = 7
    switch_urgency = 4

    if condition_id == 1:
        privacy = [0] * T
        urgency = [0] * T
    elif condition_id == 2:
        privacy = [0] * T
        urgency = [1] * T
    elif condition_id == 3:
        privacy = [1] * T
        urgency = [0] * T
    elif condition_id == 4:
        privacy = [1] * T
        urgency = [1] * T
    elif condition_id == 5:
        privacy = [0] * switch_privacy + [1] * (T - switch_privacy)
        urgency = [0] * T
    elif condition_id == 6:
        privacy = [0] * switch_privacy + [1] * (T - switch_privacy)
        urgency = [1] * T
    elif condition_id == 7:
        privacy = [0] * switch_privacy + [1] * (T - switch_privacy)
        urgency = [0] * switch_urgency + [1] * (T - switch_urgency)
    else:
        raise ValueError(f"Unknown condition: {condition_id}")

    return privacy, urgency


def sample_observation(A, true_state, rng_key):
    """Sample observation from generative model given true hidden state.

    Args:
        A: list of A matrices (without batch dim)
        true_state: tuple (position, privacy, urgency)
        rng_key: JAX PRNG key

    Returns:
        list of observation indices (one per modality)
    """
    import jax.random as jr

    pos, priv, urg = true_state
    keys = jr.split(rng_key, 4)

    # M0: position_obs | position
    p0 = A[0][:, pos]
    o0 = jr.categorical(keys[0], jnp.log(p0 + 1e-16))

    # M1: privacy_cue | position, privacy
    p1 = A[1][:, pos, priv]
    o1 = jr.categorical(keys[1], jnp.log(p1 + 1e-16))

    # M2: emergency_signal | urgency
    p2 = A[2][:, urg]
    o2 = jr.categorical(keys[2], jnp.log(p2 + 1e-16))

    # M3: complaint | position, privacy
    p3 = A[3][:, pos, priv]
    o3 = jr.categorical(keys[3], jnp.log(p3 + 1e-16))

    return [o0, o1, o2, o3]
