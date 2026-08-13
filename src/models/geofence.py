"""
Experiment 2: Geofence Compliance (EASA)

State factors (4):
  F0: drone_zone     [4] — SAFE, BOUNDARY, RESTRICTED, EMERGENCY_ZONE  (controllable)
  F1: target_position[3] — IN_SAFE, AT_BOUNDARY, IN_RESTRICTED          (uncontrollable)
  F2: airspace_status[2] — OPEN, RESTRICTED                             (uncontrollable)
  F3: urgency        [2] — NORMAL, EMERGENCY                            (uncontrollable)

Observation modalities (5):
  M0: gps_zone        [4] — precise self-location
  M1: target_bearing  [3] — AHEAD, LATERAL, BEHIND
  M2: geofence_alert  [2] — CLEAR, WARNING
  M3: tracking_status [2] — LOCKED, LOST
  M4: dispatch_signal [2] — NONE, EMERGENCY_DISPATCH

Actions for F0 (3): STAY=0, ADVANCE=1, RETREAT=2

C Subtensors: 4 preference profiles indexed by (airspace, urgency),
blended at runtime via belief-weighted mixing.

The urgency factor is what makes this a genuine conflict rather than a
compliance demo. EASA's U-space rules admit emergency exceptions, so
entering restricted airspace is the correct action under a confirmed
emergency and a violation without one. The agent has to infer which
regime it is in from a noisy dispatch signal, and the boundary zone is
the behaviour that falls out of the mixture when it cannot yet tell.
"""

import jax.numpy as jnp

# Labels
ZONES = ["SAFE", "BOUNDARY", "RESTRICTED", "EMERGENCY_ZONE"]
TARGET_POS = ["IN_SAFE", "AT_BOUNDARY", "IN_RESTRICTED"]
AIRSPACE = ["OPEN", "RESTRICTED"]
BEARING = ["AHEAD", "LATERAL", "BEHIND"]

STAY = 0
ADVANCE = 1
RETREAT = 2

# Airspace state indices
OPEN = 0
RESTRICTED = 1

# Urgency state indices
NORMAL = 0
EMERGENCY = 1


def build_A_matrices(noise=0.05, dispatch_noise=0.125):
    """Build observation likelihoods.

    A[0]: gps_zone | drone_zone — shape (4, 4): near-identity
    A[1]: target_bearing | drone_zone, target_pos — shape (3, 4, 3)
    A[2]: geofence_alert | drone_zone, airspace — shape (2, 4, 2)
    A[3]: tracking_status | drone_zone, target_pos — shape (2, 4, 3)
    A[4]: dispatch_signal | urgency — shape (2, 2)

    Args:
        noise: small noise for GPS localization (default 0.05)
        dispatch_noise: unreliability of the emergency dispatch channel
            (default 0.125, matching the emergency-override model)
    """
    a = noise

    # M0: GPS zone — near-perfect localization
    n = 4
    A0 = (1.0 - a) * jnp.eye(n) + (a / (n - 1)) * (jnp.ones((n, n)) - jnp.eye(n))

    # M1: target_bearing — depends on relative position
    A1 = jnp.zeros((3, 4, 3))
    for dz in range(4):
        for tp in range(3):
            diff = tp - dz
            if diff > 0:
                A1 = A1.at[:, dz, tp].set(jnp.array([0.8, 0.15, 0.05]))
            elif diff == 0:
                A1 = A1.at[:, dz, tp].set(jnp.array([0.2, 0.6, 0.2]))
            else:
                A1 = A1.at[:, dz, tp].set(jnp.array([0.05, 0.15, 0.8]))

    # M2: geofence_alert — WARNING when at BOUNDARY/RESTRICTED and airspace RESTRICTED
    A2 = jnp.zeros((2, 4, 2))
    A2 = A2.at[:, 0, :].set(jnp.array([[0.95, 0.9], [0.05, 0.1]]))
    A2 = A2.at[:, 1, :].set(jnp.array([[0.7, 0.2], [0.3, 0.8]]))
    A2 = A2.at[:, 2, :].set(jnp.array([[0.1, 0.05], [0.9, 0.95]]))
    A2 = A2.at[:, 3, :].set(jnp.array([[0.05, 0.02], [0.95, 0.98]]))

    # M3: tracking_status — LOCKED when drone near target
    A3 = jnp.zeros((2, 4, 3))
    for dz in range(4):
        for tp in range(3):
            dist = abs(dz - tp)
            if dist == 0:
                A3 = A3.at[:, dz, tp].set(jnp.array([0.9, 0.1]))
            elif dist == 1:
                A3 = A3.at[:, dz, tp].set(jnp.array([0.5, 0.5]))
            else:
                A3 = A3.at[:, dz, tp].set(jnp.array([0.1, 0.9]))

    # M4: dispatch_signal — noisy report of the emergency authorisation
    b = dispatch_noise
    A4 = jnp.array([
        [1.0 - b, b],
        [b, 1.0 - b],
    ])

    return [A0, A1, A2, A3, A4]


def build_B_matrices(a_urg=0.02):
    """Build transition matrices.

    B[0]: drone_zone — shape (4, 4, 3): STAY/ADVANCE/RETREAT
    B[1]: target_position — shape (3, 3, 1): uncontrollable, drifts toward restricted
    B[2]: airspace_status — shape (2, 2, 1): uncontrollable
    B[3]: urgency — shape (2, 2, 1): uncontrollable, EMERGENCY absorbing

    Args:
        a_urg: NORMAL -> EMERGENCY transition rate. Emergencies are rare
            and, once authorised, do not spontaneously lapse, so the
            state is absorbing (same treatment as the emergency model).
    """
    # B[0]: Drone zone transitions
    B0 = jnp.zeros((4, 4, 3))
    B0 = B0.at[:, :, 0].set(jnp.eye(4))
    for s in range(4):
        next_s = min(s + 1, 3)
        B0 = B0.at[next_s, s, 1].set(1.0)
    for s in range(4):
        next_s = max(s - 1, 0)
        B0 = B0.at[next_s, s, 2].set(1.0)

    # B[1]: Target walks toward restricted zone
    B1 = jnp.array([
        [0.5, 0.1, 0.0],
        [0.4, 0.5, 0.2],
        [0.1, 0.4, 0.8],
    ])[..., None]

    # B[2]: Airspace status — mostly stable
    B2 = jnp.array([
        [0.9, 0.1],
        [0.1, 0.9],
    ])[..., None]

    # B[3]: Urgency — NORMAL -> EMERGENCY only, then absorbing
    B3 = jnp.array([
        [1.0 - a_urg, 0.0],
        [a_urg, 1.0],
    ])[..., None]

    return [B0, B1, B2, B3]


def build_C_profiles():
    """Build preference profiles indexed by (airspace, urgency).

    Returns dict mapping (airspace_idx, urgency_idx) -> [C0, C1, C2, C3, C4].

    4 profiles:
      (OPEN, NORMAL):          tracking-dominant, no restriction to respect
      (OPEN, EMERGENCY):       tracking-dominant, pull toward the target
      (RESTRICTED, NORMAL):    geofence aversion dominates, retreat to SAFE
      (RESTRICTED, EMERGENCY): authorised override, enter restricted airspace

    The two RESTRICTED profiles disagree about where the drone should be:
    NORMAL puts SAFE two units above BOUNDARY, EMERGENCY puts RESTRICTED
    three units above SAFE. Under an unresolved urgency belief the mixture
    of the two lands on BOUNDARY, which is neither profile's own optimum.
    That is the compromise behaviour, and it is a consequence of the
    mixing rule rather than a tuned-in preference for the boundary.
    """
    bearing = jnp.array([1.0, 0.0, -1.0])       # prefer target ahead
    tracking = jnp.array([2.0, -2.0])           # prefer locked tracking
    no_pref = jnp.array([0.0, 0.0])             # dispatch signal is evidence, not a goal

    profiles = {}

    # Open airspace, normal operations: nothing to comply with, follow the target
    profiles[(OPEN, NORMAL)] = [
        jnp.array([0.5, 1.0, -1.0, -3.0]),
        bearing,
        jnp.array([1.0, -1.0]),
        tracking,
        no_pref,
    ]

    # Open airspace, emergency: same freedom, stronger pull toward the target
    profiles[(OPEN, EMERGENCY)] = [
        jnp.array([0.0, 1.0, 2.0, -1.0]),
        bearing,
        jnp.array([1.0, -1.0]),
        tracking,
        no_pref,
    ]

    # Restricted airspace, normal operations: the geofence binds
    profiles[(RESTRICTED, NORMAL)] = [
        jnp.array([2.0, 0.0, -4.0, -6.0]),
        bearing,
        jnp.array([1.0, -1.0]),
        tracking,
        no_pref,
    ]

    # Restricted airspace, authorised emergency: the exception applies.
    # Entering is now preferred, and the geofence warning is tolerated
    # rather than avoided, since alerting is expected during an override.
    profiles[(RESTRICTED, EMERGENCY)] = [
        jnp.array([0.0, 2.0, 3.0, -2.0]),
        bearing,
        jnp.array([0.5, -0.5]),
        tracking,
        no_pref,
    ]

    return profiles


def build_C_vectors_default():
    """Build default C vectors for agent initialization.

    Returns the open-airspace, normal-urgency profile as a starting point.
    The effective C is recomputed by profile mixing at each timestep.
    """
    profiles = build_C_profiles()
    return profiles[(OPEN, NORMAL)]


def build_D_priors():
    D0 = jnp.array([1.0, 0.0, 0.0, 0.0])  # start in SAFE zone
    D1 = jnp.array([0.7, 0.2, 0.1])        # target mostly in safe
    D2 = jnp.array([0.5, 0.5])             # uncertain about airspace
    D3 = jnp.array([7 / 8, 1 / 8])         # emergencies are rare a priori
    return [D0, D1, D2, D3]


def get_A_dependencies():
    return [
        [0],      # gps_zone depends on drone_zone
        [0, 1],   # target_bearing depends on drone_zone AND target_pos
        [0, 2],   # geofence_alert depends on drone_zone AND airspace
        [0, 1],   # tracking_status depends on drone_zone AND target_pos
        [3],      # dispatch_signal depends on urgency only
    ]


def get_B_dependencies():
    return [[0], [1], [2], [3]]


def build_target_schedule(T):
    """Target walks toward restricted zone over time."""
    schedule = []
    for t in range(T):
        if t < T // 3:
            schedule.append(0)
        elif t < 2 * T // 3:
            schedule.append(1)
        else:
            schedule.append(2)
    return schedule
