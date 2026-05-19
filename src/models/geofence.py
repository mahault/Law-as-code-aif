"""
Experiment 2: Geofence Compliance (EASA)

State factors (3):
  F0: drone_zone     [4] — SAFE, BOUNDARY, RESTRICTED, EMERGENCY_ZONE  (controllable)
  F1: target_position[3] — IN_SAFE, AT_BOUNDARY, IN_RESTRICTED          (uncontrollable)
  F2: airspace_status[2] — OPEN, RESTRICTED                             (uncontrollable)

Observation modalities (4):
  M0: gps_zone       [4] — precise self-location
  M1: target_bearing  [3] — AHEAD, LATERAL, BEHIND
  M2: geofence_alert  [2] — CLEAR, WARNING
  M3: tracking_status [2] — LOCKED, LOST

Actions for F0 (3): STAY=0, ADVANCE=1, RETREAT=2

C Subtensors: 2 preference profiles indexed by airspace status,
blended at runtime via belief-weighted mixing.
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


def build_A_matrices(noise=0.05):
    """Build observation likelihoods.

    A[0]: gps_zone | drone_zone — shape (4, 4): near-identity
    A[1]: target_bearing | drone_zone, target_pos — shape (3, 4, 3)
    A[2]: geofence_alert | drone_zone, airspace — shape (2, 4, 2)
    A[3]: tracking_status | drone_zone, target_pos — shape (2, 4, 3)

    Args:
        noise: small noise for GPS localization (default 0.05)
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

    return [A0, A1, A2, A3]


def build_B_matrices():
    """Build transition matrices.

    B[0]: drone_zone — shape (4, 4, 3): STAY/ADVANCE/RETREAT
    B[1]: target_position — shape (3, 3, 1): uncontrollable, drifts toward restricted
    B[2]: airspace_status — shape (2, 2, 1): uncontrollable
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

    return [B0, B1, B2]


def build_C_profiles():
    """Build preference profiles indexed by airspace status.

    Returns dict mapping (airspace_idx,) -> [C0, C1, C2, C3].

    2 profiles:
      OPEN:       tracking-dominant preferences
      RESTRICTED: strong geofence aversion
    """
    profiles = {}

    # OPEN airspace: tracking dominates
    profiles[(OPEN,)] = [
        jnp.array([0.5, 1.0, -1.0, -3.0]),    # mild position preference
        jnp.array([1.0, 0.0, -1.0]),            # prefer target ahead
        jnp.array([1.0, -1.0]),                  # prefer clear alerts
        jnp.array([2.0, -2.0]),                  # prefer locked tracking
    ]

    # RESTRICTED airspace: geofence aversion dominates
    profiles[(RESTRICTED,)] = [
        jnp.array([2.0, 0.0, -4.0, -6.0]),     # strong aversion to restricted zones
        jnp.array([1.0, 0.0, -1.0]),            # still prefer target ahead
        jnp.array([1.0, -1.0]),                  # prefer clear alerts
        jnp.array([2.0, -2.0]),                  # prefer locked tracking
    ]

    return profiles


def build_C_vectors_default():
    """Build default C vectors for agent initialization.

    Returns the OPEN profile as a starting point.
    """
    profiles = build_C_profiles()
    return profiles[(OPEN,)]


def build_D_priors():
    D0 = jnp.array([1.0, 0.0, 0.0, 0.0])  # start in SAFE zone
    D1 = jnp.array([0.7, 0.2, 0.1])        # target mostly in safe
    D2 = jnp.array([0.5, 0.5])             # uncertain about airspace
    return [D0, D1, D2]


def get_A_dependencies():
    return [
        [0],      # gps_zone depends on drone_zone
        [0, 1],   # target_bearing depends on drone_zone AND target_pos
        [0, 2],   # geofence_alert depends on drone_zone AND airspace
        [0, 1],   # tracking_status depends on drone_zone AND target_pos
    ]


def get_B_dependencies():
    return [[0], [1], [2]]


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
