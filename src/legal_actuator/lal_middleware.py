"""
Legal Actuator Layer (LAL) — Middleware for drone legal compliance.

Intercepts and mutates data flows and kinematic commands to ensure
real-time regulatory adherence.
"""


class LegalActuatorLayer:
    """Programmatic middleware that enforces legal constraints on drone operations.

    Combines three compliance modules:
    1. Data Minimization (GDPR Art 5): anonymization of non-target faces
    2. Geofence Enforcement (EASA): boundary compliance
    3. Algorithmic Traceability (EU AI Act): CIL logging

    The LAL sits between the perception pipeline and the physical actuator,
    intercepting commands and observations.
    """

    def __init__(self, anonymizer=None, geofence_checker=None, logger=None):
        self.anonymizer = anonymizer
        self.geofence_checker = geofence_checker
        self.logger = logger

    def intercept_observation(self, observation, metadata=None):
        """Process incoming observation through data minimization.

        Args:
            observation: raw observation dict
            metadata: additional context (scene composition, consent, etc.)

        Returns:
            processed observation (potentially anonymized)
        """
        if self.anonymizer and metadata:
            scene = metadata.get("scene_composition", "unknown")
            consent = metadata.get("consent", False)
            if scene != "target_only" and not consent:
                observation = self.anonymizer.apply(observation)
        return observation

    def intercept_action(self, action, state_belief, context):
        """Process outgoing action through geofence and legal checks.

        Args:
            action: proposed action from AIF agent
            state_belief: current belief about hidden states
            context: dict with airspace_status, drone_zone, etc.

        Returns:
            (approved_action, overridden, reason)
        """
        if self.geofence_checker:
            approved, reason = self.geofence_checker.check(action, context)
            if not approved:
                safe_action = self.geofence_checker.get_safe_action(context)
                return safe_action, True, reason

        return action, False, "approved"

    def log_decision(self, perception, action, efe_values, q_pi):
        """Log decision to CIL for EU AI Act compliance."""
        if self.logger:
            self.logger.log_decision(
                perception=perception,
                actuation=action,
                efe=efe_values,
                q_pi=q_pi,
            )


class AnonymizationModule:
    """Edge-level data minimization via face anonymization."""

    def apply(self, observation):
        """Apply anonymization to non-target faces in observation."""
        # In simulation: mark observation as anonymized
        processed = dict(observation)
        processed["anonymized"] = True
        processed["exposure_level"] = "partial"
        return processed


class GeofenceChecker:
    """EASA geofence boundary enforcement."""

    def __init__(self, restricted_zones=None):
        self.restricted_zones = restricted_zones or [2, 3]  # RESTRICTED, EMERGENCY

    def check(self, action, context):
        """Check if proposed action would violate geofence.

        Returns:
            (approved: bool, reason: str)
        """
        current_zone = context.get("drone_zone", 0)
        proposed_zone = context.get("proposed_zone", current_zone)

        if proposed_zone in self.restricted_zones:
            return False, f"geofence_violation: zone {proposed_zone}"
        return True, "clear"

    def get_safe_action(self, context):
        """Return safest available action."""
        current_zone = context.get("drone_zone", 0)
        if current_zone >= 1:
            return 2  # RETREAT
        return 0  # STAY
