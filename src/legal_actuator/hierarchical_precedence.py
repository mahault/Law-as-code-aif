"""
Hierarchical Precedence Matrix (HPM) — Defeasible Rule Arbitration.

Legal reasoning is defeasible: rules must yield to overriding contexts.
The HPM acts as a real-time judicial arbiter using a Dual-Path
Asynchronous Interceptor (DPAI) pattern.

Rule priority (highest first):
  1. Safety (collision avoidance, vital interests)
  2. Emergency override (suspend privacy for emergency)
  3. Privacy (GDPR data minimization)
  4. Mission (tracking, navigation)
  5. Optimization (efficiency, data quality)
"""

from enum import IntEnum
from dataclasses import dataclass


class Priority(IntEnum):
    OPTIMIZATION = 0
    MISSION = 1
    PRIVACY = 2
    EMERGENCY = 3
    SAFETY = 4


@dataclass
class LegalRule:
    """A defeasible legal rule with priority and applicability conditions."""
    name: str
    priority: Priority
    action_override: int  # -1 = no override (advisory only)
    condition: callable   # function(context) -> bool

    def applies(self, context):
        return self.condition(context)


class HierarchicalPrecedenceMatrix:
    """Defeasible rule arbitration for drone legal compliance.

    Uses priority ordering to resolve conflicts between competing
    legal obligations. Higher priority rules override lower ones.
    """

    def __init__(self):
        self.rules = []

    def add_rule(self, rule):
        self.rules.append(rule)
        self.rules.sort(key=lambda r: r.priority, reverse=True)

    def arbitrate(self, context):
        """Evaluate all rules and return the highest-priority applicable override.

        Args:
            context: dict with current state information

        Returns:
            (action_override, rule_name, priority) or (None, None, None) if no override
        """
        for rule in self.rules:
            if rule.applies(context):
                if rule.action_override >= 0:
                    return rule.action_override, rule.name, rule.priority
        return None, None, None


def build_drone_hpm():
    """Build a standard HPM for drone operations."""
    hpm = HierarchicalPrecedenceMatrix()

    # Safety: collision avoidance overrides everything
    hpm.add_rule(LegalRule(
        name="collision_avoidance",
        priority=Priority.SAFETY,
        action_override=0,  # HOLD/STAY
        condition=lambda ctx: ctx.get("collision_imminent", False),
    ))

    # Emergency: suspend privacy constraints
    hpm.add_rule(LegalRule(
        name="emergency_override",
        priority=Priority.EMERGENCY,
        action_override=-1,  # advisory: relax privacy, don't override action
        condition=lambda ctx: ctx.get("urgency", "normal") == "emergency",
    ))

    # Privacy: prevent entering privacy zones
    hpm.add_rule(LegalRule(
        name="privacy_boundary",
        priority=Priority.PRIVACY,
        action_override=0,  # HOLD
        condition=lambda ctx: (
            ctx.get("privacy_active", False) and
            ctx.get("proposed_zone", 0) == 2 and  # PRIVACY_ZONE
            ctx.get("urgency", "normal") != "emergency"
        ),
    ))

    # Geofence: prevent restricted zone entry
    hpm.add_rule(LegalRule(
        name="geofence_restriction",
        priority=Priority.PRIVACY,
        action_override=2,  # RETREAT
        condition=lambda ctx: ctx.get("proposed_zone", 0) >= 2,
    ))

    return hpm
