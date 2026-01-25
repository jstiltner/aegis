"""
Commitment violation detection.

This module provides detection of commitment violations including:
- Deadline violations
- Condition violations
- Policy violations
- Behavioral anomalies
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable
from uuid import uuid4

from pydantic import BaseModel, Field, ConfigDict

from ..commitments import RuntimeCommitment, CommitmentStatus


class ViolationType(str, Enum):
    """Types of commitment violations."""
    
    DEADLINE_EXCEEDED = "deadline_exceeded"
    CONDITION_VIOLATED = "condition_violated"
    POLICY_VIOLATED = "policy_violated"
    UNAUTHORIZED_ACTION = "unauthorized_action"
    RESOURCE_EXCEEDED = "resource_exceeded"
    BEHAVIORAL_ANOMALY = "behavioral_anomaly"
    COMMITMENT_ABANDONED = "commitment_abandoned"
    INVALID_STATE = "invalid_state"


class Violation(BaseModel):
    """
    A detected commitment violation.
    
    Attributes:
        id: Unique violation identifier
        type: Type of violation
        commitment_id: ID of the violated commitment
        agent_id: ID of the agent that violated
        description: Description of the violation
        severity: Severity level (0-1)
        detected_at: When the violation was detected
        context: Additional context about the violation
        resolved: Whether the violation has been resolved
        resolution: How the violation was resolved
    """
    
    model_config = ConfigDict(frozen=False)
    
    id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique violation identifier",
    )
    type: ViolationType = Field(..., description="Type of violation")
    commitment_id: str = Field(..., description="ID of the violated commitment")
    agent_id: str = Field(..., description="ID of the violating agent")
    description: str = Field(..., description="Description of the violation")
    severity: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Severity level",
    )
    detected_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When detected",
    )
    context: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context",
    )
    resolved: bool = Field(
        default=False,
        description="Whether resolved",
    )
    resolution: str | None = Field(
        default=None,
        description="How resolved",
    )
    
    def resolve(self, resolution: str) -> None:
        """Mark the violation as resolved."""
        self.resolved = True
        self.resolution = resolution


# Type for violation handlers
ViolationHandler = Callable[[Violation], None]


class ViolationDetector:
    """
    Detector for commitment violations.
    
    The ViolationDetector monitors commitments and detects various
    types of violations including deadline, condition, and policy violations.
    
    Example:
        >>> detector = ViolationDetector()
        >>> detector.add_handler(lambda v: print(f"Violation: {v.type}"))
        >>> violations = detector.check_commitment(commitment)
    """
    
    def __init__(self) -> None:
        """Initialize the detector."""
        self._handlers: list[ViolationHandler] = []
        self._violations: dict[str, Violation] = {}
        self._condition_evaluators: dict[str, Callable[[str, dict], bool]] = {}
    
    def add_handler(self, handler: ViolationHandler) -> None:
        """Add a violation handler."""
        self._handlers.append(handler)
    
    def remove_handler(self, handler: ViolationHandler) -> bool:
        """Remove a violation handler."""
        try:
            self._handlers.remove(handler)
            return True
        except ValueError:
            return False
    
    def register_condition_evaluator(
        self,
        name: str,
        evaluator: Callable[[str, dict], bool],
    ) -> None:
        """
        Register a condition evaluator.
        
        Args:
            name: Name of the evaluator
            evaluator: Function that takes (condition, context) and returns bool
        """
        self._condition_evaluators[name] = evaluator
    
    def check_commitment(
        self,
        commitment: RuntimeCommitment,
        context: dict[str, Any] | None = None,
    ) -> list[Violation]:
        """
        Check a commitment for violations.
        
        Args:
            commitment: The commitment to check
            context: Additional context for evaluation
            
        Returns:
            List of detected violations
        """
        violations = []
        ctx = context or {}
        
        # Check deadline
        deadline_violation = self._check_deadline(commitment)
        if deadline_violation:
            violations.append(deadline_violation)
        
        # Check condition
        condition_violation = self._check_condition(commitment, ctx)
        if condition_violation:
            violations.append(condition_violation)
        
        # Check for abandoned commitment
        abandoned_violation = self._check_abandoned(commitment)
        if abandoned_violation:
            violations.append(abandoned_violation)
        
        # Check for invalid state
        state_violation = self._check_state(commitment)
        if state_violation:
            violations.append(state_violation)
        
        # Notify handlers and store violations
        for violation in violations:
            self._violations[violation.id] = violation
            for handler in self._handlers:
                try:
                    handler(violation)
                except Exception:
                    pass
        
        return violations
    
    def _check_deadline(
        self,
        commitment: RuntimeCommitment,
    ) -> Violation | None:
        """Check for deadline violation."""
        if commitment.deadline is None:
            return None
        
        if commitment.status in (
            CommitmentStatus.FULFILLED,
            CommitmentStatus.CANCELLED,
            CommitmentStatus.VIOLATED,
        ):
            return None
        
        now = datetime.now(timezone.utc)
        if now > commitment.deadline:
            return Violation(
                type=ViolationType.DEADLINE_EXCEEDED,
                commitment_id=commitment.id,
                agent_id=commitment.debtor,
                description=f"Commitment deadline exceeded by {now - commitment.deadline}",
                severity=0.8,
                context={
                    "deadline": commitment.deadline.isoformat(),
                    "current_time": now.isoformat(),
                    "overdue_seconds": (now - commitment.deadline).total_seconds(),
                },
            )
        
        return None
    
    def _check_condition(
        self,
        commitment: RuntimeCommitment,
        context: dict[str, Any],
    ) -> Violation | None:
        """Check for condition violation."""
        if not commitment.condition or commitment.condition == "true":
            return None
        
        if commitment.status != CommitmentStatus.ACTIVE:
            return None
        
        # Try registered evaluators
        for name, evaluator in self._condition_evaluators.items():
            try:
                if not evaluator(commitment.condition, context):
                    return Violation(
                        type=ViolationType.CONDITION_VIOLATED,
                        commitment_id=commitment.id,
                        agent_id=commitment.debtor,
                        description=f"Condition '{commitment.condition}' violated",
                        severity=0.7,
                        context={
                            "condition": commitment.condition,
                            "evaluator": name,
                            "context": context,
                        },
                    )
            except Exception:
                pass
        
        return None
    
    def _check_abandoned(
        self,
        commitment: RuntimeCommitment,
    ) -> Violation | None:
        """Check for abandoned commitment."""
        if commitment.status != CommitmentStatus.ACTIVE:
            return None
        
        # Check if commitment has been inactive too long
        now = datetime.now(timezone.utc)
        inactive_threshold = 3600  # 1 hour
        
        if hasattr(commitment, "last_activity"):
            last_activity = commitment.last_activity
            if (now - last_activity).total_seconds() > inactive_threshold:
                return Violation(
                    type=ViolationType.COMMITMENT_ABANDONED,
                    commitment_id=commitment.id,
                    agent_id=commitment.debtor,
                    description="Commitment appears to be abandoned (no activity)",
                    severity=0.5,
                    context={
                        "last_activity": last_activity.isoformat(),
                        "inactive_seconds": (now - last_activity).total_seconds(),
                    },
                )
        
        return None
    
    def _check_state(
        self,
        commitment: RuntimeCommitment,
    ) -> Violation | None:
        """Check for invalid state."""
        # Check for inconsistent state
        if commitment.status == CommitmentStatus.FULFILLED:
            if commitment.deadline and datetime.now(timezone.utc) < commitment.created_at:
                return Violation(
                    type=ViolationType.INVALID_STATE,
                    commitment_id=commitment.id,
                    agent_id=commitment.debtor,
                    description="Commitment fulfilled before creation time",
                    severity=0.9,
                    context={
                        "status": commitment.status.value,
                        "created_at": commitment.created_at.isoformat(),
                    },
                )
        
        return None
    
    def check_policy_violation(
        self,
        commitment: RuntimeCommitment,
        action: str,
        policy_result: dict[str, Any],
    ) -> Violation | None:
        """
        Check for policy violation.
        
        Args:
            commitment: The commitment
            action: The action that was attempted
            policy_result: Result from policy evaluation
            
        Returns:
            Violation if policy was violated
        """
        if policy_result.get("allowed", True):
            return None
        
        violation = Violation(
            type=ViolationType.POLICY_VIOLATED,
            commitment_id=commitment.id,
            agent_id=commitment.debtor,
            description=f"Policy violation for action '{action}'",
            severity=policy_result.get("severity", 0.8),
            context={
                "action": action,
                "policy_result": policy_result,
                "reason": policy_result.get("reason", "Unknown"),
            },
        )
        
        self._violations[violation.id] = violation
        for handler in self._handlers:
            try:
                handler(violation)
            except Exception:
                pass
        
        return violation
    
    def check_resource_violation(
        self,
        commitment: RuntimeCommitment,
        resource: str,
        used: float,
        limit: float,
    ) -> Violation | None:
        """
        Check for resource limit violation.
        
        Args:
            commitment: The commitment
            resource: Name of the resource
            used: Amount used
            limit: Maximum allowed
            
        Returns:
            Violation if limit exceeded
        """
        if used <= limit:
            return None
        
        violation = Violation(
            type=ViolationType.RESOURCE_EXCEEDED,
            commitment_id=commitment.id,
            agent_id=commitment.debtor,
            description=f"Resource '{resource}' exceeded: {used} > {limit}",
            severity=min(1.0, (used - limit) / limit),
            context={
                "resource": resource,
                "used": used,
                "limit": limit,
                "exceeded_by": used - limit,
            },
        )
        
        self._violations[violation.id] = violation
        for handler in self._handlers:
            try:
                handler(violation)
            except Exception:
                pass
        
        return violation
    
    def get_violation(self, violation_id: str) -> Violation | None:
        """Get a violation by ID."""
        return self._violations.get(violation_id)
    
    def get_violations_for_commitment(
        self,
        commitment_id: str,
    ) -> list[Violation]:
        """Get all violations for a commitment."""
        return [
            v for v in self._violations.values()
            if v.commitment_id == commitment_id
        ]
    
    def get_violations_for_agent(
        self,
        agent_id: str,
    ) -> list[Violation]:
        """Get all violations for an agent."""
        return [
            v for v in self._violations.values()
            if v.agent_id == agent_id
        ]
    
    def get_unresolved_violations(self) -> list[Violation]:
        """Get all unresolved violations."""
        return [
            v for v in self._violations.values()
            if not v.resolved
        ]
    
    def get_violations_by_type(
        self,
        violation_type: ViolationType,
    ) -> list[Violation]:
        """Get all violations of a specific type."""
        return [
            v for v in self._violations.values()
            if v.type == violation_type
        ]
    
    def get_violations_by_severity(
        self,
        min_severity: float = 0.0,
        max_severity: float = 1.0,
    ) -> list[Violation]:
        """Get violations within a severity range."""
        return [
            v for v in self._violations.values()
            if min_severity <= v.severity <= max_severity
        ]
    
    def resolve_violation(
        self,
        violation_id: str,
        resolution: str,
    ) -> bool:
        """
        Resolve a violation.
        
        Args:
            violation_id: ID of the violation
            resolution: Description of how it was resolved
            
        Returns:
            True if violation was found and resolved
        """
        violation = self._violations.get(violation_id)
        if violation:
            violation.resolve(resolution)
            return True
        return False
    
    def get_statistics(self) -> dict[str, Any]:
        """Get violation statistics."""
        violations = list(self._violations.values())
        
        by_type = {}
        for v in violations:
            by_type[v.type.value] = by_type.get(v.type.value, 0) + 1
        
        resolved = sum(1 for v in violations if v.resolved)
        
        return {
            "total": len(violations),
            "resolved": resolved,
            "unresolved": len(violations) - resolved,
            "by_type": by_type,
            "average_severity": (
                sum(v.severity for v in violations) / len(violations)
                if violations else 0.0
            ),
        }
