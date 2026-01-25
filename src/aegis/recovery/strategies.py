"""
Recovery strategies for commitment violations.

This module provides various strategies for recovering from violations:
- Retry: Attempt the action again
- Escalate: Escalate to a supervisor
- Compensate: Execute compensating actions
- Renegotiate: Renegotiate the commitment
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Callable, Awaitable
from uuid import uuid4

from pydantic import BaseModel, Field, ConfigDict

from .detector import Violation, ViolationType
from ..commitments import RuntimeCommitment, CommitmentStatus


class RecoveryAction(str, Enum):
    """Types of recovery actions."""
    
    RETRY = "retry"
    ESCALATE = "escalate"
    COMPENSATE = "compensate"
    RENEGOTIATE = "renegotiate"
    ABORT = "abort"
    IGNORE = "ignore"
    MANUAL = "manual"


class RecoveryPlan(BaseModel):
    """
    A plan for recovering from a violation.
    
    Attributes:
        id: Unique plan identifier
        violation_id: ID of the violation being addressed
        action: The recovery action to take
        parameters: Parameters for the recovery action
        priority: Priority of this plan
        created_at: When the plan was created
        executed: Whether the plan has been executed
        result: Result of execution
    """
    
    model_config = ConfigDict(frozen=False)
    
    id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique plan identifier",
    )
    violation_id: str = Field(..., description="ID of the violation")
    action: RecoveryAction = Field(..., description="Recovery action")
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Action parameters",
    )
    priority: int = Field(
        default=0,
        description="Priority (higher = more important)",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When created",
    )
    executed: bool = Field(
        default=False,
        description="Whether executed",
    )
    result: dict[str, Any] | None = Field(
        default=None,
        description="Execution result",
    )


class RecoveryStrategy(ABC):
    """
    Abstract base class for recovery strategies.
    
    Recovery strategies determine how to handle specific types
    of violations and generate recovery plans.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the strategy."""
        pass
    
    @property
    @abstractmethod
    def supported_violations(self) -> list[ViolationType]:
        """List of violation types this strategy can handle."""
        pass
    
    @abstractmethod
    def can_handle(
        self,
        violation: Violation,
        commitment: RuntimeCommitment,
    ) -> bool:
        """
        Check if this strategy can handle the violation.
        
        Args:
            violation: The violation to handle
            commitment: The violated commitment
            
        Returns:
            True if this strategy can handle the violation
        """
        pass
    
    @abstractmethod
    def create_plan(
        self,
        violation: Violation,
        commitment: RuntimeCommitment,
        context: dict[str, Any] | None = None,
    ) -> RecoveryPlan:
        """
        Create a recovery plan for the violation.
        
        Args:
            violation: The violation to address
            commitment: The violated commitment
            context: Additional context
            
        Returns:
            A recovery plan
        """
        pass
    
    @abstractmethod
    async def execute(
        self,
        plan: RecoveryPlan,
        commitment: RuntimeCommitment,
    ) -> dict[str, Any]:
        """
        Execute a recovery plan.
        
        Args:
            plan: The plan to execute
            commitment: The commitment being recovered
            
        Returns:
            Result of the execution
        """
        pass


class RetryStrategy(RecoveryStrategy):
    """
    Retry strategy for transient failures.
    
    This strategy attempts to retry the failed action with
    exponential backoff.
    """
    
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
    ) -> None:
        """
        Initialize the retry strategy.
        
        Args:
            max_retries: Maximum number of retries
            base_delay: Base delay between retries (seconds)
            max_delay: Maximum delay between retries (seconds)
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
    
    @property
    def name(self) -> str:
        return "retry"
    
    @property
    def supported_violations(self) -> list[ViolationType]:
        return [
            ViolationType.DEADLINE_EXCEEDED,
            ViolationType.RESOURCE_EXCEEDED,
        ]
    
    def can_handle(
        self,
        violation: Violation,
        commitment: RuntimeCommitment,
    ) -> bool:
        # Check if we haven't exceeded max retries
        retry_count = violation.context.get("retry_count", 0)
        if retry_count >= self.max_retries:
            return False
        
        # Check if violation type is supported
        if violation.type not in self.supported_violations:
            return False
        
        # Don't retry high-severity violations
        if violation.severity > 0.9:
            return False
        
        return True
    
    def create_plan(
        self,
        violation: Violation,
        commitment: RuntimeCommitment,
        context: dict[str, Any] | None = None,
    ) -> RecoveryPlan:
        retry_count = violation.context.get("retry_count", 0)
        delay = min(
            self.base_delay * (2 ** retry_count),
            self.max_delay,
        )
        
        return RecoveryPlan(
            violation_id=violation.id,
            action=RecoveryAction.RETRY,
            parameters={
                "retry_count": retry_count + 1,
                "delay": delay,
                "max_retries": self.max_retries,
            },
            priority=5,
        )
    
    async def execute(
        self,
        plan: RecoveryPlan,
        commitment: RuntimeCommitment,
    ) -> dict[str, Any]:
        import asyncio
        
        delay = plan.parameters.get("delay", self.base_delay)
        await asyncio.sleep(delay)
        
        # In a real implementation, this would re-execute the action
        return {
            "success": True,
            "action": "retry",
            "retry_count": plan.parameters.get("retry_count", 1),
            "delay": delay,
        }


class EscalateStrategy(RecoveryStrategy):
    """
    Escalation strategy for serious violations.
    
    This strategy escalates violations to a supervisor or
    human operator for manual intervention.
    """
    
    def __init__(
        self,
        escalation_handler: Callable[[Violation, RuntimeCommitment], Awaitable[None]] | None = None,
    ) -> None:
        """
        Initialize the escalation strategy.
        
        Args:
            escalation_handler: Async function to handle escalations
        """
        self.escalation_handler = escalation_handler
    
    @property
    def name(self) -> str:
        return "escalate"
    
    @property
    def supported_violations(self) -> list[ViolationType]:
        return [
            ViolationType.POLICY_VIOLATED,
            ViolationType.UNAUTHORIZED_ACTION,
            ViolationType.BEHAVIORAL_ANOMALY,
        ]
    
    def can_handle(
        self,
        violation: Violation,
        commitment: RuntimeCommitment,
    ) -> bool:
        # Always can escalate high-severity violations
        if violation.severity >= 0.8:
            return True
        
        # Escalate policy and security violations
        if violation.type in self.supported_violations:
            return True
        
        return False
    
    def create_plan(
        self,
        violation: Violation,
        commitment: RuntimeCommitment,
        context: dict[str, Any] | None = None,
    ) -> RecoveryPlan:
        return RecoveryPlan(
            violation_id=violation.id,
            action=RecoveryAction.ESCALATE,
            parameters={
                "severity": violation.severity,
                "violation_type": violation.type.value,
                "requires_human": violation.severity >= 0.9,
            },
            priority=10,  # High priority
        )
    
    async def execute(
        self,
        plan: RecoveryPlan,
        commitment: RuntimeCommitment,
    ) -> dict[str, Any]:
        if self.escalation_handler:
            # Get the violation (would need to be passed or looked up)
            # For now, create a minimal violation object
            violation = Violation(
                id=plan.violation_id,
                type=ViolationType(plan.parameters.get("violation_type", "policy_violated")),
                commitment_id=commitment.id,
                agent_id=commitment.debtor,
                description="Escalated violation",
                severity=plan.parameters.get("severity", 0.8),
            )
            await self.escalation_handler(violation, commitment)
        
        return {
            "success": True,
            "action": "escalate",
            "escalated_to": "supervisor",
            "requires_human": plan.parameters.get("requires_human", False),
        }


class CompensateStrategy(RecoveryStrategy):
    """
    Compensation strategy for irreversible violations.
    
    This strategy executes compensating actions to mitigate
    the effects of a violation.
    """
    
    def __init__(self) -> None:
        """Initialize the compensation strategy."""
        self._compensators: dict[str, Callable[[RuntimeCommitment], Awaitable[dict]]] = {}
    
    @property
    def name(self) -> str:
        return "compensate"
    
    @property
    def supported_violations(self) -> list[ViolationType]:
        return [
            ViolationType.CONDITION_VIOLATED,
            ViolationType.COMMITMENT_ABANDONED,
        ]
    
    def register_compensator(
        self,
        action: str,
        compensator: Callable[[RuntimeCommitment], Awaitable[dict]],
    ) -> None:
        """Register a compensating action for a specific action type."""
        self._compensators[action] = compensator
    
    def can_handle(
        self,
        violation: Violation,
        commitment: RuntimeCommitment,
    ) -> bool:
        # Check if we have a compensator for this action
        if commitment.action in self._compensators:
            return True
        
        # Can handle condition violations
        if violation.type == ViolationType.CONDITION_VIOLATED:
            return True
        
        return False
    
    def create_plan(
        self,
        violation: Violation,
        commitment: RuntimeCommitment,
        context: dict[str, Any] | None = None,
    ) -> RecoveryPlan:
        return RecoveryPlan(
            violation_id=violation.id,
            action=RecoveryAction.COMPENSATE,
            parameters={
                "original_action": commitment.action,
                "has_compensator": commitment.action in self._compensators,
            },
            priority=7,
        )
    
    async def execute(
        self,
        plan: RecoveryPlan,
        commitment: RuntimeCommitment,
    ) -> dict[str, Any]:
        action = plan.parameters.get("original_action", commitment.action)
        
        if action in self._compensators:
            result = await self._compensators[action](commitment)
            return {
                "success": True,
                "action": "compensate",
                "compensator_result": result,
            }
        
        # Default compensation: mark commitment as violated
        return {
            "success": True,
            "action": "compensate",
            "compensation": "marked_violated",
        }


class RenegotiateStrategy(RecoveryStrategy):
    """
    Renegotiation strategy for deadline and condition violations.
    
    This strategy attempts to renegotiate the commitment terms
    to allow for successful completion.
    """
    
    def __init__(
        self,
        max_deadline_extension: timedelta = timedelta(hours=24),
    ) -> None:
        """
        Initialize the renegotiation strategy.
        
        Args:
            max_deadline_extension: Maximum deadline extension allowed
        """
        self.max_deadline_extension = max_deadline_extension
    
    @property
    def name(self) -> str:
        return "renegotiate"
    
    @property
    def supported_violations(self) -> list[ViolationType]:
        return [
            ViolationType.DEADLINE_EXCEEDED,
            ViolationType.CONDITION_VIOLATED,
        ]
    
    def can_handle(
        self,
        violation: Violation,
        commitment: RuntimeCommitment,
    ) -> bool:
        # Can renegotiate deadline violations
        if violation.type == ViolationType.DEADLINE_EXCEEDED:
            # Check if not too far past deadline
            overdue = violation.context.get("overdue_seconds", 0)
            if overdue < self.max_deadline_extension.total_seconds():
                return True
        
        # Can renegotiate condition violations
        if violation.type == ViolationType.CONDITION_VIOLATED:
            return True
        
        return False
    
    def create_plan(
        self,
        violation: Violation,
        commitment: RuntimeCommitment,
        context: dict[str, Any] | None = None,
    ) -> RecoveryPlan:
        parameters: dict[str, Any] = {}
        
        if violation.type == ViolationType.DEADLINE_EXCEEDED:
            # Propose deadline extension
            overdue = violation.context.get("overdue_seconds", 0)
            extension = min(
                overdue * 2,  # Double the overdue time
                self.max_deadline_extension.total_seconds(),
            )
            parameters["new_deadline"] = (
                datetime.now(timezone.utc) + timedelta(seconds=extension)
            ).isoformat()
            parameters["extension_seconds"] = extension
        
        if violation.type == ViolationType.CONDITION_VIOLATED:
            # Propose condition relaxation
            parameters["original_condition"] = commitment.condition
            parameters["proposed_condition"] = "true"  # Simplified
        
        return RecoveryPlan(
            violation_id=violation.id,
            action=RecoveryAction.RENEGOTIATE,
            parameters=parameters,
            priority=6,
        )
    
    async def execute(
        self,
        plan: RecoveryPlan,
        commitment: RuntimeCommitment,
    ) -> dict[str, Any]:
        # In a real implementation, this would send a renegotiation
        # request to the creditor and wait for approval
        
        result: dict[str, Any] = {
            "success": True,
            "action": "renegotiate",
            "proposed_changes": {},
        }
        
        if "new_deadline" in plan.parameters:
            result["proposed_changes"]["deadline"] = plan.parameters["new_deadline"]
        
        if "proposed_condition" in plan.parameters:
            result["proposed_changes"]["condition"] = plan.parameters["proposed_condition"]
        
        return result


class StrategySelector:
    """
    Selector for choosing the best recovery strategy.
    
    The StrategySelector evaluates available strategies and
    selects the most appropriate one for a given violation.
    
    Example:
        >>> selector = StrategySelector()
        >>> selector.register(RetryStrategy())
        >>> selector.register(EscalateStrategy())
        >>> strategy = selector.select(violation, commitment)
    """
    
    def __init__(self) -> None:
        """Initialize the selector."""
        self._strategies: list[RecoveryStrategy] = []
    
    def register(self, strategy: RecoveryStrategy) -> None:
        """Register a recovery strategy."""
        self._strategies.append(strategy)
    
    def unregister(self, strategy_name: str) -> bool:
        """Unregister a strategy by name."""
        for i, s in enumerate(self._strategies):
            if s.name == strategy_name:
                del self._strategies[i]
                return True
        return False
    
    def select(
        self,
        violation: Violation,
        commitment: RuntimeCommitment,
    ) -> RecoveryStrategy | None:
        """
        Select the best strategy for a violation.
        
        Args:
            violation: The violation to handle
            commitment: The violated commitment
            
        Returns:
            The best strategy, or None if no strategy can handle it
        """
        candidates = []
        
        for strategy in self._strategies:
            if strategy.can_handle(violation, commitment):
                candidates.append(strategy)
        
        if not candidates:
            return None
        
        # Sort by priority (based on violation severity and type)
        def priority(s: RecoveryStrategy) -> int:
            # Escalate for high severity
            if violation.severity >= 0.9 and s.name == "escalate":
                return 100
            
            # Retry for low severity transient failures
            if violation.severity < 0.5 and s.name == "retry":
                return 80
            
            # Renegotiate for deadline issues
            if violation.type == ViolationType.DEADLINE_EXCEEDED and s.name == "renegotiate":
                return 70
            
            # Compensate for condition violations
            if violation.type == ViolationType.CONDITION_VIOLATED and s.name == "compensate":
                return 60
            
            return 50
        
        candidates.sort(key=priority, reverse=True)
        return candidates[0]
    
    def select_all(
        self,
        violation: Violation,
        commitment: RuntimeCommitment,
    ) -> list[RecoveryStrategy]:
        """
        Get all strategies that can handle a violation.
        
        Args:
            violation: The violation to handle
            commitment: The violated commitment
            
        Returns:
            List of applicable strategies
        """
        return [
            s for s in self._strategies
            if s.can_handle(violation, commitment)
        ]
    
    def get_strategy(self, name: str) -> RecoveryStrategy | None:
        """Get a strategy by name."""
        for s in self._strategies:
            if s.name == name:
                return s
        return None
