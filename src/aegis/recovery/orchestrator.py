"""
Recovery orchestrator for automated violation recovery.

This module provides the RecoveryOrchestrator which coordinates
the detection and recovery of commitment violations.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Awaitable
from uuid import uuid4

from pydantic import BaseModel, Field, ConfigDict

from .detector import ViolationDetector, Violation, ViolationType
from .strategies import (
    RecoveryStrategy,
    RecoveryPlan,
    RecoveryAction,
    StrategySelector,
    RetryStrategy,
    EscalateStrategy,
    CompensateStrategy,
    RenegotiateStrategy,
)
from ..commitments import RuntimeCommitment, CommitmentStatus


class RecoveryStatus(str, Enum):
    """Status of a recovery attempt."""
    
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    ESCALATED = "escalated"
    ABORTED = "aborted"


class RecoveryResult(BaseModel):
    """
    Result of a recovery attempt.
    
    Attributes:
        id: Unique result identifier
        violation_id: ID of the violation
        commitment_id: ID of the commitment
        status: Recovery status
        strategy_used: Name of the strategy used
        plan: The recovery plan that was executed
        result: Result data from the strategy
        started_at: When recovery started
        completed_at: When recovery completed
        error: Error message if failed
    """
    
    model_config = ConfigDict(frozen=False)
    
    id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique result identifier",
    )
    violation_id: str = Field(..., description="ID of the violation")
    commitment_id: str = Field(..., description="ID of the commitment")
    status: RecoveryStatus = Field(
        default=RecoveryStatus.PENDING,
        description="Recovery status",
    )
    strategy_used: str | None = Field(
        default=None,
        description="Name of strategy used",
    )
    plan: RecoveryPlan | None = Field(
        default=None,
        description="The recovery plan",
    )
    result: dict[str, Any] = Field(
        default_factory=dict,
        description="Result data",
    )
    started_at: datetime | None = Field(
        default=None,
        description="When started",
    )
    completed_at: datetime | None = Field(
        default=None,
        description="When completed",
    )
    error: str | None = Field(
        default=None,
        description="Error message",
    )


# Type for recovery event handlers
RecoveryEventHandler = Callable[[RecoveryResult], Awaitable[None]]


class RecoveryOrchestrator:
    """
    Orchestrator for automated violation recovery.
    
    The RecoveryOrchestrator coordinates the detection and recovery
    of commitment violations, selecting appropriate strategies and
    executing recovery plans.
    
    Example:
        >>> orchestrator = RecoveryOrchestrator()
        >>> result = await orchestrator.recover(violation, commitment)
        >>> if result.status == RecoveryStatus.SUCCEEDED:
        ...     print("Recovery successful!")
    """
    
    def __init__(
        self,
        detector: ViolationDetector | None = None,
        selector: StrategySelector | None = None,
        auto_register_strategies: bool = True,
    ) -> None:
        """
        Initialize the orchestrator.
        
        Args:
            detector: Violation detector
            selector: Strategy selector
            auto_register_strategies: Whether to register default strategies
        """
        self.detector = detector or ViolationDetector()
        self.selector = selector or StrategySelector()
        
        self._results: dict[str, RecoveryResult] = {}
        self._event_handlers: list[RecoveryEventHandler] = []
        self._running = False
        self._monitor_task: asyncio.Task | None = None
        self._commitments: dict[str, RuntimeCommitment] = {}
        
        if auto_register_strategies:
            self._register_default_strategies()
    
    def _register_default_strategies(self) -> None:
        """Register default recovery strategies."""
        self.selector.register(RetryStrategy())
        self.selector.register(EscalateStrategy())
        self.selector.register(CompensateStrategy())
        self.selector.register(RenegotiateStrategy())
    
    def add_event_handler(self, handler: RecoveryEventHandler) -> None:
        """Add a recovery event handler."""
        self._event_handlers.append(handler)
    
    def remove_event_handler(self, handler: RecoveryEventHandler) -> bool:
        """Remove a recovery event handler."""
        try:
            self._event_handlers.remove(handler)
            return True
        except ValueError:
            return False
    
    async def _notify_handlers(self, result: RecoveryResult) -> None:
        """Notify all event handlers of a recovery result."""
        for handler in self._event_handlers:
            try:
                await handler(result)
            except Exception:
                pass
    
    def register_commitment(self, commitment: RuntimeCommitment) -> None:
        """Register a commitment for monitoring."""
        self._commitments[commitment.id] = commitment
    
    def unregister_commitment(self, commitment_id: str) -> bool:
        """Unregister a commitment from monitoring."""
        if commitment_id in self._commitments:
            del self._commitments[commitment_id]
            return True
        return False
    
    async def check_commitment(
        self,
        commitment: RuntimeCommitment,
        context: dict[str, Any] | None = None,
    ) -> list[Violation]:
        """
        Check a commitment for violations.
        
        Args:
            commitment: The commitment to check
            context: Additional context
            
        Returns:
            List of detected violations
        """
        return self.detector.check_commitment(commitment, context)
    
    async def recover(
        self,
        violation: Violation,
        commitment: RuntimeCommitment,
        strategy_name: str | None = None,
    ) -> RecoveryResult:
        """
        Attempt to recover from a violation.
        
        Args:
            violation: The violation to recover from
            commitment: The violated commitment
            strategy_name: Specific strategy to use (optional)
            
        Returns:
            Result of the recovery attempt
        """
        result = RecoveryResult(
            violation_id=violation.id,
            commitment_id=commitment.id,
            started_at=datetime.now(timezone.utc),
        )
        
        try:
            # Select strategy
            if strategy_name:
                strategy = self.selector.get_strategy(strategy_name)
            else:
                strategy = self.selector.select(violation, commitment)
            
            if not strategy:
                result.status = RecoveryStatus.FAILED
                result.error = "No suitable recovery strategy found"
                result.completed_at = datetime.now(timezone.utc)
                self._results[result.id] = result
                await self._notify_handlers(result)
                return result
            
            result.strategy_used = strategy.name
            result.status = RecoveryStatus.IN_PROGRESS
            
            # Create recovery plan
            plan = strategy.create_plan(violation, commitment)
            result.plan = plan
            
            # Execute recovery
            execution_result = await strategy.execute(plan, commitment)
            
            plan.executed = True
            plan.result = execution_result
            
            # Determine success
            if execution_result.get("success", False):
                result.status = RecoveryStatus.SUCCEEDED
                result.result = execution_result
                
                # Mark violation as resolved
                self.detector.resolve_violation(
                    violation.id,
                    f"Recovered using {strategy.name} strategy",
                )
            else:
                result.status = RecoveryStatus.FAILED
                result.error = execution_result.get("error", "Recovery failed")
                result.result = execution_result
            
            # Check if escalated
            if plan.action == RecoveryAction.ESCALATE:
                result.status = RecoveryStatus.ESCALATED
            
        except Exception as e:
            result.status = RecoveryStatus.FAILED
            result.error = str(e)
        
        result.completed_at = datetime.now(timezone.utc)
        self._results[result.id] = result
        await self._notify_handlers(result)
        
        return result
    
    async def recover_all(
        self,
        commitment: RuntimeCommitment,
        context: dict[str, Any] | None = None,
    ) -> list[RecoveryResult]:
        """
        Check for violations and recover from all of them.
        
        Args:
            commitment: The commitment to check and recover
            context: Additional context
            
        Returns:
            List of recovery results
        """
        violations = await self.check_commitment(commitment, context)
        results = []
        
        for violation in violations:
            result = await self.recover(violation, commitment)
            results.append(result)
        
        return results
    
    async def start_monitoring(
        self,
        interval: float = 60.0,
    ) -> None:
        """
        Start background monitoring of commitments.
        
        Args:
            interval: Check interval in seconds
        """
        if self._running:
            return
        
        self._running = True
        self._monitor_task = asyncio.create_task(
            self._monitor_loop(interval)
        )
    
    async def stop_monitoring(self) -> None:
        """Stop background monitoring."""
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None
    
    async def _monitor_loop(self, interval: float) -> None:
        """Background monitoring loop."""
        while self._running:
            for commitment in list(self._commitments.values()):
                if commitment.status == CommitmentStatus.ACTIVE:
                    try:
                        await self.recover_all(commitment)
                    except Exception:
                        pass
            
            await asyncio.sleep(interval)
    
    def get_result(self, result_id: str) -> RecoveryResult | None:
        """Get a recovery result by ID."""
        return self._results.get(result_id)
    
    def get_results_for_commitment(
        self,
        commitment_id: str,
    ) -> list[RecoveryResult]:
        """Get all recovery results for a commitment."""
        return [
            r for r in self._results.values()
            if r.commitment_id == commitment_id
        ]
    
    def get_results_for_violation(
        self,
        violation_id: str,
    ) -> list[RecoveryResult]:
        """Get all recovery results for a violation."""
        return [
            r for r in self._results.values()
            if r.violation_id == violation_id
        ]
    
    def get_results_by_status(
        self,
        status: RecoveryStatus,
    ) -> list[RecoveryResult]:
        """Get all recovery results with a specific status."""
        return [
            r for r in self._results.values()
            if r.status == status
        ]
    
    def get_statistics(self) -> dict[str, Any]:
        """Get recovery statistics."""
        results = list(self._results.values())
        
        by_status = {}
        for r in results:
            by_status[r.status.value] = by_status.get(r.status.value, 0) + 1
        
        by_strategy = {}
        for r in results:
            if r.strategy_used:
                by_strategy[r.strategy_used] = by_strategy.get(r.strategy_used, 0) + 1
        
        succeeded = sum(1 for r in results if r.status == RecoveryStatus.SUCCEEDED)
        
        return {
            "total_attempts": len(results),
            "succeeded": succeeded,
            "failed": sum(1 for r in results if r.status == RecoveryStatus.FAILED),
            "escalated": sum(1 for r in results if r.status == RecoveryStatus.ESCALATED),
            "success_rate": succeeded / len(results) if results else 0.0,
            "by_status": by_status,
            "by_strategy": by_strategy,
            "monitored_commitments": len(self._commitments),
            "unresolved_violations": len(self.detector.get_unresolved_violations()),
        }
    
    async def run_recovery_demo(self) -> dict[str, Any]:
        """
        Run a demonstration of the recovery system.
        
        This creates sample violations and demonstrates recovery.
        
        Returns:
            Demo results
        """
        from datetime import timedelta
        
        demo_results = {
            "violations_created": [],
            "recovery_attempts": [],
            "final_statistics": {},
        }
        
        # Create a sample commitment
        commitment = RuntimeCommitment(
            debtor="demo-agent",
            creditor="demo-system",
            action="process_data",
            condition="data_available",
            deadline=datetime.now(timezone.utc) - timedelta(hours=1),  # Already past
        )
        
        # Create violations
        deadline_violation = Violation(
            type=ViolationType.DEADLINE_EXCEEDED,
            commitment_id=commitment.id,
            agent_id=commitment.debtor,
            description="Demo deadline violation",
            severity=0.6,
            context={
                "deadline": commitment.deadline.isoformat() if commitment.deadline else None,
                "overdue_seconds": 3600,
            },
        )
        demo_results["violations_created"].append({
            "id": deadline_violation.id,
            "type": deadline_violation.type.value,
            "severity": deadline_violation.severity,
        })
        
        # Attempt recovery
        result = await self.recover(deadline_violation, commitment)
        demo_results["recovery_attempts"].append({
            "violation_id": deadline_violation.id,
            "strategy": result.strategy_used,
            "status": result.status.value,
            "result": result.result,
        })
        
        # Create a policy violation
        policy_violation = Violation(
            type=ViolationType.POLICY_VIOLATED,
            commitment_id=commitment.id,
            agent_id=commitment.debtor,
            description="Demo policy violation",
            severity=0.9,
            context={
                "policy": "no_external_access",
                "action": "http_request",
            },
        )
        demo_results["violations_created"].append({
            "id": policy_violation.id,
            "type": policy_violation.type.value,
            "severity": policy_violation.severity,
        })
        
        # Attempt recovery (should escalate due to high severity)
        result = await self.recover(policy_violation, commitment)
        demo_results["recovery_attempts"].append({
            "violation_id": policy_violation.id,
            "strategy": result.strategy_used,
            "status": result.status.value,
            "result": result.result,
        })
        
        # Get final statistics
        demo_results["final_statistics"] = self.get_statistics()
        
        return demo_results
