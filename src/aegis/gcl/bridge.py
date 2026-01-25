"""
GCL Bridge for deep integration with the GCL framework.

This module provides bidirectional conversion between agent runtime
types and GCL types, enabling full integration with the GCL verification
engine and commitment portfolio system.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, TYPE_CHECKING

from pydantic import BaseModel, Field, ConfigDict

from .models import (
    RuntimeCommitment,
    CommitmentStatus,
    CommitmentResult,
    ToolCommitment,
)

# Try to import GCL types
try:
    from gcl.core.commitment import (
        GroundedCommitment,
        ActionSpec,
        Consequence,
        FailureMode,
        ContextRegion,
        VerificationResult as GCLVerificationResult,
        VerificationStatus as GCLVerificationStatus,
        CommitmentPortfolio,
    )
    from gcl.core.predicates import Predicate
    from gcl.core.verification import (
        VerificationEngine,
        VerificationContext as GCLVerificationContext,
        BatchVerificationResult,
        VerificationReport,
    )
    GCL_AVAILABLE = True
except ImportError:
    GCL_AVAILABLE = False
    # Define placeholder types for type hints
    GroundedCommitment = Any
    CommitmentPortfolio = Any
    VerificationEngine = Any
    GCLVerificationContext = Any
    GCLVerificationResult = Any
    BatchVerificationResult = Any


class GCLBridge:
    """
    Bridge between agent runtime and GCL framework.
    
    The GCLBridge provides:
    - Bidirectional type conversion
    - Portfolio synchronization
    - Verification engine integration
    - Result mapping
    
    Example:
        >>> bridge = GCLBridge()
        >>> if bridge.is_available:
        ...     gcl_commitment = bridge.to_gcl(runtime_commitment)
        ...     result = bridge.verify(gcl_commitment, context)
    """
    
    def __init__(
        self,
        verification_timeout: float | None = None,
        strict_context_matching: bool = False,
    ) -> None:
        """
        Initialize the GCL bridge.
        
        Args:
            verification_timeout: Default timeout for verification operations
            strict_context_matching: If True, fail if context doesn't match bounds
        """
        self._available = GCL_AVAILABLE
        self._engine: VerificationEngine | None = None
        self._portfolios: dict[str, CommitmentPortfolio] = {}
        
        if self._available:
            self._engine = VerificationEngine(
                timeout_seconds=verification_timeout,
                strict_context_matching=strict_context_matching,
            )
    
    @property
    def is_available(self) -> bool:
        """Check if GCL integration is available."""
        return self._available
    
    @property
    def engine(self) -> VerificationEngine | None:
        """Get the verification engine."""
        return self._engine
    
    # Type Conversion: Runtime -> GCL
    
    def to_gcl(self, commitment: RuntimeCommitment) -> GroundedCommitment | None:
        """
        Convert a RuntimeCommitment to a GCL GroundedCommitment.
        
        Args:
            commitment: The runtime commitment to convert
            
        Returns:
            GCL GroundedCommitment, or None if GCL not available
        """
        if not self._available:
            return None
        
        # Build trigger conditions
        trigger_conditions = []
        if commitment.trigger_condition:
            trigger_conditions.append(Predicate(
                name="trigger",
                expression=commitment.trigger_condition,
            ))
        else:
            # Default trigger: always active
            trigger_conditions.append(Predicate(
                name="always",
                expression="true",
            ))
        
        # Build failure modes
        failure_modes = []
        for name, condition in commitment.failure_conditions.items():
            failure_modes.append(FailureMode(
                name=name,
                condition=Predicate(name=name, expression=condition),
                consequence=Consequence(
                    consequence_type="stake_slash",
                    magnitude=0.5,
                    description=f"Failure: {name}",
                ),
                severity=0.5,
            ))
        
        # Ensure at least one failure mode (GCL requires this)
        if not failure_modes:
            failure_modes.append(FailureMode(
                name="default_failure",
                condition=Predicate(
                    name="default_failure",
                    expression="not success",
                ),
                consequence=Consequence(
                    consequence_type="stake_slash",
                    magnitude=0.5,
                    description="Default failure",
                ),
                severity=0.5,
            ))
        
        return GroundedCommitment(
            id=commitment.id,
            issuer=commitment.agent_id,
            trigger_conditions=trigger_conditions,
            promised_behavior=ActionSpec(
                action_type=commitment.action_type,
                description=commitment.description,
                timeout_seconds=commitment.timeout_seconds,
            ),
            success_condition=Predicate(
                name="success",
                expression=commitment.success_condition,
            ),
            failure_modes=failure_modes,
            stake=commitment.stake,
            confidence=commitment.confidence,
            valid_contexts=ContextRegion(
                description=commitment.description,
            ),
            created_at=commitment.created_at,
            expires_at=commitment.expires_at,
            metadata=commitment.metadata,
        )
    
    def to_gcl_context(
        self,
        pre_state: dict[str, Any],
        post_state: dict[str, Any],
        action: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> GCLVerificationContext | None:
        """
        Create a GCL VerificationContext.
        
        Args:
            pre_state: State before the action
            post_state: State after the action
            action: The action that was performed
            metadata: Additional context
            
        Returns:
            GCL VerificationContext, or None if GCL not available
        """
        if not self._available:
            return None
        
        return GCLVerificationContext(
            pre_state=pre_state,
            post_state=post_state,
            action=action or {},
            metadata=metadata or {},
        )
    
    # Type Conversion: GCL -> Runtime
    
    def from_gcl(
        self,
        gcl_commitment: GroundedCommitment,
        agent_id: str | None = None,
    ) -> RuntimeCommitment:
        """
        Convert a GCL GroundedCommitment to a RuntimeCommitment.
        
        Args:
            gcl_commitment: The GCL commitment to convert
            agent_id: Override agent ID (uses issuer if not provided)
            
        Returns:
            RuntimeCommitment
        """
        # Extract failure conditions
        failure_conditions = {}
        for fm in gcl_commitment.failure_modes:
            failure_conditions[fm.name] = fm.condition.expression
        
        # Extract trigger condition
        trigger_condition = None
        if gcl_commitment.trigger_conditions:
            trigger_condition = gcl_commitment.trigger_conditions[0].expression
        
        return RuntimeCommitment(
            id=gcl_commitment.id,
            agent_id=agent_id or gcl_commitment.issuer,
            action_type=gcl_commitment.promised_behavior.action_type,
            description=gcl_commitment.promised_behavior.description or "",
            trigger_condition=trigger_condition,
            success_condition=gcl_commitment.success_condition.expression,
            failure_conditions=failure_conditions,
            stake=gcl_commitment.stake,
            confidence=gcl_commitment.confidence,
            timeout_seconds=gcl_commitment.promised_behavior.timeout_seconds,
            created_at=gcl_commitment.created_at,
            expires_at=gcl_commitment.expires_at,
            metadata=gcl_commitment.metadata,
        )
    
    def from_gcl_result(
        self,
        gcl_result: GCLVerificationResult,
    ) -> CommitmentResult:
        """
        Convert a GCL VerificationResult to a CommitmentResult.
        
        Args:
            gcl_result: The GCL verification result
            
        Returns:
            CommitmentResult
        """
        # Map GCL status to runtime status
        status_map = {
            GCLVerificationStatus.SUCCESS: CommitmentStatus.FULFILLED,
            GCLVerificationStatus.FAILURE: CommitmentStatus.FAILED,
            GCLVerificationStatus.TIMEOUT: CommitmentStatus.EXPIRED,
            GCLVerificationStatus.ERROR: CommitmentStatus.FAILED,
            GCLVerificationStatus.PENDING: CommitmentStatus.PENDING,
        }
        
        return CommitmentResult(
            commitment_id=gcl_result.commitment_id,
            status=status_map.get(gcl_result.status, CommitmentStatus.FAILED),
            success=gcl_result.is_success,
            failure_mode=gcl_result.triggered_failure_mode,
            details=gcl_result.details,
            timestamp=gcl_result.timestamp,
            duration_ms=gcl_result.verification_duration_ms,
        )
    
    # Verification
    
    def verify(
        self,
        commitment: RuntimeCommitment | GroundedCommitment,
        pre_state: dict[str, Any],
        post_state: dict[str, Any],
        action: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> CommitmentResult | None:
        """
        Verify a commitment using the GCL verification engine.
        
        Args:
            commitment: The commitment to verify (runtime or GCL)
            pre_state: State before the action
            post_state: State after the action
            action: The action that was performed
            metadata: Additional context
            
        Returns:
            CommitmentResult, or None if GCL not available
        """
        if not self._available or self._engine is None:
            return None
        
        # Convert to GCL commitment if needed
        if isinstance(commitment, RuntimeCommitment):
            gcl_commitment = self.to_gcl(commitment)
            if gcl_commitment is None:
                return None
        else:
            gcl_commitment = commitment
        
        # Create verification context
        context = GCLVerificationContext(
            pre_state=pre_state,
            post_state=post_state,
            action=action or {},
            metadata=metadata or {},
        )
        
        # Verify
        gcl_result = self._engine.verify(gcl_commitment, context)
        
        # Convert result
        return self.from_gcl_result(gcl_result)
    
    def verify_batch(
        self,
        commitments: list[RuntimeCommitment],
        pre_state: dict[str, Any],
        post_state: dict[str, Any],
        action: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[CommitmentResult]:
        """
        Verify multiple commitments.
        
        Args:
            commitments: List of commitments to verify
            pre_state: State before the action
            post_state: State after the action
            action: The action that was performed
            metadata: Additional context
            
        Returns:
            List of CommitmentResults
        """
        if not self._available or self._engine is None:
            return []
        
        # Convert commitments
        gcl_commitments = []
        for c in commitments:
            gcl_c = self.to_gcl(c)
            if gcl_c is not None:
                gcl_commitments.append(gcl_c)
        
        if not gcl_commitments:
            return []
        
        # Create context
        context = GCLVerificationContext(
            pre_state=pre_state,
            post_state=post_state,
            action=action or {},
            metadata=metadata or {},
        )
        
        # Verify batch
        batch_result = self._engine.verify_batch(gcl_commitments, context)
        
        # Convert results
        return [self.from_gcl_result(r) for r in batch_result.results]
    
    # Portfolio Management
    
    def get_portfolio(self, agent_id: str) -> CommitmentPortfolio | None:
        """
        Get or create a portfolio for an agent.
        
        Args:
            agent_id: The agent ID
            
        Returns:
            CommitmentPortfolio, or None if GCL not available
        """
        if not self._available:
            return None
        
        if agent_id not in self._portfolios:
            self._portfolios[agent_id] = CommitmentPortfolio(
                owner=agent_id,
            )
        
        return self._portfolios[agent_id]
    
    def add_to_portfolio(
        self,
        agent_id: str,
        commitment: RuntimeCommitment,
    ) -> bool:
        """
        Add a commitment to an agent's portfolio.
        
        Args:
            agent_id: The agent ID
            commitment: The commitment to add
            
        Returns:
            True if added successfully, False otherwise
        """
        portfolio = self.get_portfolio(agent_id)
        if portfolio is None:
            return False
        
        gcl_commitment = self.to_gcl(commitment)
        if gcl_commitment is None:
            return False
        
        try:
            portfolio.add_commitment(gcl_commitment)
            return True
        except ValueError:
            # Portfolio constraints violated
            return False
    
    def remove_from_portfolio(
        self,
        agent_id: str,
        commitment_id: str,
    ) -> bool:
        """
        Remove a commitment from an agent's portfolio.
        
        Args:
            agent_id: The agent ID
            commitment_id: The commitment ID to remove
            
        Returns:
            True if removed, False otherwise
        """
        portfolio = self.get_portfolio(agent_id)
        if portfolio is None:
            return False
        
        return portfolio.remove_commitment(commitment_id)
    
    def verify_portfolio(
        self,
        agent_id: str,
        pre_state: dict[str, Any],
        post_state: dict[str, Any],
        action: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        only_triggered: bool = True,
    ) -> list[CommitmentResult]:
        """
        Verify all commitments in an agent's portfolio.
        
        Args:
            agent_id: The agent ID
            pre_state: State before the action
            post_state: State after the action
            action: The action that was performed
            metadata: Additional context
            only_triggered: Only verify triggered commitments
            
        Returns:
            List of CommitmentResults
        """
        if not self._available or self._engine is None:
            return []
        
        portfolio = self.get_portfolio(agent_id)
        if portfolio is None:
            return []
        
        context = GCLVerificationContext(
            pre_state=pre_state,
            post_state=post_state,
            action=action or {},
            metadata=metadata or {},
        )
        
        batch_result = self._engine.verify_portfolio(
            portfolio,
            context,
            only_triggered=only_triggered,
        )
        
        return [self.from_gcl_result(r) for r in batch_result.results]
    
    def get_portfolio_stats(self, agent_id: str) -> dict[str, Any]:
        """
        Get statistics for an agent's portfolio.
        
        Args:
            agent_id: The agent ID
            
        Returns:
            Dictionary with portfolio statistics
        """
        portfolio = self.get_portfolio(agent_id)
        if portfolio is None:
            return {
                "available": False,
                "total_commitments": 0,
                "active_commitments": 0,
                "total_stake": 0.0,
                "average_confidence": 0.0,
                "expected_value": 0.0,
            }
        
        return {
            "available": True,
            "total_commitments": len(portfolio),
            "active_commitments": len(portfolio.active_commitments),
            "expired_commitments": len(portfolio.expired_commitments),
            "total_stake": portfolio.total_stake,
            "average_confidence": portfolio.average_confidence,
            "expected_value": portfolio.expected_portfolio_value,
            "remaining_budget": portfolio.remaining_stake_budget(),
        }
    
    # Utility Methods
    
    def create_predicate(
        self,
        name: str,
        expression: str,
    ) -> Predicate | None:
        """
        Create a GCL Predicate.
        
        Args:
            name: Predicate name
            expression: Predicate expression
            
        Returns:
            Predicate, or None if GCL not available
        """
        if not self._available:
            return None
        
        return Predicate(name=name, expression=expression)
    
    def evaluate_predicate(
        self,
        expression: str,
        context: dict[str, Any],
    ) -> bool | None:
        """
        Evaluate a predicate expression.
        
        Args:
            expression: The predicate expression
            context: The evaluation context
            
        Returns:
            Evaluation result, or None if GCL not available
        """
        if not self._available:
            return None
        
        predicate = Predicate(name="eval", expression=expression)
        return predicate.evaluate(context)


# Global bridge instance
_bridge: GCLBridge | None = None


def get_bridge() -> GCLBridge:
    """Get the global GCL bridge instance."""
    global _bridge
    if _bridge is None:
        _bridge = GCLBridge()
    return _bridge


def is_gcl_available() -> bool:
    """Check if GCL integration is available."""
    return GCL_AVAILABLE
