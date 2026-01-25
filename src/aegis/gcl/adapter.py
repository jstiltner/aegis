"""
GCL adapter for bridging agent runtime with the GCL framework.

This module provides the main adapter class that connects the agent
runtime's commitment system with the GCL framework.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, TYPE_CHECKING

from pydantic import BaseModel, Field, ConfigDict

from .models import (
    RuntimeCommitment,
    CommitmentStatus,
    CommitmentResult,
    ToolCommitment,
)
from .verifier import (
    RuntimeVerifier,
    VerificationContext,
)
from .manager import (
    CommitmentManager,
    CommitmentPolicy,
)

# Optional GCL imports
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
    )
    GCL_AVAILABLE = True
except ImportError:
    GCL_AVAILABLE = False


class CommitmentConfig(BaseModel):
    """
    Configuration for commitment handling.
    
    Attributes:
        enable_gcl_integration: Whether to use GCL framework when available
        default_stake: Default stake for commitments
        default_confidence: Default confidence level
        default_timeout_seconds: Default timeout
        auto_verify_on_tool_complete: Auto-verify tool commitments
        log_commitments: Whether to log commitment events
    """
    
    model_config = ConfigDict(frozen=True)
    
    enable_gcl_integration: bool = Field(
        default=True,
        description="Whether to use GCL framework when available",
    )
    default_stake: float = Field(
        default=1.0,
        ge=0.0,
        description="Default stake for commitments",
    )
    default_confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Default confidence level",
    )
    default_timeout_seconds: float = Field(
        default=300.0,
        ge=0,
        description="Default timeout",
    )
    auto_verify_on_tool_complete: bool = Field(
        default=True,
        description="Auto-verify tool commitments when tools complete",
    )
    log_commitments: bool = Field(
        default=True,
        description="Whether to log commitment events",
    )


class CommitmentContext(BaseModel):
    """
    Context for commitment operations.
    
    Attributes:
        agent_id: ID of the agent
        session_id: ID of the session
        tool_name: Name of the tool (if applicable)
        tool_call_id: Tool call ID (if applicable)
        invocation_id: Invocation ID (if applicable)
        metadata: Additional context data
    """
    
    model_config = ConfigDict(frozen=True)
    
    agent_id: str = Field(..., description="ID of the agent")
    session_id: str | None = Field(default=None, description="ID of the session")
    tool_name: str | None = Field(default=None, description="Name of the tool")
    tool_call_id: str | None = Field(default=None, description="Tool call ID")
    invocation_id: str | None = Field(default=None, description="Invocation ID")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional context")


class GCLAdapter:
    """
    Adapter for integrating GCL with the agent runtime.
    
    The GCLAdapter provides:
    - Conversion between runtime and GCL commitment formats
    - Unified commitment management
    - Integration with GCL verification when available
    - Audit logging for commitments
    
    Example:
        >>> adapter = GCLAdapter()
        >>> commitment = await adapter.create_commitment(
        ...     context=CommitmentContext(agent_id="agent-1"),
        ...     action_type="tool:search",
        ...     description="Execute search",
        ...     success_condition="tool_success",
        ... )
        >>> result = await adapter.verify_commitment(
        ...     commitment.id,
        ...     post_state={"tool_success": True},
        ... )
    """
    
    def __init__(
        self,
        config: CommitmentConfig | None = None,
        manager: CommitmentManager | None = None,
        logger: Any = None,
    ) -> None:
        """
        Initialize the GCL adapter.
        
        Args:
            config: Commitment configuration
            manager: Commitment manager (creates default if None)
            logger: Audit logger for commitment events
        """
        self.config = config or CommitmentConfig()
        self.manager = manager or CommitmentManager()
        self.logger = logger
        
        # GCL integration
        self._gcl_available = GCL_AVAILABLE and self.config.enable_gcl_integration
        self._gcl_engine = None
        self._gcl_portfolios: dict[str, Any] = {}  # agent_id -> CommitmentPortfolio
        
        if self._gcl_available:
            try:
                self._gcl_engine = VerificationEngine()
            except Exception:
                self._gcl_available = False
    
    @property
    def gcl_available(self) -> bool:
        """Check if GCL integration is available."""
        return self._gcl_available
    
    # Commitment creation
    
    async def create_commitment(
        self,
        context: CommitmentContext,
        action_type: str,
        description: str,
        success_condition: str,
        failure_conditions: dict[str, str] | None = None,
        trigger_condition: str | None = None,
        stake: float | None = None,
        confidence: float | None = None,
        timeout_seconds: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> RuntimeCommitment:
        """
        Create a new commitment.
        
        Args:
            context: Commitment context
            action_type: Type of action
            description: Human-readable description
            success_condition: Success condition expression
            failure_conditions: Failure condition expressions
            trigger_condition: Trigger condition expression
            stake: Amount to stake (uses config default if None)
            confidence: Confidence level (uses config default if None)
            timeout_seconds: Timeout (uses config default if None)
            metadata: Additional metadata
            
        Returns:
            The created commitment
        """
        commitment = await self.manager.create_commitment(
            agent_id=context.agent_id,
            action_type=action_type,
            description=description,
            success_condition=success_condition,
            failure_conditions=failure_conditions,
            trigger_condition=trigger_condition,
            stake=stake if stake is not None else self.config.default_stake,
            confidence=confidence if confidence is not None else self.config.default_confidence,
            timeout_seconds=timeout_seconds if timeout_seconds is not None else self.config.default_timeout_seconds,
            tool_call_id=context.tool_call_id,
            invocation_id=context.invocation_id,
            metadata={
                **(metadata or {}),
                "session_id": context.session_id,
                "tool_name": context.tool_name,
            },
        )
        
        # Log commitment creation
        if self.config.log_commitments and self.logger:
            await self._log_commitment_created(commitment, context)
        
        # Create GCL commitment if available
        if self._gcl_available:
            await self._create_gcl_commitment(commitment, context)
        
        return commitment
    
    async def create_tool_commitment(
        self,
        context: CommitmentContext,
        tool_name: str,
        expected_outcome: str,
        max_duration_seconds: float = 30.0,
        must_succeed: bool = True,
        output_validation: str | None = None,
        stake: float | None = None,
        confidence: float | None = None,
    ) -> RuntimeCommitment:
        """
        Create a commitment for tool execution.
        
        Args:
            context: Commitment context
            tool_name: Name of the tool
            expected_outcome: Description of expected outcome
            max_duration_seconds: Maximum execution time
            must_succeed: Whether the tool must succeed
            output_validation: Optional output validation expression
            stake: Amount to stake
            confidence: Confidence level
            
        Returns:
            The created commitment
        """
        tool_commitment = ToolCommitment(
            tool_name=tool_name,
            expected_outcome=expected_outcome,
            max_duration_seconds=max_duration_seconds,
            must_succeed=must_succeed,
            output_validation=output_validation,
        )
        
        commitment = await self.manager.create_tool_commitment(
            agent_id=context.agent_id,
            tool_commitment=tool_commitment,
            tool_call_id=context.tool_call_id,
            invocation_id=context.invocation_id,
            stake=stake if stake is not None else self.config.default_stake,
            confidence=confidence if confidence is not None else self.config.default_confidence,
        )
        
        # Update metadata
        commitment.metadata["session_id"] = context.session_id
        commitment.metadata["tool_name"] = tool_name
        
        # Log commitment creation
        if self.config.log_commitments and self.logger:
            await self._log_commitment_created(commitment, context)
        
        return commitment
    
    # Commitment verification
    
    async def verify_commitment(
        self,
        commitment_id: str,
        pre_state: dict[str, Any] | None = None,
        post_state: dict[str, Any] | None = None,
        action: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> CommitmentResult | None:
        """
        Verify a commitment.
        
        Args:
            commitment_id: ID of the commitment to verify
            pre_state: State before the action
            post_state: State after the action
            action: The action that was performed
            metadata: Additional context
            
        Returns:
            Verification result, or None if commitment not found
        """
        context = VerificationContext(
            pre_state=pre_state or {},
            post_state=post_state or {},
            action=action or {},
            metadata=metadata or {},
        )
        
        result = await self.manager.verify_commitment(commitment_id, context)
        
        # Log verification result
        if result and self.config.log_commitments and self.logger:
            commitment = self.manager.get_commitment(commitment_id)
            if commitment:
                await self._log_commitment_verified(commitment, result)
        
        return result
    
    async def verify_tool_execution(
        self,
        commitment_id: str,
        tool_name: str,
        success: bool,
        duration_seconds: float,
        output: Any = None,
        error: str | None = None,
    ) -> CommitmentResult | None:
        """
        Verify a tool execution commitment.
        
        Args:
            commitment_id: ID of the commitment
            tool_name: Name of the executed tool
            success: Whether the tool succeeded
            duration_seconds: Execution duration
            output: Tool output
            error: Error message (if any)
            
        Returns:
            Verification result
        """
        return await self.verify_commitment(
            commitment_id=commitment_id,
            post_state={
                "tool_name": tool_name,
                "tool_success": success,
                "tool_error": error is not None,
                "duration_seconds": duration_seconds,
                "output": output,
            },
            action={
                "type": "tool_execution",
                "tool_name": tool_name,
            },
            metadata={
                "error": error,
            },
        )
    
    # Commitment lifecycle
    
    async def activate_commitment(self, commitment_id: str) -> RuntimeCommitment | None:
        """Activate a pending commitment."""
        return await self.manager.activate_commitment(commitment_id)
    
    async def cancel_commitment(self, commitment_id: str) -> RuntimeCommitment | None:
        """Cancel a commitment."""
        commitment = await self.manager.cancel_commitment(commitment_id)
        
        if commitment and self.config.log_commitments and self.logger:
            await self._log_commitment_cancelled(commitment)
        
        return commitment
    
    # Commitment retrieval
    
    def get_commitment(self, commitment_id: str) -> RuntimeCommitment | None:
        """Get a commitment by ID."""
        return self.manager.get_commitment(commitment_id)
    
    def get_active_commitments(self, agent_id: str) -> list[RuntimeCommitment]:
        """Get all active commitments for an agent."""
        return self.manager.get_active_commitments(agent_id)
    
    def get_pending_commitments(self, agent_id: str) -> list[RuntimeCommitment]:
        """Get all pending commitments for an agent."""
        return self.manager.get_pending_commitments(agent_id)
    
    def get_tool_commitments(
        self,
        agent_id: str,
        tool_name: str | None = None,
    ) -> list[RuntimeCommitment]:
        """
        Get tool-related commitments for an agent.
        
        Args:
            agent_id: Agent ID
            tool_name: Filter by tool name (None for all tools)
            
        Returns:
            List of tool commitments
        """
        commitments = self.manager.get_agent_commitments(agent_id)
        tool_commitments = [
            c for c in commitments
            if c.action_type.startswith("tool:")
        ]
        
        if tool_name:
            tool_commitments = [
                c for c in tool_commitments
                if c.metadata.get("tool_name") == tool_name
                or c.action_type == f"tool:{tool_name}"
            ]
        
        return tool_commitments
    
    # GCL conversion
    
    def to_gcl_commitment(
        self,
        commitment: RuntimeCommitment,
    ) -> Any | None:
        """
        Convert a runtime commitment to a GCL GroundedCommitment.
        
        Args:
            commitment: Runtime commitment
            
        Returns:
            GCL GroundedCommitment, or None if GCL not available
        """
        if not self._gcl_available:
            return None
        
        # Build trigger conditions
        trigger_conditions = []
        if commitment.trigger_condition:
            trigger_conditions.append(Predicate(
                name="trigger",
                expression=commitment.trigger_condition,
            ))
        else:
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
        
        # Ensure at least one failure mode
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
    
    def from_gcl_commitment(
        self,
        gcl_commitment: Any,
        agent_id: str | None = None,
    ) -> RuntimeCommitment:
        """
        Convert a GCL GroundedCommitment to a runtime commitment.
        
        Args:
            gcl_commitment: GCL GroundedCommitment
            agent_id: Override agent ID
            
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
    
    # Internal methods
    
    async def _create_gcl_commitment(
        self,
        commitment: RuntimeCommitment,
        context: CommitmentContext,
    ) -> None:
        """Create a GCL commitment and add to portfolio."""
        if not self._gcl_available:
            return
        
        gcl_commitment = self.to_gcl_commitment(commitment)
        if gcl_commitment is None:
            return
        
        # Get or create portfolio for agent
        if context.agent_id not in self._gcl_portfolios:
            self._gcl_portfolios[context.agent_id] = CommitmentPortfolio(
                owner=context.agent_id,
            )
        
        portfolio = self._gcl_portfolios[context.agent_id]
        try:
            portfolio.add_commitment(gcl_commitment)
        except ValueError:
            pass  # Portfolio constraints violated
    
    async def _log_commitment_created(
        self,
        commitment: RuntimeCommitment,
        context: CommitmentContext,
    ) -> None:
        """Log commitment creation."""
        if self.logger:
            try:
                from ..audit import AuditEventType
                await self.logger.log_event(
                    AuditEventType.COMMITMENT_CREATED,
                    f"Commitment created: {commitment.description}",
                    data={
                        "commitment_id": commitment.id,
                        "action_type": commitment.action_type,
                        "stake": commitment.stake,
                        "confidence": commitment.confidence,
                    },
                    agent_id=context.agent_id,
                    session_id=context.session_id,
                )
            except Exception:
                pass
    
    async def _log_commitment_verified(
        self,
        commitment: RuntimeCommitment,
        result: CommitmentResult,
    ) -> None:
        """Log commitment verification."""
        if self.logger:
            try:
                from ..audit import AuditEventType
                event_type = (
                    AuditEventType.COMMITMENT_VERIFIED
                    if result.success
                    else AuditEventType.COMMITMENT_FAILED
                )
                await self.logger.log_event(
                    event_type,
                    f"Commitment {'fulfilled' if result.success else 'failed'}: {commitment.description}",
                    data={
                        "commitment_id": commitment.id,
                        "success": result.success,
                        "failure_mode": result.failure_mode,
                    },
                    agent_id=commitment.agent_id,
                )
            except Exception:
                pass
    
    async def _log_commitment_cancelled(
        self,
        commitment: RuntimeCommitment,
    ) -> None:
        """Log commitment cancellation."""
        if self.logger:
            try:
                from ..audit import AuditEventType
                await self.logger.log_event(
                    AuditEventType.COMMITMENT_EXPIRED,  # Using expired for cancelled
                    f"Commitment cancelled: {commitment.description}",
                    data={
                        "commitment_id": commitment.id,
                        "action_type": commitment.action_type,
                    },
                    agent_id=commitment.agent_id,
                )
            except Exception:
                pass
