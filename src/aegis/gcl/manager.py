"""
Commitment manager for the agent runtime.

This module provides commitment lifecycle management, including
creation, tracking, verification, and cleanup.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone, timedelta
from typing import Any, Callable, Awaitable
from collections import defaultdict

from pydantic import BaseModel, Field, ConfigDict

from .models import (
    RuntimeCommitment,
    CommitmentStatus,
    CommitmentResult,
    ToolCommitment,
    CommitmentTemplate,
)
from .verifier import (
    RuntimeVerifier,
    VerificationContext,
    VerificationHook,
)


class CommitmentPolicy(BaseModel):
    """
    Policy for commitment management.
    
    Attributes:
        max_active_commitments: Maximum active commitments per agent
        max_total_stake: Maximum total stake per agent
        default_timeout_seconds: Default commitment timeout
        auto_expire_check_interval: Interval for checking expired commitments
        require_stake: Whether commitments require stake
        min_confidence: Minimum confidence level required
    """
    
    model_config = ConfigDict(frozen=True)
    
    max_active_commitments: int = Field(
        default=100,
        ge=1,
        description="Maximum active commitments per agent",
    )
    max_total_stake: float = Field(
        default=100.0,
        ge=0,
        description="Maximum total stake per agent",
    )
    default_timeout_seconds: float = Field(
        default=300.0,
        ge=0,
        description="Default commitment timeout",
    )
    auto_expire_check_interval: float = Field(
        default=60.0,
        ge=1,
        description="Interval for checking expired commitments",
    )
    require_stake: bool = Field(
        default=False,
        description="Whether commitments require stake",
    )
    min_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum confidence level required",
    )


class CommitmentStats(BaseModel):
    """Statistics for commitment tracking."""
    
    model_config = ConfigDict(frozen=True)
    
    total_created: int = 0
    total_fulfilled: int = 0
    total_failed: int = 0
    total_expired: int = 0
    total_cancelled: int = 0
    active_count: int = 0
    total_stake: float = 0.0
    average_confidence: float = 0.0
    fulfillment_rate: float = 0.0


class CommitmentManager:
    """
    Manager for commitment lifecycle.
    
    The CommitmentManager handles:
    - Creating and registering commitments
    - Tracking active commitments
    - Verifying commitments
    - Cleaning up expired commitments
    - Maintaining commitment statistics
    
    Example:
        >>> manager = CommitmentManager()
        >>> commitment = await manager.create_commitment(
        ...     agent_id="agent-1",
        ...     action_type="tool:search",
        ...     description="Execute search",
        ...     success_condition="tool_success",
        ... )
        >>> result = await manager.verify_commitment(
        ...     commitment.id,
        ...     context,
        ... )
    """
    
    def __init__(
        self,
        policy: CommitmentPolicy | None = None,
        verifier: RuntimeVerifier | None = None,
    ) -> None:
        """
        Initialize the commitment manager.
        
        Args:
            policy: Commitment policy
            verifier: Commitment verifier
        """
        self.policy = policy or CommitmentPolicy()
        self.verifier = verifier or RuntimeVerifier()
        
        # Commitment storage
        self._commitments: dict[str, RuntimeCommitment] = {}
        self._by_agent: dict[str, list[str]] = defaultdict(list)
        self._by_status: dict[CommitmentStatus, list[str]] = defaultdict(list)
        
        # Statistics
        self._stats = {
            "total_created": 0,
            "total_fulfilled": 0,
            "total_failed": 0,
            "total_expired": 0,
            "total_cancelled": 0,
        }
        
        # Callbacks
        self._on_fulfilled: list[Callable[[RuntimeCommitment, CommitmentResult], Awaitable[None]]] = []
        self._on_failed: list[Callable[[RuntimeCommitment, CommitmentResult], Awaitable[None]]] = []
        self._on_expired: list[Callable[[RuntimeCommitment], Awaitable[None]]] = []
        
        # Background task
        self._expire_task: asyncio.Task[None] | None = None
    
    # Commitment creation
    
    async def create_commitment(
        self,
        agent_id: str,
        action_type: str,
        description: str,
        success_condition: str,
        failure_conditions: dict[str, str] | None = None,
        trigger_condition: str | None = None,
        stake: float = 1.0,
        confidence: float = 0.8,
        timeout_seconds: float | None = None,
        tool_call_id: str | None = None,
        invocation_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> RuntimeCommitment:
        """
        Create and register a new commitment.
        
        Args:
            agent_id: ID of the agent making the commitment
            action_type: Type of action
            description: Human-readable description
            success_condition: Success condition expression
            failure_conditions: Failure condition expressions
            trigger_condition: Trigger condition expression
            stake: Amount to stake
            confidence: Confidence level
            timeout_seconds: Timeout (uses policy default if None)
            tool_call_id: Associated tool call ID
            invocation_id: Associated invocation ID
            metadata: Additional metadata
            
        Returns:
            The created commitment
            
        Raises:
            ValueError: If policy constraints are violated
        """
        # Check policy constraints
        await self._check_policy_constraints(agent_id, stake, confidence)
        
        # Calculate expiration
        timeout = timeout_seconds or self.policy.default_timeout_seconds
        expires_at = datetime.now(timezone.utc) + timedelta(seconds=timeout)
        
        # Create commitment
        commitment = RuntimeCommitment(
            agent_id=agent_id,
            action_type=action_type,
            description=description,
            trigger_condition=trigger_condition,
            success_condition=success_condition,
            failure_conditions=failure_conditions or {},
            stake=stake,
            confidence=confidence,
            timeout_seconds=timeout,
            expires_at=expires_at,
            status=CommitmentStatus.PENDING,
            tool_call_id=tool_call_id,
            invocation_id=invocation_id,
            metadata=metadata or {},
        )
        
        # Register commitment
        self._register_commitment(commitment)
        
        return commitment
    
    async def create_tool_commitment(
        self,
        agent_id: str,
        tool_commitment: ToolCommitment,
        tool_call_id: str | None = None,
        invocation_id: str | None = None,
        stake: float = 1.0,
        confidence: float = 0.8,
    ) -> RuntimeCommitment:
        """
        Create a commitment for tool execution.
        
        Args:
            agent_id: ID of the agent
            tool_commitment: Tool commitment specification
            tool_call_id: Associated tool call ID
            invocation_id: Associated invocation ID
            stake: Amount to stake
            confidence: Confidence level
            
        Returns:
            The created commitment
        """
        commitment = tool_commitment.to_runtime_commitment(
            agent_id=agent_id,
            tool_call_id=tool_call_id,
            invocation_id=invocation_id,
            stake=stake,
            confidence=confidence,
        )
        
        # Check policy and register
        await self._check_policy_constraints(agent_id, stake, confidence)
        self._register_commitment(commitment)
        
        return commitment
    
    async def create_from_template(
        self,
        agent_id: str,
        template: CommitmentTemplate,
        params: dict[str, Any],
        stake: float | None = None,
        confidence: float | None = None,
        timeout_seconds: float | None = None,
    ) -> RuntimeCommitment:
        """
        Create a commitment from a template.
        
        Args:
            agent_id: ID of the agent
            template: Commitment template
            params: Template parameters
            stake: Override default stake
            confidence: Override default confidence
            timeout_seconds: Override default timeout
            
        Returns:
            The created commitment
        """
        commitment = template.instantiate(
            agent_id=agent_id,
            params=params,
            stake=stake,
            confidence=confidence,
            timeout_seconds=timeout_seconds,
        )
        
        # Check policy and register
        await self._check_policy_constraints(
            agent_id,
            commitment.stake,
            commitment.confidence,
        )
        self._register_commitment(commitment)
        
        return commitment
    
    # Commitment retrieval
    
    def get_commitment(self, commitment_id: str) -> RuntimeCommitment | None:
        """Get a commitment by ID."""
        return self._commitments.get(commitment_id)
    
    def get_agent_commitments(
        self,
        agent_id: str,
        status: CommitmentStatus | None = None,
    ) -> list[RuntimeCommitment]:
        """
        Get all commitments for an agent.
        
        Args:
            agent_id: Agent ID
            status: Filter by status (None for all)
            
        Returns:
            List of commitments
        """
        commitment_ids = self._by_agent.get(agent_id, [])
        commitments = [
            self._commitments[cid]
            for cid in commitment_ids
            if cid in self._commitments
        ]
        
        if status is not None:
            commitments = [c for c in commitments if c.status == status]
        
        return commitments
    
    def get_active_commitments(self, agent_id: str | None = None) -> list[RuntimeCommitment]:
        """
        Get all active commitments.
        
        Args:
            agent_id: Filter by agent (None for all)
            
        Returns:
            List of active commitments
        """
        if agent_id:
            return self.get_agent_commitments(agent_id, CommitmentStatus.ACTIVE)
        
        commitment_ids = self._by_status.get(CommitmentStatus.ACTIVE, [])
        return [
            self._commitments[cid]
            for cid in commitment_ids
            if cid in self._commitments
        ]
    
    def get_pending_commitments(self, agent_id: str | None = None) -> list[RuntimeCommitment]:
        """Get all pending commitments."""
        if agent_id:
            return self.get_agent_commitments(agent_id, CommitmentStatus.PENDING)
        
        commitment_ids = self._by_status.get(CommitmentStatus.PENDING, [])
        return [
            self._commitments[cid]
            for cid in commitment_ids
            if cid in self._commitments
        ]
    
    # Commitment lifecycle
    
    async def activate_commitment(self, commitment_id: str) -> RuntimeCommitment | None:
        """
        Activate a pending commitment.
        
        Args:
            commitment_id: ID of the commitment to activate
            
        Returns:
            The activated commitment, or None if not found
        """
        commitment = self._commitments.get(commitment_id)
        if commitment is None:
            return None
        
        if commitment.status != CommitmentStatus.PENDING:
            return commitment
        
        # Update status
        activated = commitment.activate()
        self._update_commitment(activated)
        
        return activated
    
    async def verify_commitment(
        self,
        commitment_id: str,
        context: VerificationContext,
    ) -> CommitmentResult | None:
        """
        Verify a commitment.
        
        Args:
            commitment_id: ID of the commitment to verify
            context: Verification context
            
        Returns:
            Verification result, or None if commitment not found
        """
        commitment = self._commitments.get(commitment_id)
        if commitment is None:
            return None
        
        # Verify
        result = await self.verifier.verify(commitment, context)
        
        # Update commitment status based on result
        if result.success:
            updated = commitment.fulfill()
            self._stats["total_fulfilled"] += 1
            
            # Call fulfilled callbacks
            for callback in self._on_fulfilled:
                try:
                    await callback(commitment, result)
                except Exception:
                    pass
        else:
            updated = commitment.fail(result.failure_mode)
            self._stats["total_failed"] += 1
            
            # Call failed callbacks
            for callback in self._on_failed:
                try:
                    await callback(commitment, result)
                except Exception:
                    pass
        
        self._update_commitment(updated)
        
        return result
    
    async def cancel_commitment(self, commitment_id: str) -> RuntimeCommitment | None:
        """
        Cancel a commitment.
        
        Args:
            commitment_id: ID of the commitment to cancel
            
        Returns:
            The cancelled commitment, or None if not found
        """
        commitment = self._commitments.get(commitment_id)
        if commitment is None:
            return None
        
        if commitment.is_terminal():
            return commitment
        
        cancelled = commitment.cancel()
        self._update_commitment(cancelled)
        self._stats["total_cancelled"] += 1
        
        return cancelled
    
    # Expiration handling
    
    async def check_expired(self) -> list[RuntimeCommitment]:
        """
        Check for and handle expired commitments.
        
        Returns:
            List of newly expired commitments
        """
        expired: list[RuntimeCommitment] = []
        
        # Check active and pending commitments
        for status in (CommitmentStatus.ACTIVE, CommitmentStatus.PENDING):
            commitment_ids = list(self._by_status.get(status, []))
            
            for cid in commitment_ids:
                commitment = self._commitments.get(cid)
                if commitment and commitment.is_expired():
                    # Mark as expired
                    updated = RuntimeCommitment(
                        **{**commitment.model_dump(), "status": CommitmentStatus.EXPIRED}
                    )
                    self._update_commitment(updated)
                    self._stats["total_expired"] += 1
                    expired.append(updated)
                    
                    # Call expired callbacks
                    for callback in self._on_expired:
                        try:
                            await callback(updated)
                        except Exception:
                            pass
        
        return expired
    
    async def start_expiration_checker(self) -> None:
        """Start the background expiration checker."""
        if self._expire_task is not None:
            return
        
        async def check_loop():
            while True:
                await asyncio.sleep(self.policy.auto_expire_check_interval)
                await self.check_expired()
        
        self._expire_task = asyncio.create_task(check_loop())
    
    async def stop_expiration_checker(self) -> None:
        """Stop the background expiration checker."""
        if self._expire_task is not None:
            self._expire_task.cancel()
            try:
                await self._expire_task
            except asyncio.CancelledError:
                pass
            self._expire_task = None
    
    # Statistics
    
    def get_stats(self, agent_id: str | None = None) -> CommitmentStats:
        """
        Get commitment statistics.
        
        Args:
            agent_id: Filter by agent (None for all)
            
        Returns:
            CommitmentStats
        """
        if agent_id:
            commitments = self.get_agent_commitments(agent_id)
        else:
            commitments = list(self._commitments.values())
        
        active = [c for c in commitments if c.status == CommitmentStatus.ACTIVE]
        fulfilled = [c for c in commitments if c.status == CommitmentStatus.FULFILLED]
        failed = [c for c in commitments if c.status == CommitmentStatus.FAILED]
        expired = [c for c in commitments if c.status == CommitmentStatus.EXPIRED]
        cancelled = [c for c in commitments if c.status == CommitmentStatus.CANCELLED]
        
        total_completed = len(fulfilled) + len(failed)
        fulfillment_rate = len(fulfilled) / total_completed if total_completed > 0 else 0.0
        
        total_stake = sum(c.stake for c in active)
        avg_confidence = (
            sum(c.confidence for c in active) / len(active)
            if active else 0.0
        )
        
        return CommitmentStats(
            total_created=len(commitments),
            total_fulfilled=len(fulfilled),
            total_failed=len(failed),
            total_expired=len(expired),
            total_cancelled=len(cancelled),
            active_count=len(active),
            total_stake=total_stake,
            average_confidence=avg_confidence,
            fulfillment_rate=fulfillment_rate,
        )
    
    def get_agent_stake(self, agent_id: str) -> float:
        """Get total stake for an agent."""
        commitments = self.get_agent_commitments(agent_id, CommitmentStatus.ACTIVE)
        return sum(c.stake for c in commitments)
    
    # Callbacks
    
    def on_fulfilled(
        self,
        callback: Callable[[RuntimeCommitment, CommitmentResult], Awaitable[None]],
    ) -> None:
        """Register a callback for fulfilled commitments."""
        self._on_fulfilled.append(callback)
    
    def on_failed(
        self,
        callback: Callable[[RuntimeCommitment, CommitmentResult], Awaitable[None]],
    ) -> None:
        """Register a callback for failed commitments."""
        self._on_failed.append(callback)
    
    def on_expired(
        self,
        callback: Callable[[RuntimeCommitment], Awaitable[None]],
    ) -> None:
        """Register a callback for expired commitments."""
        self._on_expired.append(callback)
    
    # Internal methods
    
    def _register_commitment(self, commitment: RuntimeCommitment) -> None:
        """Register a commitment in storage."""
        self._commitments[commitment.id] = commitment
        self._by_agent[commitment.agent_id].append(commitment.id)
        self._by_status[commitment.status].append(commitment.id)
        self._stats["total_created"] += 1
    
    def _update_commitment(self, commitment: RuntimeCommitment) -> None:
        """Update a commitment in storage."""
        old = self._commitments.get(commitment.id)
        if old and old.status != commitment.status:
            # Remove from old status index
            if commitment.id in self._by_status[old.status]:
                self._by_status[old.status].remove(commitment.id)
            # Add to new status index
            self._by_status[commitment.status].append(commitment.id)
        
        self._commitments[commitment.id] = commitment
    
    async def _check_policy_constraints(
        self,
        agent_id: str,
        stake: float,
        confidence: float,
    ) -> None:
        """
        Check policy constraints for a new commitment.
        
        Raises:
            ValueError: If constraints are violated
        """
        # Check confidence
        if confidence < self.policy.min_confidence:
            raise ValueError(
                f"Confidence {confidence} below minimum {self.policy.min_confidence}"
            )
        
        # Check stake requirement
        if self.policy.require_stake and stake <= 0:
            raise ValueError("Stake is required but not provided")
        
        # Check active commitment limit
        active = self.get_agent_commitments(agent_id, CommitmentStatus.ACTIVE)
        if len(active) >= self.policy.max_active_commitments:
            raise ValueError(
                f"Agent has reached maximum active commitments "
                f"({self.policy.max_active_commitments})"
            )
        
        # Check total stake limit
        current_stake = self.get_agent_stake(agent_id)
        if current_stake + stake > self.policy.max_total_stake:
            raise ValueError(
                f"Adding stake {stake} would exceed maximum "
                f"({current_stake + stake} > {self.policy.max_total_stake})"
            )
    
    # Cleanup
    
    async def cleanup_terminal(self, max_age_seconds: float = 3600) -> int:
        """
        Clean up old terminal commitments.
        
        Args:
            max_age_seconds: Maximum age for terminal commitments
            
        Returns:
            Number of commitments removed
        """
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=max_age_seconds)
        removed = 0
        
        for status in (
            CommitmentStatus.FULFILLED,
            CommitmentStatus.FAILED,
            CommitmentStatus.EXPIRED,
            CommitmentStatus.CANCELLED,
        ):
            commitment_ids = list(self._by_status.get(status, []))
            
            for cid in commitment_ids:
                commitment = self._commitments.get(cid)
                if commitment and commitment.created_at < cutoff:
                    # Remove from storage
                    del self._commitments[cid]
                    self._by_status[status].remove(cid)
                    if cid in self._by_agent[commitment.agent_id]:
                        self._by_agent[commitment.agent_id].remove(cid)
                    removed += 1
        
        return removed
