"""
Multi-agent commitment protocol.

This module implements the commitment protocol for multi-agent coordination,
allowing agents to request, accept, reject, and fulfill commitments.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, ConfigDict

from .messaging import Message, MessageType, MessageBus, get_message_bus
from .registry import AgentRegistry, get_registry


class CommitmentState(str, Enum):
    """State of a commitment in the protocol."""
    
    PROPOSED = "proposed"  # Commitment has been proposed
    ACCEPTED = "accepted"  # Commitment has been accepted
    REJECTED = "rejected"  # Commitment was rejected
    ACTIVE = "active"      # Commitment is being fulfilled
    COMPLETED = "completed"  # Commitment was fulfilled
    FAILED = "failed"      # Commitment failed
    CANCELLED = "cancelled"  # Commitment was cancelled


class ProtocolCommitment(BaseModel):
    """
    A commitment in the multi-agent protocol.
    
    This represents a commitment between agents, tracking its state
    through the protocol lifecycle.
    
    Attributes:
        id: Unique commitment identifier
        debtor_id: ID of the agent making the commitment
        creditor_id: ID of the agent receiving the commitment
        action: The action being committed to
        condition: Condition that must hold for the commitment
        deadline: When the commitment must be fulfilled
        state: Current state of the commitment
        created_at: When the commitment was created
        updated_at: When the commitment was last updated
        metadata: Additional metadata
    """
    
    model_config = ConfigDict(frozen=False)
    
    id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique commitment identifier",
    )
    debtor_id: str = Field(..., description="ID of the committing agent")
    creditor_id: str = Field(..., description="ID of the receiving agent")
    action: str = Field(..., description="The action being committed to")
    condition: str = Field(
        default="true",
        description="Condition for the commitment",
    )
    deadline: datetime | None = Field(
        default=None,
        description="When the commitment must be fulfilled",
    )
    state: CommitmentState = Field(
        default=CommitmentState.PROPOSED,
        description="Current state",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When created",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When last updated",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata",
    )
    
    def is_active(self) -> bool:
        """Check if commitment is active."""
        return self.state in (CommitmentState.ACCEPTED, CommitmentState.ACTIVE)
    
    def is_terminal(self) -> bool:
        """Check if commitment is in a terminal state."""
        return self.state in (
            CommitmentState.COMPLETED,
            CommitmentState.FAILED,
            CommitmentState.CANCELLED,
            CommitmentState.REJECTED,
        )
    
    def is_overdue(self) -> bool:
        """Check if commitment is past its deadline."""
        if self.deadline is None:
            return False
        return datetime.now(timezone.utc) > self.deadline
    
    def update_state(self, new_state: CommitmentState) -> None:
        """Update the commitment state."""
        self.state = new_state
        self.updated_at = datetime.now(timezone.utc)


class CommitmentProtocol:
    """
    Protocol for managing commitments between agents.
    
    This implements the commitment lifecycle:
    1. Agent A proposes a commitment to Agent B
    2. Agent B accepts or rejects
    3. If accepted, Agent B works on fulfilling
    4. Agent B reports completion or failure
    
    Example:
        >>> protocol = CommitmentProtocol()
        >>> commitment = await protocol.propose(
        ...     debtor_id="agent-b",
        ...     creditor_id="agent-a",
        ...     action="search_web",
        ... )
        >>> await protocol.accept(commitment.id, "agent-b")
    """
    
    def __init__(
        self,
        registry: AgentRegistry | None = None,
        bus: MessageBus | None = None,
    ) -> None:
        """
        Initialize the protocol.
        
        Args:
            registry: Agent registry (uses global if None)
            bus: Message bus (uses global if None)
        """
        self.registry = registry or get_registry()
        self.bus = bus or get_message_bus()
        self._commitments: dict[str, ProtocolCommitment] = {}
        self._pending_responses: dict[str, asyncio.Future] = {}
    
    async def propose(
        self,
        debtor_id: str,
        creditor_id: str,
        action: str,
        condition: str = "true",
        deadline: datetime | None = None,
        timeout: float = 30.0,
        metadata: dict[str, Any] | None = None,
    ) -> ProtocolCommitment:
        """
        Propose a commitment.
        
        The creditor proposes that the debtor commit to an action.
        
        Args:
            debtor_id: ID of the agent to make the commitment
            creditor_id: ID of the agent requesting the commitment
            action: The action to commit to
            condition: Condition for the commitment
            deadline: When the commitment must be fulfilled
            timeout: Time to wait for response
            metadata: Additional metadata
            
        Returns:
            The created commitment
        """
        commitment = ProtocolCommitment(
            debtor_id=debtor_id,
            creditor_id=creditor_id,
            action=action,
            condition=condition,
            deadline=deadline,
            metadata=metadata or {},
        )
        
        self._commitments[commitment.id] = commitment
        
        # Send proposal message
        message = Message(
            type=MessageType.COMMITMENT_REQUEST,
            sender_id=creditor_id,
            recipient_id=debtor_id,
            content={
                "commitment_id": commitment.id,
                "action": action,
                "condition": condition,
                "deadline": deadline.isoformat() if deadline else None,
                "metadata": metadata or {},
            },
            correlation_id=commitment.id,
        )
        
        await self.bus.send(message)
        
        return commitment
    
    async def accept(
        self,
        commitment_id: str,
        agent_id: str,
    ) -> bool:
        """
        Accept a commitment.
        
        The debtor accepts the proposed commitment.
        
        Args:
            commitment_id: ID of the commitment
            agent_id: ID of the accepting agent
            
        Returns:
            True if accepted successfully
        """
        commitment = self._commitments.get(commitment_id)
        if not commitment:
            return False
        
        if commitment.debtor_id != agent_id:
            return False
        
        if commitment.state != CommitmentState.PROPOSED:
            return False
        
        commitment.update_state(CommitmentState.ACCEPTED)
        
        # Send acceptance message
        message = Message(
            type=MessageType.COMMITMENT_ACCEPT,
            sender_id=agent_id,
            recipient_id=commitment.creditor_id,
            content={
                "commitment_id": commitment_id,
            },
            correlation_id=commitment_id,
        )
        
        await self.bus.send(message)
        
        return True
    
    async def reject(
        self,
        commitment_id: str,
        agent_id: str,
        reason: str = "",
    ) -> bool:
        """
        Reject a commitment.
        
        The debtor rejects the proposed commitment.
        
        Args:
            commitment_id: ID of the commitment
            agent_id: ID of the rejecting agent
            reason: Reason for rejection
            
        Returns:
            True if rejected successfully
        """
        commitment = self._commitments.get(commitment_id)
        if not commitment:
            return False
        
        if commitment.debtor_id != agent_id:
            return False
        
        if commitment.state != CommitmentState.PROPOSED:
            return False
        
        commitment.update_state(CommitmentState.REJECTED)
        
        # Send rejection message
        message = Message(
            type=MessageType.COMMITMENT_REJECT,
            sender_id=agent_id,
            recipient_id=commitment.creditor_id,
            content={
                "commitment_id": commitment_id,
                "reason": reason,
            },
            correlation_id=commitment_id,
        )
        
        await self.bus.send(message)
        
        return True
    
    async def activate(
        self,
        commitment_id: str,
        agent_id: str,
    ) -> bool:
        """
        Activate a commitment (start working on it).
        
        Args:
            commitment_id: ID of the commitment
            agent_id: ID of the agent
            
        Returns:
            True if activated successfully
        """
        commitment = self._commitments.get(commitment_id)
        if not commitment:
            return False
        
        if commitment.debtor_id != agent_id:
            return False
        
        if commitment.state != CommitmentState.ACCEPTED:
            return False
        
        commitment.update_state(CommitmentState.ACTIVE)
        
        return True
    
    async def complete(
        self,
        commitment_id: str,
        agent_id: str,
        result: Any = None,
    ) -> bool:
        """
        Mark a commitment as completed.
        
        Args:
            commitment_id: ID of the commitment
            agent_id: ID of the completing agent
            result: Result of the commitment
            
        Returns:
            True if completed successfully
        """
        commitment = self._commitments.get(commitment_id)
        if not commitment:
            return False
        
        if commitment.debtor_id != agent_id:
            return False
        
        if not commitment.is_active():
            return False
        
        commitment.update_state(CommitmentState.COMPLETED)
        commitment.metadata["result"] = result
        
        # Send completion message
        message = Message(
            type=MessageType.COMMITMENT_COMPLETE,
            sender_id=agent_id,
            recipient_id=commitment.creditor_id,
            content={
                "commitment_id": commitment_id,
                "result": result,
            },
            correlation_id=commitment_id,
        )
        
        await self.bus.send(message)
        
        return True
    
    async def fail(
        self,
        commitment_id: str,
        agent_id: str,
        reason: str = "",
    ) -> bool:
        """
        Mark a commitment as failed.
        
        Args:
            commitment_id: ID of the commitment
            agent_id: ID of the failing agent
            reason: Reason for failure
            
        Returns:
            True if marked as failed successfully
        """
        commitment = self._commitments.get(commitment_id)
        if not commitment:
            return False
        
        if commitment.debtor_id != agent_id:
            return False
        
        if not commitment.is_active():
            return False
        
        commitment.update_state(CommitmentState.FAILED)
        commitment.metadata["failure_reason"] = reason
        
        # Send failure message
        message = Message(
            type=MessageType.COMMITMENT_FAILED,
            sender_id=agent_id,
            recipient_id=commitment.creditor_id,
            content={
                "commitment_id": commitment_id,
                "reason": reason,
            },
            correlation_id=commitment_id,
        )
        
        await self.bus.send(message)
        
        return True
    
    async def cancel(
        self,
        commitment_id: str,
        agent_id: str,
        reason: str = "",
    ) -> bool:
        """
        Cancel a commitment.
        
        Either party can cancel a commitment.
        
        Args:
            commitment_id: ID of the commitment
            agent_id: ID of the cancelling agent
            reason: Reason for cancellation
            
        Returns:
            True if cancelled successfully
        """
        commitment = self._commitments.get(commitment_id)
        if not commitment:
            return False
        
        # Either party can cancel
        if agent_id not in (commitment.debtor_id, commitment.creditor_id):
            return False
        
        if commitment.is_terminal():
            return False
        
        commitment.update_state(CommitmentState.CANCELLED)
        commitment.metadata["cancellation_reason"] = reason
        commitment.metadata["cancelled_by"] = agent_id
        
        # Notify the other party
        other_party = (
            commitment.creditor_id
            if agent_id == commitment.debtor_id
            else commitment.debtor_id
        )
        
        message = Message(
            type=MessageType.COMMITMENT_CANCEL,
            sender_id=agent_id,
            recipient_id=other_party,
            content={
                "commitment_id": commitment_id,
                "reason": reason,
            },
            correlation_id=commitment_id,
        )
        
        await self.bus.send(message)
        
        return True
    
    def get_commitment(self, commitment_id: str) -> ProtocolCommitment | None:
        """Get a commitment by ID."""
        return self._commitments.get(commitment_id)
    
    def get_commitments_for_agent(
        self,
        agent_id: str,
        as_debtor: bool = True,
        as_creditor: bool = True,
        states: list[CommitmentState] | None = None,
    ) -> list[ProtocolCommitment]:
        """
        Get commitments for an agent.
        
        Args:
            agent_id: ID of the agent
            as_debtor: Include commitments where agent is debtor
            as_creditor: Include commitments where agent is creditor
            states: Filter by states (None for all)
            
        Returns:
            List of matching commitments
        """
        commitments = []
        
        for commitment in self._commitments.values():
            # Check role
            is_debtor = commitment.debtor_id == agent_id
            is_creditor = commitment.creditor_id == agent_id
            
            if not ((as_debtor and is_debtor) or (as_creditor and is_creditor)):
                continue
            
            # Check state
            if states and commitment.state not in states:
                continue
            
            commitments.append(commitment)
        
        return commitments
    
    def get_active_commitments(self) -> list[ProtocolCommitment]:
        """Get all active commitments."""
        return [
            c for c in self._commitments.values()
            if c.is_active()
        ]
    
    def get_overdue_commitments(self) -> list[ProtocolCommitment]:
        """Get all overdue commitments."""
        return [
            c for c in self._commitments.values()
            if c.is_active() and c.is_overdue()
        ]
    
    async def check_deadlines(self) -> list[ProtocolCommitment]:
        """
        Check for overdue commitments and mark them as failed.
        
        Returns:
            List of commitments that were marked as failed
        """
        failed = []
        
        for commitment in self.get_overdue_commitments():
            commitment.update_state(CommitmentState.FAILED)
            commitment.metadata["failure_reason"] = "deadline_exceeded"
            failed.append(commitment)
            
            # Notify creditor
            message = Message(
                type=MessageType.COMMITMENT_FAILED,
                sender_id=commitment.debtor_id,
                recipient_id=commitment.creditor_id,
                content={
                    "commitment_id": commitment.id,
                    "reason": "deadline_exceeded",
                },
                correlation_id=commitment.id,
            )
            
            await self.bus.send(message)
        
        return failed


# Global protocol instance
_protocol: CommitmentProtocol | None = None


def get_protocol() -> CommitmentProtocol:
    """Get the global commitment protocol."""
    global _protocol
    if _protocol is None:
        _protocol = CommitmentProtocol()
    return _protocol
