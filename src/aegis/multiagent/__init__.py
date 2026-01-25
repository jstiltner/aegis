"""
Multi-agent coordination module.

This module provides infrastructure for multi-agent systems including:
- Agent registry for tracking agents and capabilities
- Inter-agent messaging system
- Commitment protocol for coordination
- Coordination patterns (Contract Net, Auction, Delegation)
"""

from .registry import (
    AgentStatus,
    AgentCapability,
    AgentInfo,
    AgentRegistry,
    get_registry,
)
from .messaging import (
    MessageType,
    MessagePriority,
    Message,
    MessageHandler,
    MessageQueue,
    MessageBus,
    get_message_bus,
)
from .protocol import (
    CommitmentState,
    ProtocolCommitment,
    CommitmentProtocol,
    get_protocol,
)
from .patterns import (
    BidStatus,
    Bid,
    Task,
    ContractNetProtocol,
    AuctionType,
    Auction,
    AuctionProtocol,
    DelegationPattern,
)

__all__ = [
    # Registry
    "AgentStatus",
    "AgentCapability",
    "AgentInfo",
    "AgentRegistry",
    "get_registry",
    # Messaging
    "MessageType",
    "MessagePriority",
    "Message",
    "MessageHandler",
    "MessageQueue",
    "MessageBus",
    "get_message_bus",
    # Protocol
    "CommitmentState",
    "ProtocolCommitment",
    "CommitmentProtocol",
    "get_protocol",
    # Patterns
    "BidStatus",
    "Bid",
    "Task",
    "ContractNetProtocol",
    "AuctionType",
    "Auction",
    "AuctionProtocol",
    "DelegationPattern",
]
