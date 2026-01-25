"""
Multi-agent coordination patterns.

This module implements common coordination patterns for multi-agent systems:
- Contract Net Protocol: Task allocation through bidding
- Auction Protocol: Resource allocation through auctions
- Delegation Pattern: Hierarchical task delegation
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Callable, Awaitable
from uuid import uuid4

from pydantic import BaseModel, Field, ConfigDict

from .messaging import Message, MessageType, MessageBus, get_message_bus
from .registry import AgentRegistry, AgentInfo, get_registry
from .protocol import CommitmentProtocol, ProtocolCommitment, get_protocol


class BidStatus(str, Enum):
    """Status of a bid."""
    
    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    WITHDRAWN = "withdrawn"


class Bid(BaseModel):
    """
    A bid in a Contract Net or Auction.
    
    Attributes:
        id: Unique bid identifier
        bidder_id: ID of the bidding agent
        task_id: ID of the task being bid on
        price: Bid price (lower is better for Contract Net)
        confidence: Bidder's confidence in completing the task
        estimated_time: Estimated completion time
        conditions: Any conditions on the bid
        status: Current status
        submitted_at: When the bid was submitted
        metadata: Additional metadata
    """
    
    model_config = ConfigDict(frozen=False)
    
    id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique bid identifier",
    )
    bidder_id: str = Field(..., description="ID of the bidding agent")
    task_id: str = Field(..., description="ID of the task")
    price: float = Field(
        default=1.0,
        ge=0.0,
        description="Bid price",
    )
    confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Confidence in completing the task",
    )
    estimated_time: float | None = Field(
        default=None,
        description="Estimated completion time in seconds",
    )
    conditions: list[str] = Field(
        default_factory=list,
        description="Conditions on the bid",
    )
    status: BidStatus = Field(
        default=BidStatus.PENDING,
        description="Current status",
    )
    submitted_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When submitted",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata",
    )


class Task(BaseModel):
    """
    A task to be allocated.
    
    Attributes:
        id: Unique task identifier
        requester_id: ID of the requesting agent
        description: Task description
        required_capability: Required capability to complete the task
        max_price: Maximum acceptable price
        deadline: Task deadline
        parameters: Task parameters
        status: Current status
        assigned_to: ID of the assigned agent
        created_at: When the task was created
        metadata: Additional metadata
    """
    
    model_config = ConfigDict(frozen=False)
    
    id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique task identifier",
    )
    requester_id: str = Field(..., description="ID of the requester")
    description: str = Field(..., description="Task description")
    required_capability: str | None = Field(
        default=None,
        description="Required capability",
    )
    max_price: float | None = Field(
        default=None,
        description="Maximum acceptable price",
    )
    deadline: datetime | None = Field(
        default=None,
        description="Task deadline",
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Task parameters",
    )
    status: str = Field(
        default="open",
        description="Current status",
    )
    assigned_to: str | None = Field(
        default=None,
        description="ID of assigned agent",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When created",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata",
    )


class ContractNetProtocol:
    """
    Contract Net Protocol for task allocation.
    
    The Contract Net Protocol is a negotiation protocol where:
    1. Manager announces a task
    2. Contractors submit bids
    3. Manager evaluates bids and awards contract
    4. Contractor executes task
    
    Example:
        >>> cnp = ContractNetProtocol()
        >>> task = Task(
        ...     requester_id="manager",
        ...     description="Search for information",
        ...     required_capability="web_search",
        ... )
        >>> winner = await cnp.announce_and_award(task)
    """
    
    def __init__(
        self,
        registry: AgentRegistry | None = None,
        bus: MessageBus | None = None,
        protocol: CommitmentProtocol | None = None,
    ) -> None:
        """
        Initialize the Contract Net Protocol.
        
        Args:
            registry: Agent registry
            bus: Message bus
            protocol: Commitment protocol
        """
        self.registry = registry or get_registry()
        self.bus = bus or get_message_bus()
        self.protocol = protocol or get_protocol()
        self._tasks: dict[str, Task] = {}
        self._bids: dict[str, list[Bid]] = {}  # task_id -> bids
    
    async def announce(
        self,
        task: Task,
        bid_timeout: float = 10.0,
    ) -> list[Bid]:
        """
        Announce a task and collect bids.
        
        Args:
            task: The task to announce
            bid_timeout: Time to wait for bids
            
        Returns:
            List of received bids
        """
        self._tasks[task.id] = task
        self._bids[task.id] = []
        
        # Find eligible agents
        if task.required_capability:
            eligible = self.registry.find_by_capability(task.required_capability)
        else:
            eligible = self.registry.get_available()
        
        if not eligible:
            return []
        
        # Send bid requests
        for agent in eligible:
            message = Message(
                type=MessageType.BID_REQUEST,
                sender_id=task.requester_id,
                recipient_id=agent.agent_id,
                content={
                    "task_id": task.id,
                    "description": task.description,
                    "required_capability": task.required_capability,
                    "max_price": task.max_price,
                    "deadline": task.deadline.isoformat() if task.deadline else None,
                    "parameters": task.parameters,
                },
                correlation_id=task.id,
            )
            await self.bus.send(message)
        
        # Wait for bids
        await asyncio.sleep(bid_timeout)
        
        return self._bids.get(task.id, [])
    
    def submit_bid(self, bid: Bid) -> bool:
        """
        Submit a bid for a task.
        
        Args:
            bid: The bid to submit
            
        Returns:
            True if bid was accepted
        """
        if bid.task_id not in self._tasks:
            return False
        
        task = self._tasks[bid.task_id]
        
        # Check max price
        if task.max_price is not None and bid.price > task.max_price:
            return False
        
        self._bids.setdefault(bid.task_id, []).append(bid)
        return True
    
    def evaluate_bids(
        self,
        task_id: str,
        strategy: str = "best_value",
    ) -> Bid | None:
        """
        Evaluate bids and select a winner.
        
        Strategies:
        - best_value: Best confidence/price ratio
        - lowest_price: Lowest price
        - highest_confidence: Highest confidence
        - fastest: Shortest estimated time
        
        Args:
            task_id: ID of the task
            strategy: Evaluation strategy
            
        Returns:
            Winning bid, or None if no valid bids
        """
        bids = self._bids.get(task_id, [])
        if not bids:
            return None
        
        # Filter pending bids
        pending = [b for b in bids if b.status == BidStatus.PENDING]
        if not pending:
            return None
        
        if strategy == "lowest_price":
            return min(pending, key=lambda b: b.price)
        elif strategy == "highest_confidence":
            return max(pending, key=lambda b: b.confidence)
        elif strategy == "fastest":
            timed = [b for b in pending if b.estimated_time is not None]
            if timed:
                return min(timed, key=lambda b: b.estimated_time)
            return pending[0]
        else:  # best_value
            return max(
                pending,
                key=lambda b: b.confidence / max(b.price, 0.01),
            )
    
    async def award(
        self,
        task_id: str,
        winner_bid: Bid,
    ) -> ProtocolCommitment | None:
        """
        Award a task to the winning bidder.
        
        Args:
            task_id: ID of the task
            winner_bid: The winning bid
            
        Returns:
            The created commitment, or None if failed
        """
        task = self._tasks.get(task_id)
        if not task:
            return None
        
        # Update bid statuses
        for bid in self._bids.get(task_id, []):
            if bid.id == winner_bid.id:
                bid.status = BidStatus.ACCEPTED
            else:
                bid.status = BidStatus.REJECTED
                
                # Notify rejected bidders
                message = Message(
                    type=MessageType.BID_REJECT,
                    sender_id=task.requester_id,
                    recipient_id=bid.bidder_id,
                    content={
                        "task_id": task_id,
                        "bid_id": bid.id,
                    },
                    correlation_id=task_id,
                )
                await self.bus.send(message)
        
        # Update task
        task.status = "assigned"
        task.assigned_to = winner_bid.bidder_id
        
        # Notify winner
        message = Message(
            type=MessageType.BID_ACCEPT,
            sender_id=task.requester_id,
            recipient_id=winner_bid.bidder_id,
            content={
                "task_id": task_id,
                "bid_id": winner_bid.id,
            },
            correlation_id=task_id,
        )
        await self.bus.send(message)
        
        # Create commitment
        commitment = await self.protocol.propose(
            debtor_id=winner_bid.bidder_id,
            creditor_id=task.requester_id,
            action=task.description,
            deadline=task.deadline,
            metadata={
                "task_id": task_id,
                "bid_id": winner_bid.id,
                "price": winner_bid.price,
            },
        )
        
        return commitment
    
    async def announce_and_award(
        self,
        task: Task,
        bid_timeout: float = 10.0,
        strategy: str = "best_value",
    ) -> tuple[Bid | None, ProtocolCommitment | None]:
        """
        Announce a task, collect bids, and award to winner.
        
        Args:
            task: The task to announce
            bid_timeout: Time to wait for bids
            strategy: Bid evaluation strategy
            
        Returns:
            Tuple of (winning bid, commitment) or (None, None)
        """
        bids = await self.announce(task, bid_timeout)
        if not bids:
            return None, None
        
        winner = self.evaluate_bids(task.id, strategy)
        if not winner:
            return None, None
        
        commitment = await self.award(task.id, winner)
        return winner, commitment


class AuctionType(str, Enum):
    """Types of auctions."""
    
    ENGLISH = "english"  # Ascending price
    DUTCH = "dutch"      # Descending price
    SEALED = "sealed"    # Sealed first-price
    VICKREY = "vickrey"  # Sealed second-price


class Auction(BaseModel):
    """
    An auction for resource allocation.
    
    Attributes:
        id: Unique auction identifier
        auctioneer_id: ID of the auctioneer
        item: Item being auctioned
        auction_type: Type of auction
        starting_price: Starting price
        reserve_price: Minimum acceptable price
        current_price: Current highest bid
        current_winner: Current highest bidder
        status: Auction status
        started_at: When the auction started
        ends_at: When the auction ends
        bids: List of bids
        metadata: Additional metadata
    """
    
    model_config = ConfigDict(frozen=False)
    
    id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique auction identifier",
    )
    auctioneer_id: str = Field(..., description="ID of the auctioneer")
    item: str = Field(..., description="Item being auctioned")
    auction_type: AuctionType = Field(
        default=AuctionType.ENGLISH,
        description="Type of auction",
    )
    starting_price: float = Field(
        default=0.0,
        ge=0.0,
        description="Starting price",
    )
    reserve_price: float | None = Field(
        default=None,
        description="Minimum acceptable price",
    )
    current_price: float = Field(
        default=0.0,
        ge=0.0,
        description="Current highest bid",
    )
    current_winner: str | None = Field(
        default=None,
        description="Current highest bidder",
    )
    status: str = Field(
        default="pending",
        description="Auction status",
    )
    started_at: datetime | None = Field(
        default=None,
        description="When started",
    )
    ends_at: datetime | None = Field(
        default=None,
        description="When ends",
    )
    bids: list[Bid] = Field(
        default_factory=list,
        description="List of bids",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata",
    )


class AuctionProtocol:
    """
    Auction Protocol for resource allocation.
    
    Supports multiple auction types for allocating resources
    among competing agents.
    
    Example:
        >>> ap = AuctionProtocol()
        >>> auction = Auction(
        ...     auctioneer_id="seller",
        ...     item="compute_resource",
        ...     starting_price=10.0,
        ... )
        >>> winner = await ap.run_auction(auction, duration=30.0)
    """
    
    def __init__(
        self,
        registry: AgentRegistry | None = None,
        bus: MessageBus | None = None,
    ) -> None:
        """
        Initialize the Auction Protocol.
        
        Args:
            registry: Agent registry
            bus: Message bus
        """
        self.registry = registry or get_registry()
        self.bus = bus or get_message_bus()
        self._auctions: dict[str, Auction] = {}
    
    async def start_auction(
        self,
        auction: Auction,
        duration: float = 60.0,
    ) -> None:
        """
        Start an auction.
        
        Args:
            auction: The auction to start
            duration: Auction duration in seconds
        """
        auction.status = "active"
        auction.started_at = datetime.now(timezone.utc)
        auction.ends_at = auction.started_at + timedelta(seconds=duration)
        auction.current_price = auction.starting_price
        
        self._auctions[auction.id] = auction
        
        # Announce auction
        message = Message(
            type=MessageType.BID_REQUEST,
            sender_id=auction.auctioneer_id,
            recipient_id=None,  # Broadcast
            content={
                "auction_id": auction.id,
                "item": auction.item,
                "auction_type": auction.auction_type.value,
                "starting_price": auction.starting_price,
                "ends_at": auction.ends_at.isoformat(),
            },
            correlation_id=auction.id,
        )
        await self.bus.send(message)
    
    def place_bid(
        self,
        auction_id: str,
        bidder_id: str,
        price: float,
    ) -> bool:
        """
        Place a bid in an auction.
        
        Args:
            auction_id: ID of the auction
            bidder_id: ID of the bidder
            price: Bid price
            
        Returns:
            True if bid was accepted
        """
        auction = self._auctions.get(auction_id)
        if not auction:
            return False
        
        if auction.status != "active":
            return False
        
        # Check if auction has ended
        if auction.ends_at and datetime.now(timezone.utc) > auction.ends_at:
            return False
        
        # Validate bid based on auction type
        if auction.auction_type == AuctionType.ENGLISH:
            # Must be higher than current price
            if price <= auction.current_price:
                return False
        elif auction.auction_type == AuctionType.DUTCH:
            # Accept first bid at or above current price
            if price < auction.current_price:
                return False
        
        # Record bid
        bid = Bid(
            bidder_id=bidder_id,
            task_id=auction_id,
            price=price,
        )
        auction.bids.append(bid)
        
        # Update current winner
        if price > auction.current_price:
            auction.current_price = price
            auction.current_winner = bidder_id
        
        # For Dutch auction, first valid bid wins
        if auction.auction_type == AuctionType.DUTCH:
            auction.status = "completed"
        
        return True
    
    async def end_auction(self, auction_id: str) -> Bid | None:
        """
        End an auction and determine the winner.
        
        Args:
            auction_id: ID of the auction
            
        Returns:
            Winning bid, or None if no winner
        """
        auction = self._auctions.get(auction_id)
        if not auction:
            return None
        
        auction.status = "completed"
        
        if not auction.bids:
            return None
        
        # Check reserve price
        if auction.reserve_price and auction.current_price < auction.reserve_price:
            return None
        
        # Determine winner based on auction type
        if auction.auction_type == AuctionType.VICKREY:
            # Winner pays second-highest price
            sorted_bids = sorted(auction.bids, key=lambda b: b.price, reverse=True)
            if len(sorted_bids) >= 2:
                winner = sorted_bids[0]
                winner.price = sorted_bids[1].price  # Pay second price
            else:
                winner = sorted_bids[0]
        else:
            # Highest bidder wins
            winner = max(auction.bids, key=lambda b: b.price)
        
        winner.status = BidStatus.ACCEPTED
        
        # Notify winner
        message = Message(
            type=MessageType.BID_ACCEPT,
            sender_id=auction.auctioneer_id,
            recipient_id=winner.bidder_id,
            content={
                "auction_id": auction_id,
                "item": auction.item,
                "price": winner.price,
            },
            correlation_id=auction_id,
        )
        await self.bus.send(message)
        
        # Notify losers
        for bid in auction.bids:
            if bid.id != winner.id:
                bid.status = BidStatus.REJECTED
                message = Message(
                    type=MessageType.BID_REJECT,
                    sender_id=auction.auctioneer_id,
                    recipient_id=bid.bidder_id,
                    content={
                        "auction_id": auction_id,
                    },
                    correlation_id=auction_id,
                )
                await self.bus.send(message)
        
        return winner
    
    async def run_auction(
        self,
        auction: Auction,
        duration: float = 60.0,
    ) -> Bid | None:
        """
        Run a complete auction.
        
        Args:
            auction: The auction to run
            duration: Auction duration in seconds
            
        Returns:
            Winning bid, or None if no winner
        """
        await self.start_auction(auction, duration)
        await asyncio.sleep(duration)
        return await self.end_auction(auction.id)
    
    async def run_dutch_auction(
        self,
        auction: Auction,
        decrement: float = 1.0,
        interval: float = 1.0,
        min_price: float = 0.0,
    ) -> Bid | None:
        """
        Run a Dutch (descending price) auction.
        
        Args:
            auction: The auction to run
            decrement: Price decrement per interval
            interval: Time between decrements
            min_price: Minimum price
            
        Returns:
            Winning bid, or None if no winner
        """
        auction.auction_type = AuctionType.DUTCH
        auction.status = "active"
        auction.started_at = datetime.now(timezone.utc)
        auction.current_price = auction.starting_price
        
        self._auctions[auction.id] = auction
        
        while auction.status == "active" and auction.current_price > min_price:
            # Announce current price
            message = Message(
                type=MessageType.BID_REQUEST,
                sender_id=auction.auctioneer_id,
                recipient_id=None,
                content={
                    "auction_id": auction.id,
                    "item": auction.item,
                    "current_price": auction.current_price,
                },
                correlation_id=auction.id,
            )
            await self.bus.send(message)
            
            await asyncio.sleep(interval)
            
            if auction.status == "active":
                auction.current_price = max(
                    auction.current_price - decrement,
                    min_price,
                )
        
        if auction.bids:
            return auction.bids[-1]  # Last bid wins
        return None


class DelegationPattern:
    """
    Hierarchical task delegation pattern.
    
    Allows agents to delegate tasks to subordinates while
    maintaining accountability through commitments.
    
    Example:
        >>> delegation = DelegationPattern()
        >>> await delegation.delegate(
        ...     delegator_id="manager",
        ...     delegate_id="worker",
        ...     task="process_data",
        ... )
    """
    
    def __init__(
        self,
        protocol: CommitmentProtocol | None = None,
        bus: MessageBus | None = None,
    ) -> None:
        """
        Initialize the delegation pattern.
        
        Args:
            protocol: Commitment protocol
            bus: Message bus
        """
        self.protocol = protocol or get_protocol()
        self.bus = bus or get_message_bus()
        self._delegations: dict[str, dict[str, Any]] = {}
    
    async def delegate(
        self,
        delegator_id: str,
        delegate_id: str,
        task: str,
        parameters: dict[str, Any] | None = None,
        deadline: datetime | None = None,
    ) -> ProtocolCommitment:
        """
        Delegate a task to another agent.
        
        Args:
            delegator_id: ID of the delegating agent
            delegate_id: ID of the delegate
            task: Task description
            parameters: Task parameters
            deadline: Task deadline
            
        Returns:
            The created commitment
        """
        commitment = await self.protocol.propose(
            debtor_id=delegate_id,
            creditor_id=delegator_id,
            action=task,
            deadline=deadline,
            metadata={
                "type": "delegation",
                "parameters": parameters or {},
            },
        )
        
        self._delegations[commitment.id] = {
            "delegator_id": delegator_id,
            "delegate_id": delegate_id,
            "task": task,
            "commitment": commitment,
        }
        
        # Send task assignment
        message = Message(
            type=MessageType.TASK_ASSIGN,
            sender_id=delegator_id,
            recipient_id=delegate_id,
            content={
                "commitment_id": commitment.id,
                "task": task,
                "parameters": parameters or {},
                "deadline": deadline.isoformat() if deadline else None,
            },
            correlation_id=commitment.id,
        )
        await self.bus.send(message)
        
        return commitment
    
    async def report_progress(
        self,
        commitment_id: str,
        delegate_id: str,
        progress: float,
        status: str = "",
    ) -> bool:
        """
        Report progress on a delegated task.
        
        Args:
            commitment_id: ID of the commitment
            delegate_id: ID of the delegate
            progress: Progress percentage (0-100)
            status: Status message
            
        Returns:
            True if reported successfully
        """
        delegation = self._delegations.get(commitment_id)
        if not delegation:
            return False
        
        if delegation["delegate_id"] != delegate_id:
            return False
        
        message = Message(
            type=MessageType.TASK_PROGRESS,
            sender_id=delegate_id,
            recipient_id=delegation["delegator_id"],
            content={
                "commitment_id": commitment_id,
                "progress": progress,
                "status": status,
            },
            correlation_id=commitment_id,
        )
        await self.bus.send(message)
        
        return True
    
    async def complete_delegation(
        self,
        commitment_id: str,
        delegate_id: str,
        result: Any = None,
    ) -> bool:
        """
        Complete a delegated task.
        
        Args:
            commitment_id: ID of the commitment
            delegate_id: ID of the delegate
            result: Task result
            
        Returns:
            True if completed successfully
        """
        delegation = self._delegations.get(commitment_id)
        if not delegation:
            return False
        
        if delegation["delegate_id"] != delegate_id:
            return False
        
        success = await self.protocol.complete(commitment_id, delegate_id, result)
        
        if success:
            message = Message(
                type=MessageType.TASK_COMPLETE,
                sender_id=delegate_id,
                recipient_id=delegation["delegator_id"],
                content={
                    "commitment_id": commitment_id,
                    "result": result,
                },
                correlation_id=commitment_id,
            )
            await self.bus.send(message)
        
        return success
