"""
Inter-agent messaging system.

This module provides the messaging infrastructure for multi-agent
communication, including message types, queues, and a message bus.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Awaitable
from uuid import uuid4

from pydantic import BaseModel, Field, ConfigDict


class MessageType(str, Enum):
    """Types of inter-agent messages."""
    
    # Commitment protocol messages
    COMMITMENT_REQUEST = "commitment_request"
    COMMITMENT_ACCEPT = "commitment_accept"
    COMMITMENT_REJECT = "commitment_reject"
    COMMITMENT_CANCEL = "commitment_cancel"
    COMMITMENT_COMPLETE = "commitment_complete"
    COMMITMENT_FAILED = "commitment_failed"
    
    # Task messages
    TASK_ASSIGN = "task_assign"
    TASK_PROGRESS = "task_progress"
    TASK_COMPLETE = "task_complete"
    TASK_FAILED = "task_failed"
    
    # Coordination messages
    BID_REQUEST = "bid_request"
    BID_SUBMIT = "bid_submit"
    BID_ACCEPT = "bid_accept"
    BID_REJECT = "bid_reject"
    
    # System messages
    HEARTBEAT = "heartbeat"
    STATUS_UPDATE = "status_update"
    CAPABILITY_UPDATE = "capability_update"
    
    # General
    QUERY = "query"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    ERROR = "error"


class MessagePriority(str, Enum):
    """Priority levels for messages."""
    
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class Message(BaseModel):
    """
    An inter-agent message.
    
    Attributes:
        id: Unique message identifier
        type: Type of message
        sender_id: ID of the sending agent
        recipient_id: ID of the recipient agent (None for broadcast)
        content: Message content
        priority: Message priority
        correlation_id: ID for correlating request/response pairs
        reply_to: Message ID this is a reply to
        timestamp: When the message was created
        expires_at: When the message expires
        metadata: Additional metadata
    """
    
    model_config = ConfigDict(frozen=True)
    
    id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique message identifier",
    )
    type: MessageType = Field(..., description="Type of message")
    sender_id: str = Field(..., description="ID of the sending agent")
    recipient_id: str | None = Field(
        default=None,
        description="ID of the recipient (None for broadcast)",
    )
    content: dict[str, Any] = Field(
        default_factory=dict,
        description="Message content",
    )
    priority: MessagePriority = Field(
        default=MessagePriority.NORMAL,
        description="Message priority",
    )
    correlation_id: str | None = Field(
        default=None,
        description="ID for correlating request/response pairs",
    )
    reply_to: str | None = Field(
        default=None,
        description="Message ID this is a reply to",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the message was created",
    )
    expires_at: datetime | None = Field(
        default=None,
        description="When the message expires",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata",
    )
    
    def is_expired(self) -> bool:
        """Check if the message has expired."""
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) > self.expires_at
    
    def is_broadcast(self) -> bool:
        """Check if this is a broadcast message."""
        return self.recipient_id is None
    
    def create_reply(
        self,
        sender_id: str,
        type: MessageType,
        content: dict[str, Any],
    ) -> "Message":
        """Create a reply to this message."""
        return Message(
            type=type,
            sender_id=sender_id,
            recipient_id=self.sender_id,
            content=content,
            correlation_id=self.correlation_id or self.id,
            reply_to=self.id,
        )


# Type for message handlers
MessageHandler = Callable[[Message], Awaitable[None]]


class MessageQueue:
    """
    A message queue for an agent.
    
    Each agent has its own queue for receiving messages.
    Messages are processed in priority order.
    
    Example:
        >>> queue = MessageQueue("agent-1")
        >>> await queue.put(message)
        >>> msg = await queue.get()
    """
    
    def __init__(self, agent_id: str, max_size: int = 1000) -> None:
        """
        Initialize the queue.
        
        Args:
            agent_id: ID of the agent this queue belongs to
            max_size: Maximum queue size
        """
        self.agent_id = agent_id
        self.max_size = max_size
        self._queues: dict[MessagePriority, asyncio.Queue] = {
            priority: asyncio.Queue(maxsize=max_size // 4)
            for priority in MessagePriority
        }
        self._handlers: dict[MessageType, list[MessageHandler]] = {}
    
    async def put(self, message: Message) -> bool:
        """
        Add a message to the queue.
        
        Args:
            message: The message to add
            
        Returns:
            True if added, False if queue is full
        """
        if message.is_expired():
            return False
        
        queue = self._queues[message.priority]
        try:
            queue.put_nowait(message)
            return True
        except asyncio.QueueFull:
            return False
    
    async def get(self, timeout: float | None = None) -> Message | None:
        """
        Get the next message from the queue.
        
        Messages are returned in priority order (urgent first).
        
        Args:
            timeout: Maximum time to wait (None for no timeout)
            
        Returns:
            The next message, or None if timeout
        """
        # Check queues in priority order
        for priority in [
            MessagePriority.URGENT,
            MessagePriority.HIGH,
            MessagePriority.NORMAL,
            MessagePriority.LOW,
        ]:
            queue = self._queues[priority]
            if not queue.empty():
                try:
                    return queue.get_nowait()
                except asyncio.QueueEmpty:
                    continue
        
        # No messages available, wait on normal queue
        if timeout is not None:
            try:
                return await asyncio.wait_for(
                    self._queues[MessagePriority.NORMAL].get(),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                return None
        else:
            return await self._queues[MessagePriority.NORMAL].get()
    
    def register_handler(
        self,
        message_type: MessageType,
        handler: MessageHandler,
    ) -> None:
        """Register a handler for a message type."""
        if message_type not in self._handlers:
            self._handlers[message_type] = []
        self._handlers[message_type].append(handler)
    
    def unregister_handler(
        self,
        message_type: MessageType,
        handler: MessageHandler,
    ) -> bool:
        """Unregister a handler."""
        if message_type in self._handlers:
            try:
                self._handlers[message_type].remove(handler)
                return True
            except ValueError:
                pass
        return False
    
    async def process_message(self, message: Message) -> None:
        """Process a message using registered handlers."""
        handlers = self._handlers.get(message.type, [])
        for handler in handlers:
            try:
                await handler(message)
            except Exception:
                # Log error but continue processing
                pass
    
    def size(self) -> int:
        """Get total number of messages in queue."""
        return sum(q.qsize() for q in self._queues.values())
    
    def is_empty(self) -> bool:
        """Check if queue is empty."""
        return all(q.empty() for q in self._queues.values())
    
    def clear(self) -> int:
        """Clear all messages from the queue."""
        count = 0
        for queue in self._queues.values():
            while not queue.empty():
                try:
                    queue.get_nowait()
                    count += 1
                except asyncio.QueueEmpty:
                    break
        return count


class MessageBus:
    """
    Central message bus for inter-agent communication.
    
    The MessageBus routes messages between agents and supports
    both direct messaging and broadcast.
    
    Example:
        >>> bus = MessageBus()
        >>> bus.register_agent("agent-1")
        >>> await bus.send(message)
    """
    
    def __init__(self) -> None:
        """Initialize the message bus."""
        self._queues: dict[str, MessageQueue] = {}
        self._global_handlers: list[MessageHandler] = []
        self._running = False
        self._processor_task: asyncio.Task | None = None
    
    def register_agent(self, agent_id: str) -> MessageQueue:
        """
        Register an agent with the bus.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            The agent's message queue
        """
        if agent_id not in self._queues:
            self._queues[agent_id] = MessageQueue(agent_id)
        return self._queues[agent_id]
    
    def unregister_agent(self, agent_id: str) -> bool:
        """
        Unregister an agent from the bus.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            True if agent was found and removed
        """
        if agent_id in self._queues:
            del self._queues[agent_id]
            return True
        return False
    
    def get_queue(self, agent_id: str) -> MessageQueue | None:
        """Get an agent's message queue."""
        return self._queues.get(agent_id)
    
    async def send(self, message: Message) -> bool:
        """
        Send a message.
        
        If recipient_id is None, broadcasts to all agents.
        
        Args:
            message: The message to send
            
        Returns:
            True if message was delivered to at least one recipient
        """
        # Call global handlers
        for handler in self._global_handlers:
            try:
                await handler(message)
            except Exception:
                pass
        
        if message.is_broadcast():
            # Broadcast to all agents except sender
            delivered = False
            for agent_id, queue in self._queues.items():
                if agent_id != message.sender_id:
                    if await queue.put(message):
                        delivered = True
            return delivered
        else:
            # Direct message
            queue = self._queues.get(message.recipient_id)
            if queue:
                return await queue.put(message)
            return False
    
    async def send_and_wait(
        self,
        message: Message,
        timeout: float = 30.0,
    ) -> Message | None:
        """
        Send a message and wait for a reply.
        
        Args:
            message: The message to send
            timeout: Maximum time to wait for reply
            
        Returns:
            The reply message, or None if timeout
        """
        # Ensure we have a correlation ID
        if message.correlation_id is None:
            message = Message(
                **{**message.model_dump(), "correlation_id": message.id}
            )
        
        # Send the message
        if not await self.send(message):
            return None
        
        # Wait for reply
        sender_queue = self._queues.get(message.sender_id)
        if not sender_queue:
            return None
        
        start_time = asyncio.get_event_loop().time()
        while True:
            remaining = timeout - (asyncio.get_event_loop().time() - start_time)
            if remaining <= 0:
                return None
            
            reply = await sender_queue.get(timeout=remaining)
            if reply is None:
                return None
            
            # Check if this is the reply we're waiting for
            if reply.correlation_id == message.correlation_id:
                return reply
            
            # Not our reply, put it back (simplified - in production use separate reply queue)
            await sender_queue.put(reply)
    
    def add_global_handler(self, handler: MessageHandler) -> None:
        """Add a handler that receives all messages."""
        self._global_handlers.append(handler)
    
    def remove_global_handler(self, handler: MessageHandler) -> bool:
        """Remove a global handler."""
        try:
            self._global_handlers.remove(handler)
            return True
        except ValueError:
            return False
    
    async def start_processing(self) -> None:
        """Start background message processing."""
        if self._running:
            return
        
        self._running = True
        self._processor_task = asyncio.create_task(self._process_loop())
    
    async def stop_processing(self) -> None:
        """Stop background message processing."""
        self._running = False
        if self._processor_task:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass
            self._processor_task = None
    
    async def _process_loop(self) -> None:
        """Background processing loop."""
        while self._running:
            for queue in self._queues.values():
                if not queue.is_empty():
                    message = await queue.get(timeout=0.1)
                    if message:
                        await queue.process_message(message)
            await asyncio.sleep(0.01)
    
    def get_stats(self) -> dict[str, Any]:
        """Get message bus statistics."""
        return {
            "agent_count": len(self._queues),
            "total_messages": sum(q.size() for q in self._queues.values()),
            "queues": {
                agent_id: queue.size()
                for agent_id, queue in self._queues.items()
            },
        }


# Global message bus instance
_bus: MessageBus | None = None


def get_message_bus() -> MessageBus:
    """Get the global message bus."""
    global _bus
    if _bus is None:
        _bus = MessageBus()
    return _bus
