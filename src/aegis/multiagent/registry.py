"""
Agent registry for multi-agent coordination.

This module provides the AgentRegistry for tracking active agents,
their capabilities, and their status.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, ConfigDict


class AgentStatus(str, Enum):
    """Status of an agent in the registry."""
    
    ONLINE = "online"
    OFFLINE = "offline"
    BUSY = "busy"
    PAUSED = "paused"


class AgentCapability(BaseModel):
    """
    A capability that an agent can provide.
    
    Attributes:
        name: Name of the capability
        description: Description of what the capability does
        parameters: Parameters the capability accepts
        cost: Cost to use this capability (for auctions)
        confidence: Agent's confidence in this capability
    """
    
    model_config = ConfigDict(frozen=True)
    
    name: str = Field(..., description="Name of the capability")
    description: str = Field(..., description="Description of the capability")
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters the capability accepts",
    )
    cost: float = Field(
        default=1.0,
        ge=0.0,
        description="Cost to use this capability",
    )
    confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Agent's confidence in this capability",
    )


class AgentInfo(BaseModel):
    """
    Information about a registered agent.
    
    Attributes:
        agent_id: Unique identifier for the agent
        name: Human-readable name
        description: Description of the agent
        capabilities: List of capabilities the agent provides
        status: Current status
        endpoint: Communication endpoint (if remote)
        metadata: Additional metadata
        registered_at: When the agent was registered
        last_seen: When the agent was last active
    """
    
    model_config = ConfigDict(frozen=False)
    
    agent_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier",
    )
    name: str = Field(..., description="Human-readable name")
    description: str = Field(
        default="",
        description="Description of the agent",
    )
    capabilities: list[AgentCapability] = Field(
        default_factory=list,
        description="List of capabilities",
    )
    status: AgentStatus = Field(
        default=AgentStatus.ONLINE,
        description="Current status",
    )
    endpoint: str | None = Field(
        default=None,
        description="Communication endpoint",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata",
    )
    registered_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When registered",
    )
    last_seen: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When last active",
    )
    
    def has_capability(self, capability_name: str) -> bool:
        """Check if agent has a specific capability."""
        return any(c.name == capability_name for c in self.capabilities)
    
    def get_capability(self, capability_name: str) -> AgentCapability | None:
        """Get a specific capability."""
        for cap in self.capabilities:
            if cap.name == capability_name:
                return cap
        return None
    
    def update_last_seen(self) -> None:
        """Update the last seen timestamp."""
        self.last_seen = datetime.now(timezone.utc)
    
    def is_available(self) -> bool:
        """Check if agent is available for tasks."""
        return self.status == AgentStatus.ONLINE


class AgentRegistry:
    """
    Registry for managing active agents.
    
    The AgentRegistry tracks all agents in the system, their capabilities,
    and their current status. It supports agent discovery and lookup.
    
    Example:
        >>> registry = AgentRegistry()
        >>> registry.register(AgentInfo(
        ...     name="SearchAgent",
        ...     capabilities=[AgentCapability(name="web_search", description="Search the web")],
        ... ))
        >>> agents = registry.find_by_capability("web_search")
    """
    
    def __init__(self) -> None:
        """Initialize the registry."""
        self._agents: dict[str, AgentInfo] = {}
    
    def register(self, agent: AgentInfo) -> str:
        """
        Register an agent.
        
        Args:
            agent: Agent information
            
        Returns:
            The agent ID
        """
        self._agents[agent.agent_id] = agent
        return agent.agent_id
    
    def unregister(self, agent_id: str) -> bool:
        """
        Unregister an agent.
        
        Args:
            agent_id: ID of the agent to unregister
            
        Returns:
            True if agent was found and removed
        """
        if agent_id in self._agents:
            del self._agents[agent_id]
            return True
        return False
    
    def get(self, agent_id: str) -> AgentInfo | None:
        """Get an agent by ID."""
        return self._agents.get(agent_id)
    
    def get_by_name(self, name: str) -> AgentInfo | None:
        """Get an agent by name."""
        for agent in self._agents.values():
            if agent.name == name:
                return agent
        return None
    
    def get_all(self) -> list[AgentInfo]:
        """Get all registered agents."""
        return list(self._agents.values())
    
    def get_online(self) -> list[AgentInfo]:
        """Get all online agents."""
        return [a for a in self._agents.values() if a.status == AgentStatus.ONLINE]
    
    def get_available(self) -> list[AgentInfo]:
        """Get all available agents."""
        return [a for a in self._agents.values() if a.is_available()]
    
    def find_by_capability(
        self,
        capability_name: str,
        only_available: bool = True,
    ) -> list[AgentInfo]:
        """
        Find agents with a specific capability.
        
        Args:
            capability_name: Name of the capability
            only_available: Only return available agents
            
        Returns:
            List of agents with the capability
        """
        agents = []
        for agent in self._agents.values():
            if agent.has_capability(capability_name):
                if not only_available or agent.is_available():
                    agents.append(agent)
        return agents
    
    def find_best_for_capability(
        self,
        capability_name: str,
    ) -> AgentInfo | None:
        """
        Find the best agent for a capability.
        
        Best is determined by highest confidence and lowest cost.
        
        Args:
            capability_name: Name of the capability
            
        Returns:
            Best agent, or None if no agents have the capability
        """
        candidates = self.find_by_capability(capability_name)
        if not candidates:
            return None
        
        def score(agent: AgentInfo) -> float:
            cap = agent.get_capability(capability_name)
            if cap is None:
                return 0.0
            # Higher confidence and lower cost = better score
            return cap.confidence / max(cap.cost, 0.01)
        
        return max(candidates, key=score)
    
    def update_status(self, agent_id: str, status: AgentStatus) -> bool:
        """
        Update an agent's status.
        
        Args:
            agent_id: ID of the agent
            status: New status
            
        Returns:
            True if agent was found and updated
        """
        agent = self._agents.get(agent_id)
        if agent:
            agent.status = status
            agent.update_last_seen()
            return True
        return False
    
    def heartbeat(self, agent_id: str) -> bool:
        """
        Record a heartbeat from an agent.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            True if agent was found
        """
        agent = self._agents.get(agent_id)
        if agent:
            agent.update_last_seen()
            return True
        return False
    
    def get_capabilities(self) -> dict[str, list[AgentInfo]]:
        """
        Get all capabilities and the agents that provide them.
        
        Returns:
            Dictionary mapping capability names to agents
        """
        capabilities: dict[str, list[AgentInfo]] = {}
        
        for agent in self._agents.values():
            for cap in agent.capabilities:
                if cap.name not in capabilities:
                    capabilities[cap.name] = []
                capabilities[cap.name].append(agent)
        
        return capabilities
    
    def __len__(self) -> int:
        return len(self._agents)
    
    def __iter__(self):
        return iter(self._agents.values())
    
    def __contains__(self, agent_id: str) -> bool:
        return agent_id in self._agents


# Global registry instance
_registry: AgentRegistry | None = None


def get_registry() -> AgentRegistry:
    """Get the global agent registry."""
    global _registry
    if _registry is None:
        _registry = AgentRegistry()
    return _registry
