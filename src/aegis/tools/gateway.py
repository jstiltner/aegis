"""
Tool Gateway for centralized tool execution.

This module provides:
- Centralized tool invocation with policy enforcement
- Rate limiting and quota management
- Caching of tool results
- Integration with auth delegation
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from typing import Any, Callable, Awaitable
from uuid import uuid4

from pydantic import BaseModel, Field, ConfigDict

from aegis.tools.models import (
    Tool,
    ToolDefinition,
    ToolResult,
    ToolResultStatus,
    ToolError,
    ToolInvocation,
)
from aegis.tools.registry import ToolRegistry
from aegis.tools.policy import (
    Policy,
    PolicyEngine,
    PolicyDecision,
    PolicyDecisionType,
    PolicyViolation,
)
from aegis.tools.auth import (
    CredentialStore,
    AuthDelegator,
    ScopedToken,
)


class RateLimitConfig(BaseModel):
    """Configuration for rate limiting."""
    
    model_config = ConfigDict(frozen=True)
    
    requests_per_minute: int = Field(
        default=60,
        ge=1,
        description="Maximum requests per minute",
    )
    requests_per_hour: int = Field(
        default=1000,
        ge=1,
        description="Maximum requests per hour",
    )
    burst_size: int = Field(
        default=10,
        ge=1,
        description="Maximum burst size",
    )


class CacheConfig(BaseModel):
    """Configuration for result caching."""
    
    model_config = ConfigDict(frozen=True)
    
    enabled: bool = Field(
        default=True,
        description="Whether caching is enabled",
    )
    ttl_seconds: int = Field(
        default=300,
        ge=1,
        description="Cache TTL in seconds",
    )
    max_entries: int = Field(
        default=1000,
        ge=1,
        description="Maximum cache entries",
    )


class ToolGatewayConfig(BaseModel):
    """Configuration for the tool gateway."""
    
    model_config = ConfigDict(frozen=True)
    
    # Rate limiting
    rate_limit: RateLimitConfig = Field(
        default_factory=RateLimitConfig,
        description="Rate limiting configuration",
    )
    
    # Caching
    cache: CacheConfig = Field(
        default_factory=CacheConfig,
        description="Caching configuration",
    )
    
    # Timeouts
    default_timeout_seconds: float = Field(
        default=30.0,
        ge=0.1,
        description="Default timeout for tool execution",
    )
    
    # Retries
    max_retries: int = Field(
        default=3,
        ge=0,
        description="Maximum retries for failed tools",
    )
    retry_delay_seconds: float = Field(
        default=1.0,
        ge=0,
        description="Delay between retries",
    )
    
    # Validation
    validate_arguments: bool = Field(
        default=True,
        description="Whether to validate arguments",
    )
    
    # Logging
    log_invocations: bool = Field(
        default=True,
        description="Whether to log invocations",
    )


class RateLimiter:
    """
    Token bucket rate limiter.
    
    Tracks request rates per agent/tool combination.
    """
    
    def __init__(self, config: RateLimitConfig) -> None:
        self.config = config
        self._minute_counts: dict[str, list[float]] = defaultdict(list)
        self._hour_counts: dict[str, list[float]] = defaultdict(list)
    
    def _cleanup_old_entries(self, key: str, now: float) -> None:
        """Remove entries older than the window."""
        minute_ago = now - 60
        hour_ago = now - 3600
        
        self._minute_counts[key] = [
            t for t in self._minute_counts[key] if t > minute_ago
        ]
        self._hour_counts[key] = [
            t for t in self._hour_counts[key] if t > hour_ago
        ]
    
    def check(self, agent_id: str, tool_name: str) -> tuple[bool, str | None]:
        """
        Check if a request is allowed.
        
        Returns:
            Tuple of (allowed, reason if denied)
        """
        key = f"{agent_id}:{tool_name}"
        now = time.time()
        
        self._cleanup_old_entries(key, now)
        
        # Check minute limit
        if len(self._minute_counts[key]) >= self.config.requests_per_minute:
            return False, f"Rate limit exceeded: {self.config.requests_per_minute}/min"
        
        # Check hour limit
        if len(self._hour_counts[key]) >= self.config.requests_per_hour:
            return False, f"Rate limit exceeded: {self.config.requests_per_hour}/hour"
        
        return True, None
    
    def record(self, agent_id: str, tool_name: str) -> None:
        """Record a request."""
        key = f"{agent_id}:{tool_name}"
        now = time.time()
        
        self._minute_counts[key].append(now)
        self._hour_counts[key].append(now)
    
    def get_remaining(self, agent_id: str, tool_name: str) -> dict[str, int]:
        """Get remaining requests in each window."""
        key = f"{agent_id}:{tool_name}"
        now = time.time()
        
        self._cleanup_old_entries(key, now)
        
        return {
            "per_minute": max(0, self.config.requests_per_minute - len(self._minute_counts[key])),
            "per_hour": max(0, self.config.requests_per_hour - len(self._hour_counts[key])),
        }


class ResultCache:
    """
    Cache for tool results.
    
    Caches results based on tool name and arguments.
    """
    
    def __init__(self, config: CacheConfig) -> None:
        self.config = config
        self._cache: dict[str, tuple[ToolResult, float]] = {}
    
    def _make_key(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Create a cache key from tool name and arguments."""
        import json
        args_str = json.dumps(arguments, sort_keys=True, default=str)
        return f"{tool_name}:{args_str}"
    
    def get(self, tool_name: str, arguments: dict[str, Any]) -> ToolResult | None:
        """Get a cached result."""
        if not self.config.enabled:
            return None
        
        key = self._make_key(tool_name, arguments)
        entry = self._cache.get(key)
        
        if entry is None:
            return None
        
        result, timestamp = entry
        
        # Check TTL
        if time.time() - timestamp > self.config.ttl_seconds:
            del self._cache[key]
            return None
        
        return result
    
    def set(self, tool_name: str, arguments: dict[str, Any], result: ToolResult) -> None:
        """Cache a result."""
        if not self.config.enabled:
            return
        
        # Only cache successful results
        if not result.is_success:
            return
        
        # Enforce max entries
        if len(self._cache) >= self.config.max_entries:
            self._evict_oldest()
        
        key = self._make_key(tool_name, arguments)
        self._cache[key] = (result, time.time())
    
    def _evict_oldest(self) -> None:
        """Evict the oldest cache entry."""
        if not self._cache:
            return
        
        oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k][1])
        del self._cache[oldest_key]
    
    def invalidate(self, tool_name: str | None = None) -> int:
        """
        Invalidate cache entries.
        
        Args:
            tool_name: If provided, only invalidate entries for this tool
            
        Returns:
            Number of entries invalidated
        """
        if tool_name is None:
            count = len(self._cache)
            self._cache.clear()
            return count
        
        keys_to_remove = [
            k for k in self._cache.keys()
            if k.startswith(f"{tool_name}:")
        ]
        
        for key in keys_to_remove:
            del self._cache[key]
        
        return len(keys_to_remove)
    
    def cleanup_expired(self) -> int:
        """Remove expired entries."""
        now = time.time()
        expired = [
            k for k, (_, timestamp) in self._cache.items()
            if now - timestamp > self.config.ttl_seconds
        ]
        
        for key in expired:
            del self._cache[key]
        
        return len(expired)


# Type for invocation hooks
InvocationHook = Callable[[ToolInvocation, ToolResult], Awaitable[None]]


class ToolGateway:
    """
    Centralized gateway for tool execution.
    
    The ToolGateway provides:
    - Policy enforcement before tool execution
    - Rate limiting
    - Result caching
    - Authentication delegation
    - Timeout handling
    - Retry logic
    - Invocation logging
    
    Example:
        >>> gateway = ToolGateway(registry, policy_engine)
        >>> 
        >>> # Invoke a tool
        >>> result = await gateway.invoke(
        ...     tool_name="search",
        ...     arguments={"query": "test"},
        ...     agent_id="agent-1",
        ...     session_id="session-1",
        ... )
        >>> 
        >>> if result.is_success:
        ...     print(result.output)
    """
    
    def __init__(
        self,
        registry: ToolRegistry,
        policy_engine: PolicyEngine | None = None,
        credential_store: CredentialStore | None = None,
        config: ToolGatewayConfig | None = None,
    ) -> None:
        """
        Initialize the gateway.
        
        Args:
            registry: Tool registry
            policy_engine: Policy engine for access control
            credential_store: Credential store for auth delegation
            config: Gateway configuration
        """
        self.registry = registry
        self.policy_engine = policy_engine or PolicyEngine()
        self.credential_store = credential_store or CredentialStore()
        self.config = config or ToolGatewayConfig()
        
        # Auth delegator
        self.auth_delegator = AuthDelegator(self.credential_store)
        
        # Rate limiter
        self._rate_limiter = RateLimiter(self.config.rate_limit)
        
        # Cache
        self._cache = ResultCache(self.config.cache)
        
        # Invocation tracking
        self._invocations: list[ToolInvocation] = []
        self._results: dict[str, ToolResult] = {}
        
        # Hooks
        self._pre_invoke_hooks: list[Callable[[ToolInvocation], Awaitable[None]]] = []
        self._post_invoke_hooks: list[InvocationHook] = []
        
        # Pending approvals
        self._pending_approvals: dict[str, ToolInvocation] = {}
    
    @property
    def invocations(self) -> list[ToolInvocation]:
        """Get all invocations."""
        return self._invocations.copy()
    
    def add_pre_invoke_hook(
        self,
        hook: Callable[[ToolInvocation], Awaitable[None]],
    ) -> None:
        """Add a hook called before tool invocation."""
        self._pre_invoke_hooks.append(hook)
    
    def add_post_invoke_hook(self, hook: InvocationHook) -> None:
        """Add a hook called after tool invocation."""
        self._post_invoke_hooks.append(hook)
    
    async def invoke(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        agent_id: str,
        session_id: str,
        commitment_id: str | None = None,
        timeout_seconds: float | None = None,
        use_cache: bool = True,
        context: dict[str, Any] | None = None,
    ) -> ToolResult:
        """
        Invoke a tool through the gateway.
        
        This is the main entry point for tool execution.
        
        Args:
            tool_name: Name of the tool to invoke
            arguments: Arguments to pass to the tool
            agent_id: ID of the agent making the invocation
            session_id: ID of the current session
            commitment_id: Optional GCL commitment ID
            timeout_seconds: Custom timeout (uses default if not provided)
            use_cache: Whether to use cached results
            context: Additional context for policy evaluation
            
        Returns:
            ToolResult with output or error
        """
        # Create invocation record
        invocation = ToolInvocation(
            tool_name=tool_name,
            arguments=arguments,
            agent_id=agent_id,
            session_id=session_id,
            commitment_id=commitment_id,
            timeout_seconds=timeout_seconds,
        )
        
        self._invocations.append(invocation)
        
        # Call pre-invoke hooks
        for hook in self._pre_invoke_hooks:
            try:
                await hook(invocation)
            except Exception:
                pass  # Don't let hook errors break invocation
        
        # Execute with error handling
        try:
            result = await self._execute_invocation(
                invocation,
                use_cache=use_cache,
                context=context,
            )
        except Exception as e:
            result = ToolResult.error(tool_name, e)
        
        # Store result
        self._results[invocation.invocation_id] = result
        
        # Call post-invoke hooks
        for hook in self._post_invoke_hooks:
            try:
                await hook(invocation, result)
            except Exception:
                pass
        
        return result
    
    async def _execute_invocation(
        self,
        invocation: ToolInvocation,
        use_cache: bool,
        context: dict[str, Any] | None,
    ) -> ToolResult:
        """Execute a tool invocation with all checks."""
        tool_name = invocation.tool_name
        arguments = invocation.arguments
        
        # Check if tool exists
        tool = self.registry.get(tool_name)
        if tool is None:
            return ToolResult.error(
                tool_name,
                ToolError(
                    error_type="ToolNotFound",
                    message=f"Tool '{tool_name}' not found",
                    recoverable=False,
                ),
            )
        
        # Validate arguments
        if self.config.validate_arguments:
            validation_errors = tool.validate_arguments(arguments)
            if validation_errors:
                return ToolResult.error(
                    tool_name,
                    ToolError(
                        error_type="ValidationError",
                        message="; ".join(validation_errors),
                        recoverable=False,
                    ),
                )
        
        # Check policy
        policy_context = {
            "agent_id": invocation.agent_id,
            "session_id": invocation.session_id,
            "commitment_id": invocation.commitment_id,
            **(context or {}),
        }
        
        decision = self.policy_engine.check_and_record(
            tool_name=tool_name,
            arguments=arguments,
            agent_id=invocation.agent_id,
            session_id=invocation.session_id,
            context=policy_context,
        )
        
        if decision.is_denied:
            return ToolResult.policy_denied(
                tool_name,
                reason=decision.reason,
                policy_name=decision.policy_name,
            )
        
        if decision.requires_approval:
            # Store for approval
            self._pending_approvals[invocation.invocation_id] = invocation
            return ToolResult(
                tool_name=tool_name,
                status=ToolResultStatus.CANCELLED,
                error=ToolError(
                    error_type="ApprovalRequired",
                    message=f"Requires {decision.approval_type} approval",
                    details={"approval_type": decision.approval_type},
                    recoverable=True,
                ),
            )
        
        # Check rate limit
        allowed, reason = self._rate_limiter.check(invocation.agent_id, tool_name)
        if not allowed:
            return ToolResult.error(
                tool_name,
                ToolError(
                    error_type="RateLimitExceeded",
                    message=reason or "Rate limit exceeded",
                    recoverable=True,
                ),
            )
        
        # Check cache
        if use_cache:
            cached = self._cache.get(tool_name, arguments)
            if cached is not None:
                return cached
        
        # Get auth token if needed
        token: ScopedToken | None = None
        if self.auth_delegator.requires_auth(tool_name):
            try:
                token = await self.auth_delegator.get_token_for_tool(
                    tool_name=tool_name,
                    agent_id=invocation.agent_id,
                )
            except Exception as e:
                return ToolResult.error(
                    tool_name,
                    ToolError(
                        error_type="AuthenticationError",
                        message=str(e),
                        recoverable=False,
                    ),
                )
        
        # Execute with timeout and retries
        timeout = invocation.timeout_seconds or self.config.default_timeout_seconds
        result = await self._execute_with_retries(
            tool=tool,
            arguments=arguments,
            timeout=timeout,
            token=token,
        )
        
        # Record rate limit
        self._rate_limiter.record(invocation.agent_id, tool_name)
        
        # Cache result
        if use_cache and result.is_success:
            self._cache.set(tool_name, arguments, result)
        
        return result
    
    async def _execute_with_retries(
        self,
        tool: Tool,
        arguments: dict[str, Any],
        timeout: float,
        token: ScopedToken | None,
    ) -> ToolResult:
        """Execute a tool with retry logic."""
        last_error: Exception | None = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                result = await self._execute_once(tool, arguments, timeout, token)
                
                # Don't retry on success or non-recoverable errors
                if result.is_success:
                    return result
                if result.error and not result.error.recoverable:
                    return result
                
                last_error = Exception(result.error.message if result.error else "Unknown error")
                
            except asyncio.TimeoutError:
                return ToolResult.timeout(tool.name, timeout)
            except Exception as e:
                last_error = e
            
            # Wait before retry
            if attempt < self.config.max_retries:
                await asyncio.sleep(self.config.retry_delay_seconds)
        
        # All retries failed
        return ToolResult.error(tool.name, last_error or Exception("Unknown error"))
    
    async def _execute_once(
        self,
        tool: Tool,
        arguments: dict[str, Any],
        timeout: float,
        token: ScopedToken | None,
    ) -> ToolResult:
        """Execute a tool once with timeout."""
        started_at = datetime.now(timezone.utc)
        
        # Add token to arguments if present
        exec_args = arguments.copy()
        if token is not None:
            exec_args["_auth_token"] = token.token.get_secret_value()
        
        try:
            # Execute with timeout
            output = await asyncio.wait_for(
                tool.invoke(exec_args),
                timeout=timeout,
            )
            
            completed_at = datetime.now(timezone.utc)
            
            return ToolResult.success(
                tool_name=tool.name,
                output=output,
                started_at=started_at,
                completed_at=completed_at,
            )
            
        except asyncio.TimeoutError:
            raise
        except Exception as e:
            completed_at = datetime.now(timezone.utc)
            
            return ToolResult.error(
                tool_name=tool.name,
                error=e,
                started_at=started_at,
                completed_at=completed_at,
            )
    
    async def approve_invocation(
        self,
        invocation_id: str,
        approver: str,
    ) -> ToolResult | None:
        """
        Approve a pending invocation and execute it.
        
        Args:
            invocation_id: ID of the invocation to approve
            approver: ID of the approver
            
        Returns:
            ToolResult if invocation was found and executed
        """
        invocation = self._pending_approvals.pop(invocation_id, None)
        if invocation is None:
            return None
        
        # Execute the approved invocation
        return await self._execute_invocation(
            invocation,
            use_cache=True,
            context={"approved_by": approver},
        )
    
    async def reject_invocation(
        self,
        invocation_id: str,
        reason: str,
    ) -> bool:
        """
        Reject a pending invocation.
        
        Args:
            invocation_id: ID of the invocation to reject
            reason: Reason for rejection
            
        Returns:
            True if invocation was found and rejected
        """
        invocation = self._pending_approvals.pop(invocation_id, None)
        if invocation is None:
            return False
        
        # Record rejection
        result = ToolResult(
            tool_name=invocation.tool_name,
            status=ToolResultStatus.CANCELLED,
            error=ToolError(
                error_type="Rejected",
                message=reason,
                recoverable=False,
            ),
        )
        self._results[invocation_id] = result
        
        return True
    
    def get_pending_approvals(self) -> list[ToolInvocation]:
        """Get all pending approval requests."""
        return list(self._pending_approvals.values())
    
    def get_result(self, invocation_id: str) -> ToolResult | None:
        """Get the result of an invocation."""
        return self._results.get(invocation_id)
    
    def get_tool_definitions(self) -> list[ToolDefinition]:
        """Get all available tool definitions."""
        return self.registry.get_definitions()
    
    def get_rate_limit_status(
        self,
        agent_id: str,
        tool_name: str,
    ) -> dict[str, int]:
        """Get rate limit status for an agent/tool."""
        return self._rate_limiter.get_remaining(agent_id, tool_name)
    
    def invalidate_cache(self, tool_name: str | None = None) -> int:
        """Invalidate cache entries."""
        return self._cache.invalidate(tool_name)
    
    def get_invocation_stats(self) -> dict[str, Any]:
        """Get invocation statistics."""
        total = len(self._invocations)
        by_tool: dict[str, int] = defaultdict(int)
        by_status: dict[str, int] = defaultdict(int)
        
        for inv in self._invocations:
            by_tool[inv.tool_name] += 1
            result = self._results.get(inv.invocation_id)
            if result:
                by_status[result.status.value] += 1
        
        return {
            "total_invocations": total,
            "by_tool": dict(by_tool),
            "by_status": dict(by_status),
            "pending_approvals": len(self._pending_approvals),
            "policy_violations": len(self.policy_engine.violations),
        }
