"""
Unit tests for the tools package.

Tests:
- Tool models and validation
- Tool registry
- Policy engine
- Auth delegation
- Tool gateway
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone, timedelta
from typing import Any

import pytest

from aegis.tools.models import (
    Tool,
    ToolDefinition,
    ToolParameter,
    ParameterType,
    ToolResult,
    ToolResultStatus,
    ToolError,
    ToolInvocation,
)
from aegis.tools.registry import ToolRegistry
from aegis.tools.policy import (
    Policy,
    PolicyRule,
    PolicyEngine,
    PolicyDecision,
    PolicyDecisionType,
)
from aegis.tools.auth import (
    Credential,
    CredentialStore,
    AuthDelegator,
    ScopedToken,
)
from aegis.tools.gateway import (
    ToolGateway,
    ToolGatewayConfig,
    RateLimitConfig,
    CacheConfig,
    RateLimiter,
    ResultCache,
)


# =============================================================================
# Tool Models Tests
# =============================================================================

class TestToolParameter:
    """Tests for ToolParameter."""
    
    def test_create_required_parameter(self) -> None:
        """Test creating a required parameter."""
        param = ToolParameter(
            name="query",
            type=ParameterType.STRING,
            description="Search query",
            required=True,
        )
        
        assert param.name == "query"
        assert param.type == ParameterType.STRING
        assert param.required is True
    
    def test_create_optional_parameter_with_default(self) -> None:
        """Test creating an optional parameter with default."""
        param = ToolParameter(
            name="limit",
            type=ParameterType.INTEGER,
            description="Result limit",
            required=False,
            default=10,
        )
        
        assert param.name == "limit"
        assert param.required is False
        assert param.default == 10
    
    def test_parameter_with_enum(self) -> None:
        """Test parameter with enum values."""
        param = ToolParameter(
            name="format",
            type=ParameterType.STRING,
            description="Output format",
            required=True,
            enum=["json", "xml", "csv"],
        )
        
        assert param.enum == ["json", "xml", "csv"]
    
    def test_to_json_schema(self) -> None:
        """Test converting parameter to JSON schema."""
        param = ToolParameter(
            name="count",
            type=ParameterType.INTEGER,
            description="Number of items",
            required=True,
        )
        
        schema = param.to_json_schema()
        
        assert schema["type"] == "integer"
        assert schema["description"] == "Number of items"


class TestToolDefinition:
    """Tests for ToolDefinition."""
    
    def test_create_definition(self) -> None:
        """Test creating a tool definition."""
        definition = ToolDefinition(
            name="search",
            description="Search for items",
            parameters=(
                ToolParameter(
                    name="query",
                    type=ParameterType.STRING,
                    description="Search query",
                    required=True,
                ),
            ),
            category="search",
            tags=("web", "api"),
        )
        
        assert definition.name == "search"
        assert len(definition.parameters) == 1
        assert definition.category == "search"
    
    def test_to_openai_function(self) -> None:
        """Test converting definition to OpenAI function format."""
        definition = ToolDefinition(
            name="test",
            description="Test tool",
            parameters=(
                ToolParameter(
                    name="input",
                    type=ParameterType.STRING,
                    description="Input value",
                    required=True,
                ),
            ),
        )
        
        schema = definition.to_openai_function()
        
        assert schema["name"] == "test"
        assert schema["description"] == "Test tool"
        assert "input" in schema["parameters"]["properties"]
        assert "input" in schema["parameters"]["required"]


class TestTool:
    """Tests for Tool."""
    
    @pytest.fixture
    def simple_tool(self) -> Tool:
        """Create a simple test tool."""
        async def handler(args: dict[str, Any]) -> str:
            return f"Hello, {args['name']}!"
        
        return Tool.create(
            name="greet",
            description="Greet someone",
            handler=handler,
            parameters=[
                ToolParameter(
                    name="name",
                    type=ParameterType.STRING,
                    description="Name to greet",
                    required=True,
                ),
            ],
        )
    
    def test_create_tool(self, simple_tool: Tool) -> None:
        """Test creating a tool."""
        assert simple_tool.name == "greet"
        assert simple_tool.description == "Greet someone"
    
    @pytest.mark.asyncio
    async def test_invoke_tool(self, simple_tool: Tool) -> None:
        """Test invoking a tool."""
        result = await simple_tool.invoke({"name": "World"})
        assert result == "Hello, World!"
    
    def test_validate_arguments_valid(self, simple_tool: Tool) -> None:
        """Test validating valid arguments."""
        errors = simple_tool.validate_arguments({"name": "Test"})
        assert len(errors) == 0
    
    def test_validate_arguments_missing_required(self, simple_tool: Tool) -> None:
        """Test validating missing required argument."""
        errors = simple_tool.validate_arguments({})
        assert len(errors) == 1
        assert "name" in errors[0]
    
    def test_validate_arguments_wrong_type(self, simple_tool: Tool) -> None:
        """Test validating wrong argument type."""
        errors = simple_tool.validate_arguments({"name": 123})
        assert len(errors) == 1
        assert "type" in errors[0].lower()


class TestToolResult:
    """Tests for ToolResult."""
    
    def test_success_result(self) -> None:
        """Test creating a success result."""
        result = ToolResult.success(
            tool_name="test",
            output={"data": "value"},
        )
        
        assert result.is_success is True
        assert result.status == ToolResultStatus.SUCCESS
        assert result.output == {"data": "value"}
    
    def test_error_result(self) -> None:
        """Test creating an error result."""
        result = ToolResult(
            tool_name="test",
            status=ToolResultStatus.ERROR,
            error=ToolError(
                error_type="ValueError",
                message="Something went wrong",
            ),
        )
        
        assert result.is_success is False
        assert result.status == ToolResultStatus.ERROR
        assert result.error is not None
        assert "Something went wrong" in result.error.message
    
    def test_timeout_result(self) -> None:
        """Test creating a timeout result."""
        result = ToolResult.timeout(
            tool_name="test",
            timeout_seconds=30.0,
        )
        
        assert result.is_success is False
        assert result.status == ToolResultStatus.TIMEOUT
    
    def test_policy_denied_result(self) -> None:
        """Test creating a policy denied result."""
        result = ToolResult.policy_denied(
            tool_name="test",
            reason="Not allowed",
            policy_name="test_policy",
        )
        
        assert result.is_success is False
        assert result.status == ToolResultStatus.POLICY_DENIED


# =============================================================================
# Tool Registry Tests
# =============================================================================

class TestToolRegistry:
    """Tests for ToolRegistry."""
    
    @pytest.fixture
    def registry(self) -> ToolRegistry:
        """Create a test registry."""
        return ToolRegistry()
    
    @pytest.fixture
    def sample_tool(self) -> Tool:
        """Create a sample tool."""
        async def handler(args: dict[str, Any]) -> str:
            return "result"
        
        return Tool.create(
            name="sample",
            description="Sample tool",
            handler=handler,
            parameters=[],
            category="test",
            tags=["sample"],
        )
    
    def test_register_tool(self, registry: ToolRegistry, sample_tool: Tool) -> None:
        """Test registering a tool."""
        registry.register_tool(sample_tool)
        
        assert registry.get("sample") is not None
        assert len(registry.tools) == 1
    
    def test_register_duplicate_raises(self, registry: ToolRegistry, sample_tool: Tool) -> None:
        """Test that registering duplicate raises error."""
        registry.register_tool(sample_tool)
        
        with pytest.raises(ValueError):
            registry.register_tool(sample_tool)
    
    def test_unregister_tool(self, registry: ToolRegistry, sample_tool: Tool) -> None:
        """Test unregistering a tool."""
        registry.register_tool(sample_tool)
        result = registry.unregister_tool("sample")
        
        assert result is True
        assert registry.get("sample") is None
    
    def test_get_by_category(self, registry: ToolRegistry) -> None:
        """Test getting tools by category."""
        async def handler(args: dict[str, Any]) -> str:
            return "result"
        
        tool1 = Tool.create(
            name="tool1",
            description="Tool 1",
            handler=handler,
            parameters=[],
            category="cat1",
        )
        tool2 = Tool.create(
            name="tool2",
            description="Tool 2",
            handler=handler,
            parameters=[],
            category="cat2",
        )
        
        registry.register_tool(tool1)
        registry.register_tool(tool2)
        
        cat1_tools = registry.get_by_category("cat1")
        assert len(cat1_tools) == 1
        assert cat1_tools[0].name == "tool1"
    
    def test_get_by_tag(self, registry: ToolRegistry) -> None:
        """Test getting tools by tag."""
        async def handler(args: dict[str, Any]) -> str:
            return "result"
        
        tool1 = Tool.create(
            name="tool1",
            description="Tool 1",
            handler=handler,
            parameters=[],
            tags=["tag1", "common"],
        )
        tool2 = Tool.create(
            name="tool2",
            description="Tool 2",
            handler=handler,
            parameters=[],
            tags=["tag2", "common"],
        )
        
        registry.register_tool(tool1)
        registry.register_tool(tool2)
        
        common_tools = registry.get_by_tag("common")
        assert len(common_tools) == 2
        
        tag1_tools = registry.get_by_tag("tag1")
        assert len(tag1_tools) == 1
    
    def test_decorator_registration(self, registry: ToolRegistry) -> None:
        """Test decorator-based registration."""
        @registry.register(
            name="decorated",
            description="Decorated tool",
            parameters=[],
        )
        async def decorated_handler(args: dict[str, Any]) -> str:
            return "decorated result"
        
        tool = registry.get("decorated")
        assert tool is not None
        assert tool.name == "decorated"


# =============================================================================
# Policy Engine Tests
# =============================================================================

class TestPolicyRule:
    """Tests for PolicyRule."""
    
    def test_allow_rule(self) -> None:
        """Test creating an allow rule."""
        rule = PolicyRule(
            name="allow_search",
            action=PolicyDecisionType.ALLOW,
            tools=["search_*"],
        )
        
        assert rule.action == PolicyDecisionType.ALLOW
        assert rule.matches_tool("search_web")
        assert not rule.matches_tool("delete_file")
    
    def test_deny_rule(self) -> None:
        """Test creating a deny rule."""
        rule = PolicyRule(
            name="deny_delete",
            action=PolicyDecisionType.DENY,
            tools=["*_delete"],
        )
        
        assert rule.action == PolicyDecisionType.DENY
        assert rule.matches_tool("file_delete")
    
    def test_rule_with_context_conditions(self) -> None:
        """Test rule with context conditions."""
        rule = PolicyRule(
            name="admin_only",
            action=PolicyDecisionType.ALLOW,
            tools=["admin_*"],
            context_conditions={"role": {"values": ["admin"]}},
        )
        
        assert rule.matches_context({"role": "admin"})
        assert not rule.matches_context({"role": "user"})


class TestPolicy:
    """Tests for Policy."""
    
    def test_create_policy(self) -> None:
        """Test creating a policy."""
        policy = Policy(
            name="test_policy",
            description="Test policy",
            rules=(
                PolicyRule(
                    name="allow_read",
                    action=PolicyDecisionType.ALLOW,
                    tools=["read_*"],
                ),
            ),
        )
        
        assert policy.name == "test_policy"
        assert len(policy.rules) == 1


class TestPolicyEngine:
    """Tests for PolicyEngine."""
    
    @pytest.fixture
    def engine(self) -> PolicyEngine:
        """Create a test policy engine."""
        return PolicyEngine()
    
    def test_default_allow(self, engine: PolicyEngine) -> None:
        """Test default allow behavior when no policies configured."""
        decision = engine.evaluate(
            tool_name="any_tool",
            arguments={},
            context={"agent_id": "agent-1", "session_id": "session-1"},
        )
        
        assert decision.is_allowed
    
    def test_deny_rule(self, engine: PolicyEngine) -> None:
        """Test deny rule enforcement."""
        policy = Policy(
            name="deny_dangerous",
            rules=(
                PolicyRule(
                    name="deny_delete",
                    action=PolicyDecisionType.DENY,
                    tools=["delete_*"],
                ),
            ),
            default_action=PolicyDecisionType.ALLOW,
        )
        engine.add_policy(policy)
        
        decision = engine.evaluate(
            tool_name="delete_file",
            arguments={},
            context={"agent_id": "agent-1", "session_id": "session-1"},
        )
        
        assert decision.is_denied
    
    def test_require_approval_rule(self, engine: PolicyEngine) -> None:
        """Test require approval rule."""
        policy = Policy(
            name="approve_dangerous",
            rules=(
                PolicyRule(
                    name="approve_execute",
                    action=PolicyDecisionType.REQUIRE_APPROVAL,
                    tools=["execute_*"],
                ),
            ),
            default_action=PolicyDecisionType.ALLOW,
        )
        engine.add_policy(policy)
        
        decision = engine.evaluate(
            tool_name="execute_shell",
            arguments={},
            context={"agent_id": "agent-1", "session_id": "session-1"},
        )
        
        assert decision.requires_approval
    
    def test_policy_with_default_deny(self, engine: PolicyEngine) -> None:
        """Test policy with default deny."""
        policy = Policy(
            name="restrictive",
            rules=(
                PolicyRule(
                    name="allow_read",
                    action=PolicyDecisionType.ALLOW,
                    tools=["read_*"],
                ),
            ),
            default_action=PolicyDecisionType.DENY,
        )
        engine.add_policy(policy)
        
        # Should be allowed for read tools
        decision1 = engine.evaluate(
            tool_name="read_file",
            arguments={},
            context={"agent_id": "agent-1", "session_id": "session-1"},
        )
        assert decision1.is_allowed
        
        # Should be denied for other tools
        decision2 = engine.evaluate(
            tool_name="write_file",
            arguments={},
            context={"agent_id": "agent-1", "session_id": "session-1"},
        )
        assert decision2.is_denied


# =============================================================================
# Auth Delegation Tests
# =============================================================================

class TestCredentialStore:
    """Tests for CredentialStore."""
    
    @pytest.fixture
    def store(self) -> CredentialStore:
        """Create a test credential store."""
        return CredentialStore()
    
    def test_add_credential(self, store: CredentialStore) -> None:
        """Test adding a credential."""
        cred = Credential(
            name="api_key",
            credential_type="api_key",
            secret="secret123",
            scopes=("read", "write"),
        )
        
        store.add_credential(cred)
        
        retrieved = store.get_credential_by_name("api_key")
        assert retrieved is not None
        assert retrieved.name == "api_key"
    
    def test_get_nonexistent(self, store: CredentialStore) -> None:
        """Test getting nonexistent credential."""
        result = store.get_credential_by_name("nonexistent")
        assert result is None
    
    def test_remove_credential(self, store: CredentialStore) -> None:
        """Test removing a credential."""
        cred = Credential(
            name="to_delete",
            credential_type="api_key",
            secret="secret",
        )
        
        store.add_credential(cred)
        result = store.remove_credential(cred.credential_id)
        
        assert result is True
        assert store.get_credential_by_name("to_delete") is None


class TestAuthDelegator:
    """Tests for AuthDelegator."""
    
    @pytest.fixture
    def delegator(self) -> AuthDelegator:
        """Create a test auth delegator."""
        store = CredentialStore()
        return AuthDelegator(store)
    
    def test_configure_tool(self, delegator: AuthDelegator) -> None:
        """Test configuring tool auth requirement."""
        delegator.configure_tool(
            tool_name="api_call",
            credential_name="api_key",
            scopes=["read", "write"],
        )
        
        assert delegator.requires_auth("api_call")
        assert not delegator.requires_auth("other_tool")
    
    @pytest.mark.asyncio
    async def test_get_token_for_tool(self, delegator: AuthDelegator) -> None:
        """Test getting a scoped token for a tool."""
        # Store credential
        cred = Credential(
            name="api_key",
            credential_type="simple",
            secret="secret123",
            scopes=("read", "write"),
        )
        delegator.credential_store.add_credential(cred)
        
        # Configure tool auth
        delegator.configure_tool(
            tool_name="api_call",
            credential_name="api_key",
            scopes=["read"],
        )
        
        # Get token
        token = await delegator.get_token_for_tool(
            tool_name="api_call",
            agent_id="agent-1",
        )
        
        assert token is not None
        assert "read" in token.scopes


# =============================================================================
# Rate Limiter Tests
# =============================================================================

class TestRateLimiter:
    """Tests for RateLimiter."""
    
    @pytest.fixture
    def limiter(self) -> RateLimiter:
        """Create a test rate limiter."""
        config = RateLimitConfig(
            requests_per_minute=5,
            requests_per_hour=100,
        )
        return RateLimiter(config)
    
    def test_allow_under_limit(self, limiter: RateLimiter) -> None:
        """Test allowing requests under limit."""
        allowed, reason = limiter.check("agent-1", "tool-1")
        assert allowed is True
        assert reason is None
    
    def test_deny_over_limit(self, limiter: RateLimiter) -> None:
        """Test denying requests over limit."""
        # Record 5 requests
        for _ in range(5):
            limiter.record("agent-1", "tool-1")
        
        # 6th should be denied
        allowed, reason = limiter.check("agent-1", "tool-1")
        assert allowed is False
        assert reason is not None
    
    def test_separate_agents(self, limiter: RateLimiter) -> None:
        """Test that different agents have separate limits."""
        # Record 5 requests for agent-1
        for _ in range(5):
            limiter.record("agent-1", "tool-1")
        
        # agent-2 should still be allowed
        allowed, _ = limiter.check("agent-2", "tool-1")
        assert allowed is True


# =============================================================================
# Result Cache Tests
# =============================================================================

class TestResultCache:
    """Tests for ResultCache."""
    
    @pytest.fixture
    def cache(self) -> ResultCache:
        """Create a test cache."""
        config = CacheConfig(
            enabled=True,
            ttl_seconds=60,
            max_entries=10,
        )
        return ResultCache(config)
    
    def test_cache_miss(self, cache: ResultCache) -> None:
        """Test cache miss."""
        result = cache.get("tool", {"arg": "value"})
        assert result is None
    
    def test_cache_hit(self, cache: ResultCache) -> None:
        """Test cache hit."""
        tool_result = ToolResult.success("tool", {"data": "value"})
        
        cache.set("tool", {"arg": "value"}, tool_result)
        
        cached = cache.get("tool", {"arg": "value"})
        assert cached is not None
        assert cached.output == {"data": "value"}
    
    def test_cache_different_args(self, cache: ResultCache) -> None:
        """Test that different args have different cache entries."""
        result1 = ToolResult.success("tool", {"data": "1"})
        result2 = ToolResult.success("tool", {"data": "2"})
        
        cache.set("tool", {"arg": "1"}, result1)
        cache.set("tool", {"arg": "2"}, result2)
        
        cached1 = cache.get("tool", {"arg": "1"})
        cached2 = cache.get("tool", {"arg": "2"})
        
        assert cached1 is not None
        assert cached2 is not None
        assert cached1.output != cached2.output
    
    def test_invalidate_all(self, cache: ResultCache) -> None:
        """Test invalidating all cache entries."""
        result = ToolResult.success("tool", {"data": "value"})
        cache.set("tool", {"arg": "1"}, result)
        cache.set("tool", {"arg": "2"}, result)
        
        count = cache.invalidate()
        
        assert count == 2
        assert cache.get("tool", {"arg": "1"}) is None
    
    def test_invalidate_by_tool(self, cache: ResultCache) -> None:
        """Test invalidating cache entries by tool."""
        result = ToolResult.success("tool1", {"data": "value"})
        cache.set("tool1", {"arg": "1"}, result)
        cache.set("tool2", {"arg": "1"}, result)
        
        count = cache.invalidate("tool1")
        
        assert count == 1
        assert cache.get("tool1", {"arg": "1"}) is None
        assert cache.get("tool2", {"arg": "1"}) is not None


# =============================================================================
# Tool Gateway Tests
# =============================================================================

class TestToolGateway:
    """Tests for ToolGateway."""
    
    @pytest.fixture
    def registry(self) -> ToolRegistry:
        """Create a test registry with tools."""
        registry = ToolRegistry()
        
        async def echo_handler(args: dict[str, Any]) -> str:
            return f"Echo: {args.get('message', '')}"
        
        async def slow_handler(args: dict[str, Any]) -> str:
            await asyncio.sleep(args.get("delay", 0.1))
            return "Done"
        
        async def error_handler(args: dict[str, Any]) -> str:
            raise ValueError("Intentional error")
        
        registry.register_tool(Tool.create(
            name="echo",
            description="Echo a message",
            handler=echo_handler,
            parameters=[
                ToolParameter(
                    name="message",
                    type=ParameterType.STRING,
                    description="Message to echo",
                    required=True,
                ),
            ],
        ))
        
        registry.register_tool(Tool.create(
            name="slow",
            description="Slow tool",
            handler=slow_handler,
            parameters=[
                ToolParameter(
                    name="delay",
                    type=ParameterType.NUMBER,
                    description="Delay in seconds",
                    required=False,
                    default=0.1,
                ),
            ],
        ))
        
        registry.register_tool(Tool.create(
            name="error",
            description="Error tool",
            handler=error_handler,
            parameters=[],
        ))
        
        return registry
    
    @pytest.fixture
    def gateway(self, registry: ToolRegistry) -> ToolGateway:
        """Create a test gateway."""
        config = ToolGatewayConfig(
            rate_limit=RateLimitConfig(
                requests_per_minute=100,
                requests_per_hour=1000,
            ),
            cache=CacheConfig(
                enabled=True,
                ttl_seconds=60,
            ),
            default_timeout_seconds=5.0,
            max_retries=1,
        )
        return ToolGateway(registry, config=config)
    
    @pytest.mark.asyncio
    async def test_invoke_success(self, gateway: ToolGateway) -> None:
        """Test successful tool invocation."""
        result = await gateway.invoke(
            tool_name="echo",
            arguments={"message": "Hello"},
            agent_id="agent-1",
            session_id="session-1",
        )
        
        assert result.is_success
        assert result.output == "Echo: Hello"
    
    @pytest.mark.asyncio
    async def test_invoke_with_timeout(self, gateway: ToolGateway) -> None:
        """Test tool timeout."""
        result = await gateway.invoke(
            tool_name="slow",
            arguments={"delay": 10.0},  # 10 second delay
            agent_id="agent-1",
            session_id="session-1",
            timeout_seconds=0.1,  # 100ms timeout
        )
        
        assert not result.is_success
        assert result.status == ToolResultStatus.TIMEOUT
    
    @pytest.mark.asyncio
    async def test_caching(self, gateway: ToolGateway) -> None:
        """Test result caching."""
        # First invocation
        result1 = await gateway.invoke(
            tool_name="echo",
            arguments={"message": "Test"},
            agent_id="agent-1",
            session_id="session-1",
        )
        
        # Second invocation with same args should be cached
        result2 = await gateway.invoke(
            tool_name="echo",
            arguments={"message": "Test"},
            agent_id="agent-1",
            session_id="session-1",
        )
        
        assert result1.is_success
        assert result2.is_success
        assert result1.output == result2.output
    
    @pytest.mark.asyncio
    async def test_policy_enforcement(self, gateway: ToolGateway) -> None:
        """Test policy enforcement."""
        # Add deny policy
        policy = Policy(
            name="deny_echo",
            rules=(
                PolicyRule(
                    name="deny",
                    action=PolicyDecisionType.DENY,
                    tools=["echo"],
                ),
            ),
            default_action=PolicyDecisionType.ALLOW,
        )
        gateway.policy_engine.add_policy(policy)
        
        result = await gateway.invoke(
            tool_name="echo",
            arguments={"message": "Test"},
            agent_id="agent-1",
            session_id="session-1",
        )
        
        assert not result.is_success
        assert result.status == ToolResultStatus.POLICY_DENIED
    
    def test_get_invocation_stats(self, gateway: ToolGateway) -> None:
        """Test getting invocation statistics."""
        stats = gateway.get_invocation_stats()
        
        assert "total_invocations" in stats
        assert "by_tool" in stats
        assert "by_status" in stats
    
    @pytest.mark.asyncio
    async def test_hooks(self, gateway: ToolGateway) -> None:
        """Test pre and post invoke hooks."""
        pre_called = []
        post_called = []
        
        async def pre_hook(invocation: ToolInvocation) -> None:
            pre_called.append(invocation.tool_name)
        
        async def post_hook(invocation: ToolInvocation, result: ToolResult) -> None:
            post_called.append((invocation.tool_name, result.is_success))
        
        gateway.add_pre_invoke_hook(pre_hook)
        gateway.add_post_invoke_hook(post_hook)
        
        await gateway.invoke(
            tool_name="echo",
            arguments={"message": "Test"},
            agent_id="agent-1",
            session_id="session-1",
        )
        
        assert "echo" in pre_called
        assert ("echo", True) in post_called
