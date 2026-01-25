"""
Unit tests for the LLM integration module.

Tests cover:
- LLM models (messages, responses, conversations)
- Streaming support
- Provider implementations
- LLM client functionality
"""

import pytest
from datetime import datetime, timezone
from uuid import uuid4

from aegis.llm import (
    # Models
    LLMMessage,
    LLMResponse,
    LLMConfig,
    LLMUsage,
    Conversation,
    MessageRole,
    StopReason,
    ContentBlock,
    TextContent,
    ToolUseContent,
    ToolResultContent,
    # Streaming
    StreamEvent,
    StreamEventType,
    StreamingResponse,
    StreamBuffer,
    # Providers
    LLMProvider,
    MockProvider,
    # Client
    LLMClient,
    LLMClientConfig,
    ToolExecutor,
    create_client,
)


# =============================================================================
# LLM Models Tests
# =============================================================================

class TestLLMMessage:
    """Tests for LLMMessage."""
    
    def test_create_user_message(self):
        """Test creating a user message."""
        msg = LLMMessage.user("Hello, world!")
        
        assert msg.role == MessageRole.USER
        assert msg.content == "Hello, world!"
    
    def test_create_assistant_message(self):
        """Test creating an assistant message."""
        msg = LLMMessage.assistant("Hi there!")
        
        assert msg.role == MessageRole.ASSISTANT
        assert msg.content == "Hi there!"
    
    def test_create_system_message(self):
        """Test creating a system message."""
        msg = LLMMessage.system("You are a helpful assistant.")
        
        assert msg.role == MessageRole.SYSTEM
        assert msg.content == "You are a helpful assistant."
    
    def test_create_tool_result_message(self):
        """Test creating a tool result message."""
        msg = LLMMessage.tool_result("tool-123", "Result data")
        
        assert msg.role == MessageRole.USER
        assert len(msg.content) == 1
        assert isinstance(msg.content[0], ToolResultContent)
        assert msg.content[0].tool_use_id == "tool-123"
        assert msg.content[0].content == "Result data"
    
    def test_message_with_content_blocks(self):
        """Test message with content blocks."""
        blocks = [
            TextContent(text="Here's the result:"),
            ToolUseContent(id="tool-1", name="search", input={"query": "test"}),
        ]
        msg = LLMMessage(role=MessageRole.ASSISTANT, content=blocks)
        
        assert len(msg.content) == 2
        assert isinstance(msg.content[0], TextContent)
        assert isinstance(msg.content[1], ToolUseContent)
    
    def test_get_text_from_string_content(self):
        """Test getting text from string content."""
        msg = LLMMessage.user("Hello")
        assert msg.get_text() == "Hello"
    
    def test_get_text_from_content_blocks(self):
        """Test getting text from content blocks."""
        blocks = [
            TextContent(text="Part 1. "),
            TextContent(text="Part 2."),
        ]
        msg = LLMMessage(role=MessageRole.ASSISTANT, content=blocks)
        
        assert msg.get_text() == "Part 1. Part 2."
    
    def test_to_dict(self):
        """Test converting message to dict."""
        msg = LLMMessage.user("Hello")
        d = msg.to_dict()
        
        assert d["role"] == "user"
        assert d["content"] == "Hello"
    
    def test_to_dict_with_blocks(self):
        """Test converting message with blocks to dict."""
        blocks = [TextContent(text="Hello")]
        msg = LLMMessage(role=MessageRole.ASSISTANT, content=blocks)
        d = msg.to_dict()
        
        assert d["role"] == "assistant"
        assert isinstance(d["content"], list)
        assert d["content"][0]["type"] == "text"


class TestLLMResponse:
    """Tests for LLMResponse."""
    
    def test_create_response(self):
        """Test creating a response."""
        response = LLMResponse(
            model="claude-sonnet-4-20250514",
            content=[TextContent(text="Hello!")],
            usage=LLMUsage(input_tokens=10, output_tokens=5),
        )
        
        assert response.model == "claude-sonnet-4-20250514"
        assert response.stop_reason == StopReason.END_TURN
        assert response.usage.input_tokens == 10
        assert response.usage.output_tokens == 5
    
    def test_get_text(self):
        """Test getting text from response."""
        response = LLMResponse(
            model="test",
            content=[
                TextContent(text="Part 1. "),
                TextContent(text="Part 2."),
            ],
        )
        
        assert response.get_text() == "Part 1. Part 2."
    
    def test_get_tool_uses(self):
        """Test getting tool uses from response."""
        response = LLMResponse(
            model="test",
            content=[
                TextContent(text="Let me search for that."),
                ToolUseContent(id="tool-1", name="search", input={"q": "test"}),
                ToolUseContent(id="tool-2", name="read", input={"file": "a.txt"}),
            ],
            stop_reason=StopReason.TOOL_USE,
        )
        
        tool_uses = response.get_tool_uses()
        assert len(tool_uses) == 2
        assert tool_uses[0].name == "search"
        assert tool_uses[1].name == "read"
    
    def test_has_tool_use(self):
        """Test checking for tool use."""
        response_with_tools = LLMResponse(
            model="test",
            content=[ToolUseContent(id="1", name="test", input={})],
            stop_reason=StopReason.TOOL_USE,
        )
        response_without_tools = LLMResponse(
            model="test",
            content=[TextContent(text="Hello")],
        )
        
        assert response_with_tools.has_tool_use() is True
        assert response_without_tools.has_tool_use() is False
    
    def test_total_tokens(self):
        """Test total token calculation."""
        response = LLMResponse(
            model="test",
            content=[TextContent(text="Hi")],
            usage=LLMUsage(input_tokens=100, output_tokens=50),
        )
        
        assert response.usage.total_tokens == 150


class TestLLMConfig:
    """Tests for LLMConfig."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = LLMConfig()
        
        assert config.model == "claude-sonnet-4-20250514"
        assert config.max_tokens == 4096
        assert config.temperature == 1.0
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = LLMConfig(
            model="claude-3-opus-20240229",
            max_tokens=8192,
            temperature=0.7,
            system="You are helpful.",
        )
        
        assert config.model == "claude-3-opus-20240229"
        assert config.max_tokens == 8192
        assert config.temperature == 0.7
        assert config.system == "You are helpful."
    
    def test_config_with_tools(self):
        """Test configuration with tools."""
        tools = [
            {
                "name": "search",
                "description": "Search the web",
                "input_schema": {"type": "object"},
            }
        ]
        config = LLMConfig(tools=tools)
        
        assert config.tools == tools


class TestConversation:
    """Tests for Conversation."""
    
    def test_create_conversation(self):
        """Test creating a conversation."""
        conv = Conversation(system_prompt="You are helpful.")
        
        assert conv.system_prompt == "You are helpful."
        assert len(conv.messages) == 0
    
    def test_add_messages(self):
        """Test adding messages."""
        conv = Conversation()
        conv.add_message(LLMMessage.user("Hello"))
        conv.add_message(LLMMessage.assistant("Hi!"))
        
        assert len(conv.messages) == 2
        assert conv.messages[0].role == MessageRole.USER
        assert conv.messages[1].role == MessageRole.ASSISTANT
    
    def test_get_last_message(self):
        """Test getting last message."""
        conv = Conversation()
        conv.add_message(LLMMessage.user("First"))
        conv.add_message(LLMMessage.assistant("Second"))
        
        last = conv.get_last_message()
        assert last is not None
        assert last.content == "Second"
    
    def test_get_last_message_empty(self):
        """Test getting last message from empty conversation."""
        conv = Conversation()
        assert conv.get_last_message() is None
    
    def test_clear_conversation(self):
        """Test clearing conversation."""
        conv = Conversation()
        conv.add_message(LLMMessage.user("Hello"))
        conv.clear()
        
        assert len(conv.messages) == 0
    
    def test_conversation_token_count(self):
        """Test token count estimation."""
        conv = Conversation()
        conv.add_message(LLMMessage.user("Hello world"))
        
        # Rough estimate: ~4 chars per token
        assert conv.estimated_tokens > 0


# =============================================================================
# Content Block Tests
# =============================================================================

class TestContentBlocks:
    """Tests for content blocks."""
    
    def test_text_content(self):
        """Test TextContent."""
        content = TextContent(text="Hello, world!")
        
        assert content.type == "text"
        assert content.text == "Hello, world!"
    
    def test_tool_use_content(self):
        """Test ToolUseContent."""
        content = ToolUseContent(
            id="tool-123",
            name="search",
            input={"query": "test"},
        )
        
        assert content.type == "tool_use"
        assert content.id == "tool-123"
        assert content.name == "search"
        assert content.input == {"query": "test"}
    
    def test_tool_result_content(self):
        """Test ToolResultContent."""
        content = ToolResultContent(
            tool_use_id="tool-123",
            content="Search results...",
        )
        
        assert content.type == "tool_result"
        assert content.tool_use_id == "tool-123"
        assert content.content == "Search results..."
        assert content.is_error is False
    
    def test_tool_result_error(self):
        """Test ToolResultContent with error."""
        content = ToolResultContent(
            tool_use_id="tool-123",
            content="Error: Not found",
            is_error=True,
        )
        
        assert content.is_error is True


# =============================================================================
# Streaming Tests
# =============================================================================

class TestStreamEvent:
    """Tests for StreamEvent."""
    
    def test_create_event(self):
        """Test creating a stream event."""
        event = StreamEvent(
            type=StreamEventType.TEXT_DELTA,
            data={"text": "Hello"},
        )
        
        assert event.type == StreamEventType.TEXT_DELTA
        assert event.text == "Hello"
    
    def test_text_property(self):
        """Test text property."""
        event = StreamEvent(
            type=StreamEventType.TEXT_DELTA,
            data={"text": "World"},
        )
        
        assert event.text == "World"
    
    def test_text_property_non_text_event(self):
        """Test text property on non-text event."""
        event = StreamEvent(type=StreamEventType.MESSAGE_START)
        assert event.text is None
    
    def test_tool_name_property(self):
        """Test tool_name property."""
        event = StreamEvent(
            type=StreamEventType.TOOL_USE_START,
            data={"name": "search"},
        )
        
        assert event.tool_name == "search"
    
    def test_error_message_property(self):
        """Test error_message property."""
        event = StreamEvent(
            type=StreamEventType.ERROR,
            data={"message": "Something went wrong"},
        )
        
        assert event.error_message == "Something went wrong"


class TestStreamBuffer:
    """Tests for StreamBuffer."""
    
    def test_add_events(self):
        """Test adding events to buffer."""
        buffer = StreamBuffer()
        
        buffer.add_event(StreamEvent(type=StreamEventType.MESSAGE_START))
        buffer.add_event(StreamEvent(
            type=StreamEventType.TEXT_DELTA,
            data={"text": "Hello"},
        ))
        
        assert len(buffer.events) == 2
        assert buffer.text == "Hello"
    
    def test_accumulate_text(self):
        """Test text accumulation."""
        buffer = StreamBuffer()
        
        buffer.add_event(StreamEvent(
            type=StreamEventType.TEXT_DELTA,
            data={"text": "Hello "},
        ))
        buffer.add_event(StreamEvent(
            type=StreamEventType.TEXT_DELTA,
            data={"text": "world!"},
        ))
        
        assert buffer.text == "Hello world!"
    
    def test_clear_buffer(self):
        """Test clearing buffer."""
        buffer = StreamBuffer()
        buffer.add_event(StreamEvent(
            type=StreamEventType.TEXT_DELTA,
            data={"text": "Hello"},
        ))
        
        buffer.clear()
        
        assert len(buffer.events) == 0
        assert buffer.text == ""
    
    @pytest.mark.asyncio
    async def test_replay_events(self):
        """Test replaying events."""
        buffer = StreamBuffer()
        buffer.add_event(StreamEvent(type=StreamEventType.MESSAGE_START))
        buffer.add_event(StreamEvent(
            type=StreamEventType.TEXT_DELTA,
            data={"text": "Hello"},
        ))
        
        replayed = []
        async for event in buffer.replay():
            replayed.append(event)
        
        assert len(replayed) == 2


@pytest.mark.asyncio
class TestStreamingResponse:
    """Tests for StreamingResponse."""
    
    async def test_iterate_events(self):
        """Test iterating over stream events."""
        async def event_gen():
            yield StreamEvent(type=StreamEventType.MESSAGE_START)
            yield StreamEvent(
                type=StreamEventType.CONTENT_BLOCK_START,
                data={"content_block": {"type": "text"}},
            )
            yield StreamEvent(
                type=StreamEventType.TEXT_DELTA,
                data={"text": "Hello"},
            )
            yield StreamEvent(type=StreamEventType.CONTENT_BLOCK_STOP)
            yield StreamEvent(type=StreamEventType.MESSAGE_STOP)
        
        stream = StreamingResponse(
            event_stream=event_gen(),
            model="test-model",
        )
        
        events = []
        async for event in stream:
            events.append(event)
        
        assert len(events) == 5
        assert stream.is_finished
    
    async def test_collect_text(self):
        """Test collecting text from stream."""
        async def event_gen():
            yield StreamEvent(type=StreamEventType.MESSAGE_START)
            yield StreamEvent(
                type=StreamEventType.CONTENT_BLOCK_START,
                data={"content_block": {"type": "text"}},
            )
            yield StreamEvent(
                type=StreamEventType.TEXT_DELTA,
                data={"text": "Hello "},
            )
            yield StreamEvent(
                type=StreamEventType.TEXT_DELTA,
                data={"text": "world!"},
            )
            yield StreamEvent(type=StreamEventType.CONTENT_BLOCK_STOP)
            yield StreamEvent(type=StreamEventType.MESSAGE_STOP)
        
        stream = StreamingResponse(
            event_stream=event_gen(),
            model="test-model",
        )
        
        text = await stream.collect_text()
        assert text == "Hello world!"
    
    async def test_get_final_response(self):
        """Test getting final response."""
        async def event_gen():
            yield StreamEvent(type=StreamEventType.MESSAGE_START)
            yield StreamEvent(
                type=StreamEventType.CONTENT_BLOCK_START,
                data={"content_block": {"type": "text"}},
            )
            yield StreamEvent(
                type=StreamEventType.TEXT_DELTA,
                data={"text": "Hello"},
            )
            yield StreamEvent(type=StreamEventType.CONTENT_BLOCK_STOP)
            yield StreamEvent(type=StreamEventType.MESSAGE_STOP)
        
        stream = StreamingResponse(
            event_stream=event_gen(),
            model="test-model",
        )
        
        response = await stream.collect()
        
        assert response.model == "test-model"
        assert response.get_text() == "Hello"
    
    async def test_text_callback(self):
        """Test text callback."""
        async def event_gen():
            yield StreamEvent(type=StreamEventType.MESSAGE_START)
            yield StreamEvent(
                type=StreamEventType.TEXT_DELTA,
                data={"text": "Hello"},
            )
            yield StreamEvent(type=StreamEventType.MESSAGE_STOP)
        
        stream = StreamingResponse(
            event_stream=event_gen(),
            model="test-model",
        )
        
        received_text = []
        
        async def on_text(text: str):
            received_text.append(text)
        
        stream.on_text(on_text)
        await stream.collect()
        
        assert received_text == ["Hello"]


# =============================================================================
# Provider Tests
# =============================================================================

class TestMockProvider:
    """Tests for MockProvider."""
    
    def test_provider_name(self):
        """Test provider name."""
        provider = MockProvider()
        assert provider.name == "mock"
    
    def test_supported_models(self):
        """Test supported models."""
        provider = MockProvider()
        assert "mock-model" in provider.supported_models
    
    @pytest.mark.asyncio
    async def test_complete(self):
        """Test completion."""
        provider = MockProvider(default_response="Hello from mock!")
        
        messages = [LLMMessage.user("Hi")]
        config = LLMConfig(model="mock-model")
        
        response = await provider.complete(messages, config)
        
        assert response.get_text() == "Hello from mock!"
        assert provider.call_count == 1
    
    @pytest.mark.asyncio
    async def test_complete_with_queued_responses(self):
        """Test completion with queued responses."""
        response1 = LLMResponse(
            model="mock",
            content=[TextContent(text="First response")],
        )
        response2 = LLMResponse(
            model="mock",
            content=[TextContent(text="Second response")],
        )
        
        provider = MockProvider(responses=[response1, response2])
        
        messages = [LLMMessage.user("Hi")]
        config = LLMConfig(model="mock-model")
        
        r1 = await provider.complete(messages, config)
        r2 = await provider.complete(messages, config)
        
        assert r1.get_text() == "First response"
        assert r2.get_text() == "Second response"
    
    @pytest.mark.asyncio
    async def test_stream(self):
        """Test streaming."""
        provider = MockProvider(default_response="Hello world")
        
        messages = [LLMMessage.user("Hi")]
        config = LLMConfig(model="mock-model")
        
        stream = await provider.stream(messages, config)
        text = await stream.collect_text()
        
        assert "Hello" in text
        assert "world" in text


# =============================================================================
# Tool Executor Tests
# =============================================================================

class TestToolExecutor:
    """Tests for ToolExecutor."""
    
    def test_register_tool(self):
        """Test registering a tool."""
        executor = ToolExecutor()
        
        async def my_tool(x: int) -> int:
            return x * 2
        
        executor.register(
            "double",
            my_tool,
            {"name": "double", "description": "Double a number"},
        )
        
        assert executor.has_tool("double")
        assert len(executor.get_schemas()) == 1
    
    def test_unregister_tool(self):
        """Test unregistering a tool."""
        executor = ToolExecutor()
        
        async def my_tool():
            pass
        
        executor.register("test", my_tool)
        executor.unregister("test")
        
        assert not executor.has_tool("test")
    
    @pytest.mark.asyncio
    async def test_execute_tool(self):
        """Test executing a tool."""
        executor = ToolExecutor()
        
        async def add(a: int, b: int) -> int:
            return a + b
        
        executor.register("add", add)
        
        tool_use = ToolUseContent(
            id="tool-1",
            name="add",
            input={"a": 2, "b": 3},
        )
        
        result = await executor.execute(tool_use)
        
        assert result.tool_use_id == "tool-1"
        assert result.content == "5"
        assert result.is_error is False
    
    @pytest.mark.asyncio
    async def test_execute_unknown_tool(self):
        """Test executing unknown tool."""
        executor = ToolExecutor()
        
        tool_use = ToolUseContent(
            id="tool-1",
            name="unknown",
            input={},
        )
        
        result = await executor.execute(tool_use)
        
        assert result.is_error is True
        assert "Unknown tool" in result.content
    
    @pytest.mark.asyncio
    async def test_execute_tool_error(self):
        """Test tool execution error."""
        executor = ToolExecutor()
        
        async def failing_tool():
            raise ValueError("Something went wrong")
        
        executor.register("fail", failing_tool)
        
        tool_use = ToolUseContent(
            id="tool-1",
            name="fail",
            input={},
        )
        
        result = await executor.execute(tool_use)
        
        assert result.is_error is True
        assert "Something went wrong" in result.content
    
    @pytest.mark.asyncio
    async def test_execute_all_parallel(self):
        """Test executing multiple tools in parallel."""
        executor = ToolExecutor()
        
        async def double(x: int) -> int:
            return x * 2
        
        executor.register("double", double)
        
        tool_uses = [
            ToolUseContent(id="1", name="double", input={"x": 1}),
            ToolUseContent(id="2", name="double", input={"x": 2}),
            ToolUseContent(id="3", name="double", input={"x": 3}),
        ]
        
        results = await executor.execute_all(tool_uses, parallel=True)
        
        assert len(results) == 3
        assert results[0].content == "2"
        assert results[1].content == "4"
        assert results[2].content == "6"


# =============================================================================
# LLM Client Tests
# =============================================================================

class TestLLMClientConfig:
    """Tests for LLMClientConfig."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = LLMClientConfig()
        
        assert config.default_model == "claude-sonnet-4-20250514"
        assert config.default_max_tokens == 4096
        assert config.max_retries == 3
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = LLMClientConfig(
            default_model="claude-3-opus-20240229",
            default_max_tokens=8192,
            max_retries=5,
        )
        
        assert config.default_model == "claude-3-opus-20240229"
        assert config.default_max_tokens == 8192
        assert config.max_retries == 5


@pytest.mark.asyncio
class TestLLMClient:
    """Tests for LLMClient."""
    
    async def test_create_client(self):
        """Test creating a client."""
        provider = MockProvider()
        client = LLMClient(provider=provider)
        
        assert client.provider == provider
    
    async def test_complete_with_string(self):
        """Test completion with string prompt."""
        provider = MockProvider(default_response="Hello!")
        client = LLMClient(provider=provider)
        
        response = await client.complete("Hi there")
        
        assert response.get_text() == "Hello!"
        assert client.total_requests == 1
    
    async def test_complete_with_messages(self):
        """Test completion with message list."""
        provider = MockProvider(default_response="Hello!")
        client = LLMClient(provider=provider)
        
        messages = [
            LLMMessage.system("You are helpful."),
            LLMMessage.user("Hi"),
        ]
        
        response = await client.complete(messages)
        
        assert response.get_text() == "Hello!"
    
    async def test_stream(self):
        """Test streaming completion."""
        provider = MockProvider(default_response="Hello world")
        client = LLMClient(provider=provider)
        
        stream = await client.stream("Hi")
        text = await stream.collect_text()
        
        assert "Hello" in text
    
    async def test_chat(self):
        """Test chat continuation."""
        provider = MockProvider(default_response="I'm doing well!")
        client = LLMClient(provider=provider)
        
        conv = Conversation(system_prompt="You are friendly.")
        
        response = await client.chat(conv, "How are you?")
        
        assert response.get_text() == "I'm doing well!"
        assert len(conv.messages) == 2  # User + Assistant
    
    async def test_register_tool(self):
        """Test registering a tool."""
        client = LLMClient(provider=MockProvider())
        
        async def search(query: str) -> str:
            return f"Results for: {query}"
        
        client.register_tool(
            "search",
            search,
            description="Search the web",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                },
            },
        )
        
        assert client.tool_executor.has_tool("search")
    
    async def test_token_tracking(self):
        """Test token usage tracking."""
        response = LLMResponse(
            model="mock",
            content=[TextContent(text="Hello")],
            usage=LLMUsage(input_tokens=10, output_tokens=5),
        )
        provider = MockProvider(responses=[response])
        client = LLMClient(provider=provider)
        
        await client.complete("Hi")
        
        assert client.total_tokens.input_tokens == 10
        assert client.total_tokens.output_tokens == 5


@pytest.mark.asyncio
class TestAgentLoop:
    """Tests for agent loop functionality."""
    
    async def test_simple_agent_loop(self):
        """Test simple agent loop without tools."""
        provider = MockProvider(default_response="Done!")
        client = LLMClient(provider=provider)
        
        conv = await client.run_agent_loop("Do something")
        
        assert len(conv.messages) >= 2  # User + Assistant
        assert conv.get_last_message().get_text() == "Done!"
    
    async def test_agent_loop_with_callback(self):
        """Test agent loop with response callback."""
        provider = MockProvider(default_response="Done!")
        client = LLMClient(provider=provider)
        
        responses = []
        
        async def on_response(response: LLMResponse):
            responses.append(response)
        
        await client.run_agent_loop("Do something", on_response=on_response)
        
        assert len(responses) == 1


class TestCreateClient:
    """Tests for create_client factory function."""
    
    def test_create_mock_client(self):
        """Test creating mock client."""
        client = create_client(provider="mock")
        
        assert client.provider.name == "mock"
    
    def test_create_anthropic_client(self):
        """Test creating Anthropic client."""
        client = create_client(provider="anthropic", api_key="test-key")
        
        assert client.provider.name == "anthropic"
    
    def test_create_unknown_provider(self):
        """Test creating client with unknown provider."""
        with pytest.raises(ValueError, match="Unknown provider"):
            create_client(provider="unknown")


# =============================================================================
# Integration Tests
# =============================================================================

@pytest.mark.asyncio
class TestLLMIntegration:
    """Integration tests for LLM module."""
    
    async def test_full_conversation_flow(self):
        """Test full conversation flow."""
        # Create responses for multi-turn conversation
        responses = [
            LLMResponse(
                model="mock",
                content=[TextContent(text="Hello! How can I help?")],
            ),
            LLMResponse(
                model="mock",
                content=[TextContent(text="The weather is sunny.")],
            ),
        ]
        
        provider = MockProvider(responses=responses)
        client = LLMClient(provider=provider)
        
        conv = Conversation(system_prompt="You are a helpful assistant.")
        
        # First turn
        r1 = await client.chat(conv, "Hi!")
        assert r1.get_text() == "Hello! How can I help?"
        
        # Second turn
        r2 = await client.chat(conv, "What's the weather?")
        assert r2.get_text() == "The weather is sunny."
        
        # Check conversation history
        assert len(conv.messages) == 4  # 2 user + 2 assistant
    
    async def test_tool_execution_flow(self):
        """Test tool execution flow."""
        # Response with tool use
        tool_response = LLMResponse(
            model="mock",
            content=[
                TextContent(text="Let me search for that."),
                ToolUseContent(
                    id="tool-1",
                    name="search",
                    input={"query": "weather"},
                ),
            ],
            stop_reason=StopReason.TOOL_USE,
        )
        
        # Final response after tool
        final_response = LLMResponse(
            model="mock",
            content=[TextContent(text="The weather is sunny!")],
            stop_reason=StopReason.END_TURN,
        )
        
        provider = MockProvider(responses=[tool_response, final_response])
        client = LLMClient(provider=provider)
        
        # Register the search tool
        async def search(query: str) -> str:
            return f"Search results for: {query}"
        
        client.register_tool(
            "search",
            search,
            description="Search the web",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                },
            },
        )
        
        # Track tool uses
        tool_uses_seen = []
        
        async def on_tool_use(tool_use: ToolUseContent):
            tool_uses_seen.append(tool_use)
        
        # Run agent loop
        conv = await client.run_agent_loop(
            "What's the weather?",
            on_tool_use=on_tool_use,
        )
        
        # Verify tool was called
        assert len(tool_uses_seen) == 1
        assert tool_uses_seen[0].name == "search"
        
        # Verify final response
        assert conv.get_last_message().get_text() == "The weather is sunny!"
    
    async def test_streaming_with_tool_use(self):
        """Test streaming with tool use."""
        provider = MockProvider(default_response="Streaming response")
        client = LLMClient(provider=provider)
        
        text_chunks = []
        
        async def on_text(text: str):
            text_chunks.append(text)
        
        conv = await client.run_agent_loop_streaming(
            "Hello",
            on_text=on_text,
        )
        
        # Verify text was streamed
        assert len(text_chunks) > 0
        assert "".join(text_chunks).strip() == "Streaming response"