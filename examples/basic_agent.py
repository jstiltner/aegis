#!/usr/bin/env python3
"""
Basic Agent Example

This example demonstrates how to create and run a simple autonomous agent
using the Agent Runtime infrastructure.

Features demonstrated:
- Creating an agent with tools
- Running an agentic loop
- Checkpoint and replay
- Commitment tracking
"""

import asyncio
from datetime import datetime, timezone

from aegis.core import AgentState, Event, EventType
from aegis.state_machine import StateMachine, TransitionRule, AgentStatus
from aegis.tools import ToolRegistry, ToolDefinition, ToolPolicy, PolicyAction
from aegis.llm import LLMClient, MockProvider, Conversation
from aegis.gcl import CommitmentManager, RuntimeCommitment, CommitmentStatus
from aegis.audit import AuditLogger, InMemoryAuditStore


async def main():
    """Run the basic agent example."""
    print("=" * 60)
    print("Agent Runtime - Basic Agent Example")
    print("=" * 60)
    
    # ==========================================================================
    # 1. Setup Tool Registry
    # ==========================================================================
    print("\n1. Setting up tool registry...")
    
    registry = ToolRegistry()
    
    # Register a simple calculator tool
    async def calculate(expression: str) -> str:
        """Evaluate a mathematical expression."""
        try:
            # Safe eval for simple math
            result = eval(expression, {"__builtins__": {}}, {})
            return f"Result: {result}"
        except Exception as e:
            return f"Error: {str(e)}"
    
    registry.register(ToolDefinition(
        name="calculate",
        description="Evaluate a mathematical expression",
        parameters={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate",
                },
            },
            "required": ["expression"],
        },
        handler=calculate,
    ))
    
    # Register a greeting tool
    async def greet(name: str) -> str:
        """Generate a greeting."""
        return f"Hello, {name}! Welcome to the Agent Runtime."
    
    registry.register(ToolDefinition(
        name="greet",
        description="Generate a personalized greeting",
        parameters={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name to greet",
                },
            },
            "required": ["name"],
        },
        handler=greet,
    ))
    
    print(f"   Registered {len(registry.list_tools())} tools")
    
    # ==========================================================================
    # 2. Setup Tool Policy
    # ==========================================================================
    print("\n2. Setting up tool policy...")
    
    policy = ToolPolicy()
    
    # Allow all tools by default
    policy.add_rule(
        pattern="*",
        action=PolicyAction.ALLOW,
        description="Allow all tools",
    )
    
    print("   Policy configured: Allow all tools")
    
    # ==========================================================================
    # 3. Setup State Machine
    # ==========================================================================
    print("\n3. Setting up state machine...")
    
    # Create initial state
    initial_state = AgentState(
        agent_id="example-agent",
        session_id="session-001",
    )
    
    # Create state machine with transition rules
    state_machine = StateMachine(
        initial_state=initial_state,
        rules=[
            TransitionRule(
                from_status=AgentStatus.IDLE,
                to_status=AgentStatus.THINKING,
                event_types=[EventType.USER_INPUT],
            ),
            TransitionRule(
                from_status=AgentStatus.THINKING,
                to_status=AgentStatus.EXECUTING,
                event_types=[EventType.TOOL_CALL],
            ),
            TransitionRule(
                from_status=AgentStatus.EXECUTING,
                to_status=AgentStatus.THINKING,
                event_types=[EventType.TOOL_RESULT],
            ),
            TransitionRule(
                from_status=AgentStatus.THINKING,
                to_status=AgentStatus.IDLE,
                event_types=[EventType.LLM_RESPONSE],
            ),
        ],
    )
    
    print(f"   State machine initialized in {state_machine.current_state.status} state")
    
    # ==========================================================================
    # 4. Setup Audit Logger
    # ==========================================================================
    print("\n4. Setting up audit logger...")
    
    audit_store = InMemoryAuditStore()
    audit_logger = AuditLogger(store=audit_store)
    
    print("   Audit logger ready")
    
    # ==========================================================================
    # 5. Setup Commitment Manager
    # ==========================================================================
    print("\n5. Setting up commitment manager...")
    
    commitment_manager = CommitmentManager()
    
    # Create a commitment for responding to user requests
    response_commitment = RuntimeCommitment(
        debtor="agent",
        creditor="user",
        antecedent="user_request_received",
        consequent="provide_helpful_response",
    )
    commitment_manager.add_commitment(response_commitment)
    
    print(f"   Created commitment: {response_commitment.commitment_id[:8]}...")
    
    # ==========================================================================
    # 6. Setup LLM Client
    # ==========================================================================
    print("\n6. Setting up LLM client...")
    
    # Use mock provider for this example
    provider = MockProvider(default_response="I'll help you with that!")
    client = LLMClient(provider=provider)
    
    # Register tools with the client
    for tool in registry.list_tools():
        client.register_tool(
            name=tool.name,
            handler=tool.handler,
            description=tool.description,
            parameters=tool.parameters,
        )
    
    print(f"   LLM client ready with {len(registry.list_tools())} tools")
    
    # ==========================================================================
    # 7. Run Agent Loop
    # ==========================================================================
    print("\n7. Running agent loop...")
    print("-" * 40)
    
    # Simulate user input
    user_message = "Hello! Can you calculate 2 + 2 for me?"
    print(f"\nUser: {user_message}")
    
    # Log the user input event
    await audit_logger.log_event(
        event_type="user.input",
        agent_id="example-agent",
        session_id="session-001",
        data={"content": user_message},
    )
    
    # Process through state machine
    user_event = Event(
        event_type=EventType.USER_INPUT,
        agent_id="example-agent",
        session_id="session-001",
        data={"content": user_message},
    )
    
    result = await state_machine.process_event(user_event)
    print(f"\n   State: {state_machine.current_state.status}")
    
    # Run the agent
    conversation = await client.run_agent_loop(
        prompt=user_message,
        system="You are a helpful assistant with access to tools.",
        max_iterations=3,
    )
    
    # Get the response
    response = conversation.get_last_message()
    if response:
        print(f"\nAgent: {response.get_text()}")
    
    # Log the response
    await audit_logger.log_event(
        event_type="agent.response",
        agent_id="example-agent",
        session_id="session-001",
        data={"content": response.get_text() if response else ""},
    )
    
    # Update commitment status
    commitment_manager.fulfill_commitment(response_commitment.commitment_id)
    print(f"\n   Commitment fulfilled: {response_commitment.commitment_id[:8]}...")
    
    # ==========================================================================
    # 8. Create Checkpoint
    # ==========================================================================
    print("\n8. Creating checkpoint...")
    
    checkpoint = await state_machine.create_checkpoint()
    print(f"   Checkpoint created: {checkpoint.checkpoint_id[:8]}...")
    print(f"   State version: {checkpoint.version}")
    
    # ==========================================================================
    # 9. Show Statistics
    # ==========================================================================
    print("\n9. Session Statistics")
    print("-" * 40)
    
    # Get audit events
    events = await audit_store.query(session_id="session-001")
    print(f"   Total events: {len(events)}")
    
    # Get commitment stats
    stats = commitment_manager.get_stats()
    print(f"   Commitments: {stats.total} total, {stats.fulfilled} fulfilled")
    
    # Get LLM usage
    print(f"   LLM requests: {client.total_requests}")
    print(f"   Total tokens: {client.total_tokens.total_tokens}")
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
