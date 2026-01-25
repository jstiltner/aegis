# Quickstart Guide

Get up and running with the Autonomous Agent Infrastructure in minutes.

## Prerequisites

- Python 3.11+
- pip or uv package manager

## Installation

### Using pip

```bash
# Clone the repository
git clone https://github.com/your-org/autonomous-agent.git
cd autonomous-agent

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"
```

### Using uv (recommended)

```bash
# Clone the repository
git clone https://github.com/your-org/autonomous-agent.git
cd autonomous-agent

# Install with uv
uv sync
```

## Quick Start

### 1. Start the API Server

```bash
# Start the server
python -m uvicorn agent_runtime.api.server:app --host 127.0.0.1 --port 8000

# Or with auto-reload for development
python -m uvicorn agent_runtime.api.server:app --reload
```

### 2. Create Your First Agent

```python
import asyncio
from aegis.core import AgentState, AgentStatus
from aegis.state_machine import StateMachine
from aegis.llm import create_client

async def main():
    # Create LLM client (using mock for demo)
    client = create_client("mock")
    
    # Initialize state machine
    machine = StateMachine()
    await machine.initialize(
        agent_id="my-first-agent",
        system_prompt="You are a helpful assistant.",
    )
    
    # Process user input
    from aegis.state_machine import StateEvent
    
    event = StateEvent(
        type="user_input",
        data={"content": "Hello! What can you do?"},
    )
    
    new_state = await machine.process_event(event)
    print(f"Agent status: {new_state.status}")
    
    # Get LLM response
    response = await client.complete("Hello! What can you do?")
    print(f"Response: {response.get_text()}")

asyncio.run(main())
```

### 3. Use the CLI

```bash
# Create a new session
python -m agent_runtime.cli session create --name "My Session"

# Send a message
python -m agent_runtime.cli session message <session-id> "Hello!"

# List sessions
python -m agent_runtime.cli session list

# Create a checkpoint
python -m agent_runtime.cli checkpoint create <session-id>
```

### 4. Access the Web UI

Open your browser to `http://localhost:8000` to access the web interface.

## Core Concepts

### State Machine

The state machine manages agent execution state:

```python
from aegis.state_machine import StateMachine, StateEvent

machine = StateMachine()
await machine.initialize(agent_id="agent-1")

# Process events
event = StateEvent(type="user_input", data={"content": "Hello"})
state = await machine.process_event(event)

# Create checkpoint
checkpoint_id = await machine.checkpoint()

# Restore from checkpoint
await machine.restore(checkpoint_id)
```

### Tool Gateway

Execute tools with policy enforcement:

```python
from aegis.tools import ToolGateway, Tool, ToolDefinition

# Create gateway
gateway = ToolGateway()

# Register a tool
@gateway.registry.register
def search_web(query: str) -> str:
    """Search the web for information."""
    return f"Results for: {query}"

# Invoke tool
result = await gateway.invoke(
    tool_name="search_web",
    arguments={"query": "AI news"},
    agent_id="agent-1",
)
print(result.output)
```

### Commitments

Create and verify agent commitments:

```python
from aegis.commitments import CommitmentManager
from datetime import datetime, timezone, timedelta

manager = CommitmentManager()

# Create commitment
commitment = manager.create_commitment(
    debtor="agent-1",
    creditor="user",
    action="complete_task",
    condition="task_data_available",
    deadline=datetime.now(timezone.utc) + timedelta(hours=1),
)

# Activate when ready
manager.activate_commitment(commitment.id)

# Verify progress
result = manager.verify_commitment(
    commitment.id,
    context={"task_data_available": True},
)
```

### Audit Logging

Track all agent actions:

```python
from aegis.audit import AuditLogger, EventFactory

logger = AuditLogger(agent_id="agent-1")
factory = EventFactory(agent_id="agent-1")

# Log tool invocation
event = factory.tool_invoked(
    tool_name="search_web",
    arguments={"query": "test"},
)
logger.log(event)

# Query events
events = logger.storage.query_events(agent_id="agent-1")
```

## Examples

### Run the Basic Agent Demo

```bash
python examples/basic_agent.py
```

### Run the Tool Usage Demo

```bash
python examples/tool_usage.py
```

### Run the Checkpoint Demo

```bash
python examples/checkpoint_demo.py
```

### Run the Violation Recovery Demo

```bash
python examples/violation_recovery_demo.py
```

## API Usage

### REST API

```bash
# Create session
curl -X POST http://localhost:8000/api/v1/sessions \
  -H "Content-Type: application/json" \
  -d '{"name": "My Session"}'

# Send message
curl -X POST http://localhost:8000/api/v1/sessions/{session_id}/messages \
  -H "Content-Type: application/json" \
  -d '{"content": "Hello!", "role": "user"}'

# List tools
curl http://localhost:8000/api/v1/tools
```

### WebSocket

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/sessions/{session_id}');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Received:', data);
};

ws.send(JSON.stringify({
  type: 'message',
  content: 'Hello!'
}));
```

## Configuration

### Environment Variables

```bash
# API settings
export API_HOST=0.0.0.0
export API_PORT=8000

# LLM settings
export ANTHROPIC_API_KEY=your-api-key
export DEFAULT_MODEL=claude-sonnet-4-20250514

# Storage settings
export CHECKPOINT_DIR=./checkpoints
export AUDIT_LOG_DIR=./logs
```

### Configuration File

Create `config.yaml`:

```yaml
api:
  host: 0.0.0.0
  port: 8000

llm:
  provider: anthropic
  model: claude-sonnet-4-20250514
  temperature: 0.7
  max_tokens: 4096

tools:
  rate_limit: 60  # per minute
  cache_ttl: 300  # seconds

checkpoints:
  auto_checkpoint: true
  interval: 10  # transitions
  storage: file
  path: ./checkpoints

audit:
  enabled: true
  storage: file
  path: ./logs
```

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=agent_runtime

# Run specific test file
pytest tests/unit/test_state.py

# Run with verbose output
pytest -v
```

## Next Steps

- Read the [Architecture Overview](architecture.md)
- Explore the [API Reference](api-reference.md)
- Learn about [GCL Integration](gcl-integration.md)
- Check out more [Examples](../examples/)

## Getting Help

- GitHub Issues: Report bugs and request features
- Documentation: Full documentation at `/docs`
- Examples: Working examples in `/examples`
