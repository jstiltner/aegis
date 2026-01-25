# API Reference

This document describes the REST API endpoints provided by the Agent Runtime server.

## Base URL

```
http://localhost:8000/api/v1
```

## Authentication

Currently, the API does not require authentication. In production deployments, configure authentication via environment variables or the configuration file.

## Response Format

All responses are JSON. Successful responses return the requested data. Error responses follow this format:

```json
{
  "error": "error_type",
  "message": "Human-readable error message",
  "details": {},
  "trace_id": "request-trace-id"
}
```

## Endpoints

### Health

#### GET /health

Check the health status of the API server.

**Response**

```json
{
  "status": "healthy",
  "version": "0.1.0",
  "uptime_seconds": 3600.5,
  "checks": {
    "api": "healthy",
    "database": "healthy",
    "llm": "healthy"
  }
}
```

---

### Sessions

#### POST /api/v1/sessions

Create a new agent session.

**Request Body**

```json
{
  "agent_id": "agent-001",
  "system_prompt": "You are a helpful assistant.",
  "metadata": {}
}
```

**Response** (201 Created)

```json
{
  "session_id": "sess-abc123",
  "agent_id": "agent-001",
  "status": "active",
  "created_at": "2024-01-15T10:30:00Z",
  "updated_at": "2024-01-15T10:30:00Z",
  "message_count": 0,
  "checkpoint_count": 0,
  "metadata": {}
}
```

#### GET /api/v1/sessions

List all sessions with optional filtering.

**Query Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| agent_id | string | Filter by agent ID |
| status | string | Filter by status (active, paused, completed, error) |
| offset | integer | Pagination offset (default: 0) |
| limit | integer | Pagination limit (default: 20, max: 100) |

**Response**

```json
{
  "sessions": [...],
  "total": 42,
  "offset": 0,
  "limit": 20
}
```

#### GET /api/v1/sessions/{session_id}

Get details of a specific session.

**Response**

```json
{
  "session_id": "sess-abc123",
  "agent_id": "agent-001",
  "status": "active",
  "created_at": "2024-01-15T10:30:00Z",
  "updated_at": "2024-01-15T10:35:00Z",
  "message_count": 5,
  "checkpoint_count": 2,
  "metadata": {}
}
```

#### DELETE /api/v1/sessions/{session_id}

Delete a session and all associated data.

**Response** (204 No Content)

#### POST /api/v1/sessions/{session_id}/messages

Send a message to the agent and receive a response.

**Request Body**

```json
{
  "content": "Hello, can you help me?",
  "role": "user",
  "metadata": {}
}
```

**Response** (201 Created)

```json
{
  "message_id": "msg-xyz789",
  "session_id": "sess-abc123",
  "role": "assistant",
  "content": "Of course! How can I assist you today?",
  "tool_uses": [],
  "created_at": "2024-01-15T10:35:00Z",
  "duration_ms": 1250.5,
  "token_count": 42,
  "metadata": {}
}
```

#### GET /api/v1/sessions/{session_id}/messages

Get all messages in a session.

**Query Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| offset | integer | Pagination offset (default: 0) |
| limit | integer | Pagination limit (default: 50, max: 200) |

**Response**

```json
{
  "messages": [...],
  "total": 15
}
```

#### POST /api/v1/sessions/{session_id}/messages/stream

Send a message and receive a streaming response via Server-Sent Events.

**Request Body**

```json
{
  "content": "Write a poem about coding",
  "role": "user"
}
```

**Response** (text/event-stream)

```
data: {"type": "start"}

data: {"type": "text", "content": "In "}

data: {"type": "text", "content": "lines "}

data: {"type": "text", "content": "of code..."}

data: {"type": "end"}
```

#### GET /api/v1/sessions/{session_id}/checkpoints

Get all checkpoints for a session.

**Response**

```json
{
  "checkpoints": [
    {
      "checkpoint_id": "cp-001",
      "session_id": "sess-abc123",
      "version": 5,
      "created_at": "2024-01-15T10:32:00Z",
      "parent_checkpoint_id": "cp-000",
      "state_summary": {},
      "metadata": {}
    }
  ],
  "total": 3
}
```

#### POST /api/v1/sessions/{session_id}/restore/{checkpoint_id}

Restore a session to a previous checkpoint.

**Response**

```json
{
  "session_id": "sess-abc123",
  "agent_id": "agent-001",
  "status": "active",
  "message_count": 3,
  "checkpoint_count": 1
}
```

#### GET /api/v1/sessions/{session_id}/commitments

Get all commitments for a session.

**Query Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| status | string | Filter by status (pending, active, fulfilled, violated, cancelled, expired) |

**Response**

```json
{
  "commitments": [
    {
      "commitment_id": "C001",
      "session_id": "sess-abc123",
      "debtor": "agent",
      "creditor": "user",
      "antecedent": "request_received",
      "consequent": "provide_response",
      "status": "fulfilled",
      "created_at": "2024-01-15T10:30:00Z",
      "deadline": null,
      "metadata": {}
    }
  ],
  "total": 5
}
```

---

### Agents

#### POST /api/v1/agents

Create a new agent configuration.

**Request Body**

```json
{
  "name": "Code Assistant",
  "description": "An assistant for coding tasks",
  "model": "claude-sonnet-4-20250514",
  "system_prompt": "You are a helpful coding assistant.",
  "tools": ["read_file", "write_file", "execute_command"],
  "metadata": {}
}
```

**Response** (201 Created)

```json
{
  "agent_id": "agent-001",
  "name": "Code Assistant",
  "description": "An assistant for coding tasks",
  "model": "claude-sonnet-4-20250514",
  "system_prompt": "You are a helpful coding assistant.",
  "tools": ["read_file", "write_file", "execute_command"],
  "created_at": "2024-01-15T10:00:00Z",
  "session_count": 0,
  "metadata": {}
}
```

#### GET /api/v1/agents

List all configured agents.

**Response**

```json
{
  "agents": [...],
  "total": 3
}
```

#### GET /api/v1/agents/{agent_id}

Get details of a specific agent.

**Response**

```json
{
  "agent_id": "agent-001",
  "name": "Code Assistant",
  "description": "An assistant for coding tasks",
  "model": "claude-sonnet-4-20250514",
  "system_prompt": "You are a helpful coding assistant.",
  "tools": ["read_file", "write_file"],
  "created_at": "2024-01-15T10:00:00Z",
  "session_count": 5,
  "metadata": {}
}
```

#### PUT /api/v1/agents/{agent_id}

Update an agent configuration.

**Request Body**

Same as POST /api/v1/agents

**Response**

Updated agent object.

#### DELETE /api/v1/agents/{agent_id}

Delete an agent and all associated sessions.

**Response** (204 No Content)

---

### Traces

#### GET /api/v1/traces

List traces with optional filtering.

**Query Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| session_id | string | Filter by session ID |
| start_time | datetime | Filter by start time (ISO 8601) |
| end_time | datetime | Filter by end time (ISO 8601) |
| offset | integer | Pagination offset (default: 0) |
| limit | integer | Pagination limit (default: 20, max: 100) |

**Response**

```json
{
  "traces": [
    {
      "trace_id": "trace-001",
      "session_id": "sess-abc123",
      "root_span_id": "span-001",
      "start_time": "2024-01-15T10:30:00Z",
      "end_time": "2024-01-15T10:30:02Z",
      "duration_ms": 2000,
      "span_count": 5,
      "spans": [],
      "metadata": {}
    }
  ],
  "total": 42
}
```

#### GET /api/v1/traces/{trace_id}

Get details of a specific trace including all spans.

**Response**

```json
{
  "trace_id": "trace-001",
  "session_id": "sess-abc123",
  "root_span_id": "span-001",
  "start_time": "2024-01-15T10:30:00Z",
  "end_time": "2024-01-15T10:30:02Z",
  "duration_ms": 2000,
  "span_count": 5,
  "spans": [
    {
      "span_id": "span-001",
      "trace_id": "trace-001",
      "parent_span_id": null,
      "name": "agent.run",
      "start_time": "2024-01-15T10:30:00Z",
      "end_time": "2024-01-15T10:30:02Z",
      "duration_ms": 2000,
      "status": "ok",
      "attributes": {},
      "events": []
    },
    {
      "span_id": "span-002",
      "trace_id": "trace-001",
      "parent_span_id": "span-001",
      "name": "llm.complete",
      "start_time": "2024-01-15T10:30:00.100Z",
      "end_time": "2024-01-15T10:30:01.500Z",
      "duration_ms": 1400,
      "status": "ok",
      "attributes": {"model": "claude-sonnet-4-20250514"},
      "events": []
    }
  ],
  "metadata": {}
}
```

#### GET /api/v1/traces/{trace_id}/spans

Get all spans in a trace.

**Response**

Array of span objects.

---

### Tools

#### GET /api/v1/tools

List all available tools.

**Query Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| category | string | Filter by category |
| enabled | boolean | Filter by enabled status |

**Response**

```json
{
  "tools": [
    {
      "name": "read_file",
      "description": "Read the contents of a file",
      "parameters": [
        {
          "name": "path",
          "type": "string",
          "description": "Path to the file",
          "required": true,
          "default": null
        }
      ],
      "category": "filesystem",
      "requires_auth": false,
      "enabled": true
    }
  ],
  "total": 10
}
```

#### GET /api/v1/tools/{tool_name}

Get details of a specific tool.

**Response**

Tool object as shown above.

#### POST /api/v1/tools/{tool_name}/invoke

Directly invoke a tool (for testing purposes).

**Request Body**

```json
{
  "tool_name": "read_file",
  "input": {
    "path": "/tmp/test.txt"
  },
  "session_id": null
}
```

**Response**

```json
{
  "tool_name": "read_file",
  "output": "File contents here...",
  "is_error": false,
  "error_message": null,
  "duration_ms": 15.5
}
```

---

## WebSocket API

### Connection

Connect to the WebSocket endpoint for real-time updates:

```
ws://localhost:8000/ws/{session_id}
```

### Messages

**Connection Confirmation**

```json
{
  "type": "connected",
  "session_id": "sess-abc123",
  "connection_id": "conn-xyz789"
}
```

**Broadcast Message**

```json
{
  "type": "broadcast",
  "session_id": "sess-abc123",
  "data": {
    "event": "message_received",
    "content": "..."
  }
}
```

---

## Error Codes

| Status Code | Error Type | Description |
|-------------|------------|-------------|
| 400 | validation_error | Invalid request parameters |
| 401 | unauthorized | Authentication required |
| 403 | forbidden | Insufficient permissions |
| 404 | not_found | Resource not found |
| 409 | conflict | Resource conflict |
| 429 | rate_limited | Too many requests |
| 500 | internal_error | Server error |

---

## Rate Limiting

The API implements rate limiting to prevent abuse. Default limits:

- 100 requests per minute per IP
- 1000 requests per hour per IP

Rate limit headers are included in responses:

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1705312800
```
