# API Reference

This document provides a comprehensive reference for the Autonomous Agent Infrastructure REST API.

## Base URL

```
http://localhost:8000/api/v1
```

## Authentication

All API requests require authentication via API key:

```http
Authorization: Bearer <api_key>
```

## Common Headers

| Header | Description |
|--------|-------------|
| `X-Request-ID` | Unique request identifier (auto-generated if not provided) |
| `X-Response-Time` | Response time in milliseconds (returned) |
| `Content-Type` | `application/json` |

## Endpoints

### Health

#### GET /health

Check API health status.

**Response:**
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

---

### Sessions

#### POST /sessions

Create a new agent session.

**Request:**
```json
{
  "name": "My Session",
  "agent_id": "agent-123",
  "config": {
    "max_turns": 100,
    "timeout": 3600
  }
}
```

**Response:**
```json
{
  "id": "session-456",
  "name": "My Session",
  "agent_id": "agent-123",
  "status": "active",
  "created_at": "2024-01-15T10:30:00Z",
  "config": {
    "max_turns": 100,
    "timeout": 3600
  }
}
```

#### GET /sessions

List all sessions.

**Query Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `status` | string | Filter by status (active, paused, stopped) |
| `limit` | integer | Maximum results (default: 50) |
| `offset` | integer | Pagination offset |

**Response:**
```json
{
  "sessions": [
    {
      "id": "session-456",
      "name": "My Session",
      "status": "active",
      "created_at": "2024-01-15T10:30:00Z"
    }
  ],
  "total": 1,
  "limit": 50,
  "offset": 0
}
```

#### GET /sessions/{session_id}

Get session details.

**Response:**
```json
{
  "id": "session-456",
  "name": "My Session",
  "agent_id": "agent-123",
  "status": "active",
  "created_at": "2024-01-15T10:30:00Z",
  "message_count": 10,
  "checkpoint_count": 3
}
```

#### DELETE /sessions/{session_id}

Delete a session.

**Response:**
```json
{
  "success": true,
  "message": "Session deleted"
}
```

---

### Messages

#### POST /sessions/{session_id}/messages

Send a message to the agent.

**Request:**
```json
{
  "content": "Hello, agent!",
  "role": "user"
}
```

**Response:**
```json
{
  "id": "msg-789",
  "role": "assistant",
  "content": "Hello! How can I help you today?",
  "created_at": "2024-01-15T10:31:00Z",
  "tool_uses": []
}
```

#### GET /sessions/{session_id}/messages

Get session messages.

**Query Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `limit` | integer | Maximum results |
| `before` | string | Get messages before this ID |
| `after` | string | Get messages after this ID |

**Response:**
```json
{
  "messages": [
    {
      "id": "msg-789",
      "role": "user",
      "content": "Hello, agent!",
      "created_at": "2024-01-15T10:30:00Z"
    },
    {
      "id": "msg-790",
      "role": "assistant",
      "content": "Hello! How can I help you?",
      "created_at": "2024-01-15T10:31:00Z"
    }
  ]
}
```

---

### Checkpoints

#### GET /sessions/{session_id}/checkpoints

List session checkpoints.

**Response:**
```json
{
  "checkpoints": [
    {
      "id": "cp-001",
      "created_at": "2024-01-15T10:30:00Z",
      "state_hash": "abc123",
      "parent_id": null,
      "metadata": {
        "trigger": "auto",
        "transition_count": 5
      }
    }
  ]
}
```

#### POST /sessions/{session_id}/checkpoints

Create a manual checkpoint.

**Request:**
```json
{
  "metadata": {
    "reason": "Before risky operation"
  }
}
```

**Response:**
```json
{
  "id": "cp-002",
  "created_at": "2024-01-15T10:35:00Z",
  "state_hash": "def456"
}
```

#### POST /sessions/{session_id}/checkpoints/{checkpoint_id}/restore

Restore session to a checkpoint.

**Response:**
```json
{
  "success": true,
  "restored_to": "cp-001",
  "new_checkpoint": "cp-003"
}
```

#### POST /sessions/{session_id}/checkpoints/{checkpoint_id}/branch

Create a new session branch from checkpoint.

**Request:**
```json
{
  "name": "Alternative path"
}
```

**Response:**
```json
{
  "session_id": "session-457",
  "branched_from": "cp-001"
}
```

---

### Agents

#### POST /agents

Create a new agent.

**Request:**
```json
{
  "name": "Research Agent",
  "system_prompt": "You are a helpful research assistant.",
  "model": "claude-sonnet-4-20250514",
  "tools": ["web_search", "read_file"],
  "config": {
    "temperature": 0.7,
    "max_tokens": 4096
  }
}
```

**Response:**
```json
{
  "id": "agent-123",
  "name": "Research Agent",
  "model": "claude-sonnet-4-20250514",
  "created_at": "2024-01-15T10:00:00Z"
}
```

#### GET /agents

List all agents.

**Response:**
```json
{
  "agents": [
    {
      "id": "agent-123",
      "name": "Research Agent",
      "model": "claude-sonnet-4-20250514",
      "status": "active"
    }
  ]
}
```

#### GET /agents/{agent_id}

Get agent details.

**Response:**
```json
{
  "id": "agent-123",
  "name": "Research Agent",
  "system_prompt": "You are a helpful research assistant.",
  "model": "claude-sonnet-4-20250514",
  "tools": ["web_search", "read_file"],
  "config": {
    "temperature": 0.7,
    "max_tokens": 4096
  },
  "stats": {
    "total_sessions": 10,
    "total_messages": 150,
    "total_tool_calls": 45
  }
}
```

---

### Tools

#### GET /tools

List available tools.

**Query Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `category` | string | Filter by category |
| `tag` | string | Filter by tag |

**Response:**
```json
{
  "tools": [
    {
      "name": "web_search",
      "description": "Search the web for information",
      "category": "web",
      "parameters": [
        {
          "name": "query",
          "type": "string",
          "required": true,
          "description": "Search query"
        }
      ]
    }
  ]
}
```

#### POST /tools/{tool_name}/invoke

Invoke a tool directly.

**Request:**
```json
{
  "arguments": {
    "query": "latest AI news"
  },
  "context": {
    "agent_id": "agent-123",
    "session_id": "session-456"
  }
}
```

**Response:**
```json
{
  "success": true,
  "result": {
    "results": [
      {
        "title": "AI News Article",
        "url": "https://example.com/article",
        "snippet": "..."
      }
    ]
  },
  "execution_time_ms": 250
}
```

---

### Commitments

#### GET /sessions/{session_id}/commitments

List session commitments.

**Query Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `status` | string | Filter by status |

**Response:**
```json
{
  "commitments": [
    {
      "id": "commit-001",
      "debtor": "agent-123",
      "creditor": "user",
      "action": "complete_task",
      "condition": "task_data_available",
      "status": "active",
      "deadline": "2024-01-15T12:00:00Z",
      "created_at": "2024-01-15T10:00:00Z"
    }
  ]
}
```

#### POST /sessions/{session_id}/commitments

Create a commitment.

**Request:**
```json
{
  "action": "deliver_report",
  "condition": "data_collected",
  "deadline": "2024-01-15T18:00:00Z"
}
```

**Response:**
```json
{
  "id": "commit-002",
  "status": "created",
  "created_at": "2024-01-15T10:30:00Z"
}
```

#### POST /sessions/{session_id}/commitments/{commitment_id}/verify

Verify a commitment.

**Response:**
```json
{
  "verified": true,
  "status": "fulfilled",
  "verification_time": "2024-01-15T11:00:00Z"
}
```

---

### Traces

#### GET /traces

List execution traces.

**Query Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `session_id` | string | Filter by session |
| `start_time` | string | Start time (ISO 8601) |
| `end_time` | string | End time (ISO 8601) |

**Response:**
```json
{
  "traces": [
    {
      "id": "trace-001",
      "session_id": "session-456",
      "start_time": "2024-01-15T10:30:00Z",
      "end_time": "2024-01-15T10:31:00Z",
      "span_count": 5
    }
  ]
}
```

#### GET /traces/{trace_id}

Get trace details with spans.

**Response:**
```json
{
  "id": "trace-001",
  "session_id": "session-456",
  "spans": [
    {
      "id": "span-001",
      "name": "process_message",
      "start_time": "2024-01-15T10:30:00Z",
      "end_time": "2024-01-15T10:30:05Z",
      "duration_ms": 5000,
      "parent_id": null,
      "attributes": {
        "message_id": "msg-789"
      }
    },
    {
      "id": "span-002",
      "name": "llm_call",
      "start_time": "2024-01-15T10:30:01Z",
      "end_time": "2024-01-15T10:30:04Z",
      "duration_ms": 3000,
      "parent_id": "span-001"
    }
  ]
}
```

---

### Audit Events

#### GET /audit/events

Query audit events.

**Query Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `agent_id` | string | Filter by agent |
| `event_type` | string | Filter by event type |
| `severity` | string | Minimum severity |
| `start_time` | string | Start time |
| `end_time` | string | End time |
| `limit` | integer | Maximum results |

**Response:**
```json
{
  "events": [
    {
      "id": "event-001",
      "type": "tool_invoked",
      "severity": "info",
      "timestamp": "2024-01-15T10:30:00Z",
      "agent_id": "agent-123",
      "data": {
        "tool": "web_search",
        "arguments": {"query": "..."}
      }
    }
  ],
  "total": 100
}
```

---

## WebSocket API

### Connect

```
ws://localhost:8000/ws/sessions/{session_id}
```

### Message Types

#### Client → Server

**Send Message:**
```json
{
  "type": "message",
  "content": "Hello, agent!"
}
```

**Create Checkpoint:**
```json
{
  "type": "checkpoint",
  "metadata": {}
}
```

**Pause/Resume:**
```json
{
  "type": "control",
  "action": "pause"
}
```

#### Server → Client

**Agent Response:**
```json
{
  "type": "response",
  "message": {
    "id": "msg-790",
    "role": "assistant",
    "content": "Hello!"
  }
}
```

**Tool Use:**
```json
{
  "type": "tool_use",
  "tool": "web_search",
  "arguments": {"query": "..."},
  "status": "executing"
}
```

**State Update:**
```json
{
  "type": "state_update",
  "status": "thinking",
  "checkpoint_id": "cp-003"
}
```

**Stream Token:**
```json
{
  "type": "stream",
  "token": "Hello"
}
```

---

## Error Responses

All errors follow this format:

```json
{
  "error": {
    "code": "NOT_FOUND",
    "message": "Session not found",
    "details": {
      "session_id": "session-999"
    }
  }
}
```

### Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `BAD_REQUEST` | 400 | Invalid request format |
| `UNAUTHORIZED` | 401 | Missing or invalid authentication |
| `FORBIDDEN` | 403 | Insufficient permissions |
| `NOT_FOUND` | 404 | Resource not found |
| `CONFLICT` | 409 | Resource conflict |
| `RATE_LIMITED` | 429 | Too many requests |
| `INTERNAL_ERROR` | 500 | Server error |

---

## Rate Limits

| Endpoint | Limit |
|----------|-------|
| `/sessions` | 100/minute |
| `/messages` | 60/minute |
| `/tools/invoke` | 30/minute |
| All others | 1000/minute |

Rate limit headers:
```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1705312800
```

---

## Pagination

List endpoints support cursor-based pagination:

```json
{
  "data": [...],
  "pagination": {
    "total": 100,
    "limit": 50,
    "offset": 0,
    "has_more": true,
    "next_cursor": "eyJpZCI6MTAwfQ=="
  }
}
```

Use `cursor` query parameter for next page:
```
GET /sessions?cursor=eyJpZCI6MTAwfQ==
```
