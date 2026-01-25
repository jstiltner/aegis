"""
Unit tests for the GCL integration module.
"""

import pytest
from datetime import datetime, timezone, timedelta

from aegis.gcl import (
    GCLAdapter,
    CommitmentConfig,
    CommitmentContext,
    RuntimeCommitment,
    CommitmentStatus,
    CommitmentResult,
    ToolCommitment,
    RuntimeVerifier,
    VerificationHook,
    CommitmentManager,
    CommitmentPolicy,
)
from aegis.gcl.verifier import (
    VerificationContext,
    ExpressionEvaluator,
)
from aegis.gcl.models import (
    CommitmentTemplate,
    TOOL_EXECUTION_TEMPLATE,
)


# =============================================================================
# RuntimeCommitment Tests
# =============================================================================

class TestRuntimeCommitment:
    """Tests for RuntimeCommitment model."""
    
    def test_create_commitment(self):
        """Test creating a runtime commitment."""
        commitment = RuntimeCommitment(
            agent_id="agent-1",
            action_type="tool:search",
            description="Execute search",
            success_condition="tool_success",
        )
        
        assert commitment.agent_id == "agent-1"
        assert commitment.action_type == "tool:search"
        assert commitment.description == "Execute search"
        assert commitment.success_condition == "tool_success"
        assert commitment.status == CommitmentStatus.PENDING
        assert commitment.id is not None
    
    def test_commitment_with_failure_conditions(self):
        """Test commitment with failure conditions."""
        commitment = RuntimeCommitment(
            agent_id="agent-1",
            action_type="task",
            description="Complete task",
            success_condition="task_completed",
            failure_conditions={
                "timeout": "duration_seconds > 60",
                "error": "task_error",
            },
        )
        
        assert len(commitment.failure_conditions) == 2
        assert "timeout" in commitment.failure_conditions
        assert "error" in commitment.failure_conditions
    
    def test_commitment_expiration(self):
        """Test commitment expiration."""
        # Not expired
        commitment = RuntimeCommitment(
            agent_id="agent-1",
            action_type="task",
            description="Task",
            success_condition="true",
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        )
        assert not commitment.is_expired()
        
        # Expired
        expired = RuntimeCommitment(
            agent_id="agent-1",
            action_type="task",
            description="Task",
            success_condition="true",
            expires_at=datetime.now(timezone.utc) - timedelta(hours=1),
        )
        assert expired.is_expired()
    
    def test_commitment_status_transitions(self):
        """Test commitment status transitions."""
        commitment = RuntimeCommitment(
            agent_id="agent-1",
            action_type="task",
            description="Task",
            success_condition="true",
        )
        
        # Activate
        activated = commitment.activate()
        assert activated.status == CommitmentStatus.ACTIVE
        
        # Fulfill
        fulfilled = activated.fulfill()
        assert fulfilled.status == CommitmentStatus.FULFILLED
        
        # Fail
        failed = commitment.fail("timeout")
        assert failed.status == CommitmentStatus.FAILED
        assert failed.metadata.get("failure_mode") == "timeout"
        
        # Cancel
        cancelled = commitment.cancel()
        assert cancelled.status == CommitmentStatus.CANCELLED
    
    def test_commitment_to_dict(self):
        """Test converting commitment to dictionary."""
        commitment = RuntimeCommitment(
            agent_id="agent-1",
            action_type="task",
            description="Task",
            success_condition="true",
        )
        
        d = commitment.to_dict()
        
        assert d["agent_id"] == "agent-1"
        assert d["action_type"] == "task"
        assert d["status"] == "pending"
    
    def test_commitment_from_dict(self):
        """Test creating commitment from dictionary."""
        d = {
            "agent_id": "agent-1",
            "action_type": "task",
            "description": "Task",
            "success_condition": "true",
            "status": "active",
        }
        
        commitment = RuntimeCommitment.from_dict(d)
        
        assert commitment.agent_id == "agent-1"
        assert commitment.status == CommitmentStatus.ACTIVE


class TestToolCommitment:
    """Tests for ToolCommitment model."""
    
    def test_create_tool_commitment(self):
        """Test creating a tool commitment."""
        tc = ToolCommitment(
            tool_name="search",
            expected_outcome="Find relevant results",
            max_duration_seconds=30.0,
            must_succeed=True,
        )
        
        assert tc.tool_name == "search"
        assert tc.expected_outcome == "Find relevant results"
        assert tc.max_duration_seconds == 30.0
        assert tc.must_succeed is True
    
    def test_to_runtime_commitment(self):
        """Test converting to runtime commitment."""
        tc = ToolCommitment(
            tool_name="search",
            expected_outcome="Find results",
            max_duration_seconds=30.0,
        )
        
        commitment = tc.to_runtime_commitment(
            agent_id="agent-1",
            tool_call_id="call-123",
            stake=2.0,
            confidence=0.9,
        )
        
        assert commitment.agent_id == "agent-1"
        assert commitment.action_type == "tool:search"
        assert commitment.tool_call_id == "call-123"
        assert commitment.stake == 2.0
        assert commitment.confidence == 0.9
        assert "timeout" in commitment.failure_conditions


class TestCommitmentTemplate:
    """Tests for CommitmentTemplate."""
    
    def test_instantiate_template(self):
        """Test instantiating a template."""
        template = CommitmentTemplate(
            name="test",
            description="Test {param}",
            action_type="test",
            success_condition_template="value == {expected}",
            failure_condition_templates={
                "wrong": "value != {expected}",
            },
            parameters=["param", "expected"],
        )
        
        commitment = template.instantiate(
            agent_id="agent-1",
            params={"param": "foo", "expected": "42"},
        )
        
        assert commitment.agent_id == "agent-1"
        assert commitment.success_condition == "value == 42"
        assert commitment.failure_conditions["wrong"] == "value != 42"
    
    def test_missing_parameters(self):
        """Test error on missing parameters."""
        template = CommitmentTemplate(
            name="test",
            description="Test",
            action_type="test",
            success_condition_template="value == {expected}",
            parameters=["expected"],
        )
        
        with pytest.raises(ValueError, match="Missing required parameters"):
            template.instantiate(agent_id="agent-1", params={})
    
    def test_builtin_template(self):
        """Test using built-in template."""
        commitment = TOOL_EXECUTION_TEMPLATE.instantiate(
            agent_id="agent-1",
            params={"tool_name": "search", "timeout": "30"},
        )
        
        assert commitment.action_type == "tool_execution"
        assert "search" in commitment.success_condition


# =============================================================================
# ExpressionEvaluator Tests
# =============================================================================

class TestExpressionEvaluator:
    """Tests for ExpressionEvaluator."""
    
    @pytest.fixture
    def evaluator(self):
        """Create an evaluator."""
        return ExpressionEvaluator()
    
    def test_evaluate_simple_comparison(self, evaluator):
        """Test simple comparison evaluation."""
        assert evaluator.evaluate("x > 5", {"x": 10}) is True
        assert evaluator.evaluate("x > 5", {"x": 3}) is False
        assert evaluator.evaluate("x == 5", {"x": 5}) is True
    
    def test_evaluate_logical_operators(self, evaluator):
        """Test logical operator evaluation."""
        assert evaluator.evaluate("x > 0 and y > 0", {"x": 1, "y": 1}) is True
        assert evaluator.evaluate("x > 0 and y > 0", {"x": 1, "y": -1}) is False
        assert evaluator.evaluate("x > 0 or y > 0", {"x": -1, "y": 1}) is True
        assert evaluator.evaluate("not x", {"x": False}) is True
    
    def test_evaluate_boolean_literals(self, evaluator):
        """Test boolean literal evaluation."""
        assert evaluator.evaluate("true", {}) is True
        assert evaluator.evaluate("false", {}) is False
        assert evaluator.evaluate("True", {}) is True
        assert evaluator.evaluate("False", {}) is False
    
    def test_evaluate_in_operator(self, evaluator):
        """Test 'in' operator evaluation."""
        assert evaluator.evaluate("'a' in items", {"items": ["a", "b"]}) is True
        assert evaluator.evaluate("'c' in items", {"items": ["a", "b"]}) is False
    
    def test_unsafe_expression_rejected(self, evaluator):
        """Test that unsafe expressions are rejected."""
        with pytest.raises(ValueError):
            evaluator.evaluate("__import__('os')", {})
        
        with pytest.raises(ValueError):
            evaluator.evaluate("exec('print(1)')", {})


# =============================================================================
# RuntimeVerifier Tests
# =============================================================================

class TestRuntimeVerifier:
    """Tests for RuntimeVerifier."""
    
    @pytest.fixture
    def verifier(self):
        """Create a verifier."""
        return RuntimeVerifier()
    
    @pytest.mark.asyncio
    async def test_verify_success(self, verifier):
        """Test successful verification."""
        commitment = RuntimeCommitment(
            agent_id="agent-1",
            action_type="task",
            description="Task",
            success_condition="completed == True",
        )
        
        context = VerificationContext(
            post_state={"completed": True},
        )
        
        result = await verifier.verify(commitment, context)
        
        assert result.success is True
        assert result.status == CommitmentStatus.FULFILLED
    
    @pytest.mark.asyncio
    async def test_verify_failure_condition(self, verifier):
        """Test failure condition triggered."""
        commitment = RuntimeCommitment(
            agent_id="agent-1",
            action_type="task",
            description="Task",
            success_condition="completed == True",
            failure_conditions={
                "timeout": "duration > 60",
            },
        )
        
        context = VerificationContext(
            post_state={"completed": False, "duration": 100},
        )
        
        result = await verifier.verify(commitment, context)
        
        assert result.success is False
        assert result.failure_mode == "timeout"
    
    @pytest.mark.asyncio
    async def test_verify_success_condition_not_met(self, verifier):
        """Test success condition not met."""
        commitment = RuntimeCommitment(
            agent_id="agent-1",
            action_type="task",
            description="Task",
            success_condition="completed == True",
        )
        
        context = VerificationContext(
            post_state={"completed": False},
        )
        
        result = await verifier.verify(commitment, context)
        
        assert result.success is False
        assert result.failure_mode == "success_condition_not_met"
    
    @pytest.mark.asyncio
    async def test_verify_expired_commitment(self, verifier):
        """Test verifying an expired commitment."""
        commitment = RuntimeCommitment(
            agent_id="agent-1",
            action_type="task",
            description="Task",
            success_condition="true",
            expires_at=datetime.now(timezone.utc) - timedelta(hours=1),
        )
        
        context = VerificationContext()
        
        result = await verifier.verify(commitment, context)
        
        assert result.success is False
        assert result.status == CommitmentStatus.EXPIRED
    
    @pytest.mark.asyncio
    async def test_verify_batch(self, verifier):
        """Test batch verification."""
        commitments = [
            RuntimeCommitment(
                agent_id="agent-1",
                action_type="task",
                description="Task 1",
                success_condition="value > 0",
            ),
            RuntimeCommitment(
                agent_id="agent-1",
                action_type="task",
                description="Task 2",
                success_condition="value > 10",
            ),
        ]
        
        context = VerificationContext(
            post_state={"value": 5},
        )
        
        results = await verifier.verify_batch(commitments, context)
        
        assert len(results) == 2
        assert results[0].success is True  # 5 > 0
        assert results[1].success is False  # 5 > 10 is false


# =============================================================================
# CommitmentManager Tests
# =============================================================================

class TestCommitmentManager:
    """Tests for CommitmentManager."""
    
    @pytest.fixture
    def manager(self):
        """Create a manager."""
        return CommitmentManager()
    
    @pytest.mark.asyncio
    async def test_create_commitment(self, manager):
        """Test creating a commitment."""
        commitment = await manager.create_commitment(
            agent_id="agent-1",
            action_type="task",
            description="Task",
            success_condition="completed",
        )
        
        assert commitment.agent_id == "agent-1"
        assert commitment.status == CommitmentStatus.PENDING
        
        # Should be retrievable
        retrieved = manager.get_commitment(commitment.id)
        assert retrieved is not None
        assert retrieved.id == commitment.id
    
    @pytest.mark.asyncio
    async def test_activate_commitment(self, manager):
        """Test activating a commitment."""
        commitment = await manager.create_commitment(
            agent_id="agent-1",
            action_type="task",
            description="Task",
            success_condition="completed",
        )
        
        activated = await manager.activate_commitment(commitment.id)
        
        assert activated is not None
        assert activated.status == CommitmentStatus.ACTIVE
    
    @pytest.mark.asyncio
    async def test_verify_commitment(self, manager):
        """Test verifying a commitment."""
        commitment = await manager.create_commitment(
            agent_id="agent-1",
            action_type="task",
            description="Task",
            success_condition="completed == True",
        )
        
        context = VerificationContext(
            post_state={"completed": True},
        )
        
        result = await manager.verify_commitment(commitment.id, context)
        
        assert result is not None
        assert result.success is True
        
        # Commitment should be updated
        updated = manager.get_commitment(commitment.id)
        assert updated.status == CommitmentStatus.FULFILLED
    
    @pytest.mark.asyncio
    async def test_cancel_commitment(self, manager):
        """Test cancelling a commitment."""
        commitment = await manager.create_commitment(
            agent_id="agent-1",
            action_type="task",
            description="Task",
            success_condition="completed",
        )
        
        cancelled = await manager.cancel_commitment(commitment.id)
        
        assert cancelled is not None
        assert cancelled.status == CommitmentStatus.CANCELLED
    
    @pytest.mark.asyncio
    async def test_get_agent_commitments(self, manager):
        """Test getting commitments by agent."""
        await manager.create_commitment(
            agent_id="agent-1",
            action_type="task1",
            description="Task 1",
            success_condition="true",
        )
        await manager.create_commitment(
            agent_id="agent-1",
            action_type="task2",
            description="Task 2",
            success_condition="true",
        )
        await manager.create_commitment(
            agent_id="agent-2",
            action_type="task3",
            description="Task 3",
            success_condition="true",
        )
        
        agent1_commitments = manager.get_agent_commitments("agent-1")
        
        assert len(agent1_commitments) == 2
    
    @pytest.mark.asyncio
    async def test_policy_max_commitments(self, manager):
        """Test policy enforcement for max commitments."""
        policy = CommitmentPolicy(max_active_commitments=2)
        manager = CommitmentManager(policy=policy)
        
        # Create and activate 2 commitments
        c1 = await manager.create_commitment(
            agent_id="agent-1",
            action_type="task1",
            description="Task 1",
            success_condition="true",
        )
        await manager.activate_commitment(c1.id)
        
        c2 = await manager.create_commitment(
            agent_id="agent-1",
            action_type="task2",
            description="Task 2",
            success_condition="true",
        )
        await manager.activate_commitment(c2.id)
        
        # Third should fail
        with pytest.raises(ValueError, match="maximum active commitments"):
            c3 = await manager.create_commitment(
                agent_id="agent-1",
                action_type="task3",
                description="Task 3",
                success_condition="true",
            )
            await manager.activate_commitment(c3.id)
    
    @pytest.mark.asyncio
    async def test_get_stats(self, manager):
        """Test getting statistics."""
        c1 = await manager.create_commitment(
            agent_id="agent-1",
            action_type="task1",
            description="Task 1",
            success_condition="true",
        )
        await manager.activate_commitment(c1.id)
        
        c2 = await manager.create_commitment(
            agent_id="agent-1",
            action_type="task2",
            description="Task 2",
            success_condition="completed",
        )
        
        # Verify c1 as success
        await manager.verify_commitment(
            c1.id,
            VerificationContext(post_state={}),
        )
        
        stats = manager.get_stats("agent-1")
        
        assert stats.total_created == 2
        assert stats.total_fulfilled == 1


# =============================================================================
# GCLAdapter Tests
# =============================================================================

class TestGCLAdapter:
    """Tests for GCLAdapter."""
    
    @pytest.fixture
    def adapter(self):
        """Create an adapter."""
        return GCLAdapter()
    
    @pytest.mark.asyncio
    async def test_create_commitment(self, adapter):
        """Test creating a commitment through adapter."""
        context = CommitmentContext(
            agent_id="agent-1",
            session_id="session-1",
        )
        
        commitment = await adapter.create_commitment(
            context=context,
            action_type="task",
            description="Task",
            success_condition="completed",
        )
        
        assert commitment.agent_id == "agent-1"
        assert commitment.metadata.get("session_id") == "session-1"
    
    @pytest.mark.asyncio
    async def test_create_tool_commitment(self, adapter):
        """Test creating a tool commitment."""
        context = CommitmentContext(
            agent_id="agent-1",
            tool_name="search",
            tool_call_id="call-123",
        )
        
        commitment = await adapter.create_tool_commitment(
            context=context,
            tool_name="search",
            expected_outcome="Find results",
        )
        
        assert commitment.action_type == "tool:search"
        assert commitment.tool_call_id == "call-123"
    
    @pytest.mark.asyncio
    async def test_verify_commitment(self, adapter):
        """Test verifying a commitment."""
        context = CommitmentContext(agent_id="agent-1")
        
        commitment = await adapter.create_commitment(
            context=context,
            action_type="task",
            description="Task",
            success_condition="completed == True",
        )
        
        result = await adapter.verify_commitment(
            commitment.id,
            post_state={"completed": True},
        )
        
        assert result is not None
        assert result.success is True
    
    @pytest.mark.asyncio
    async def test_verify_tool_execution(self, adapter):
        """Test verifying tool execution."""
        context = CommitmentContext(
            agent_id="agent-1",
            tool_name="search",
        )
        
        commitment = await adapter.create_tool_commitment(
            context=context,
            tool_name="search",
            expected_outcome="Find results",
            max_duration_seconds=30.0,
        )
        
        result = await adapter.verify_tool_execution(
            commitment.id,
            tool_name="search",
            success=True,
            duration_seconds=5.0,
        )
        
        assert result is not None
        assert result.success is True
    
    @pytest.mark.asyncio
    async def test_get_tool_commitments(self, adapter):
        """Test getting tool commitments."""
        context = CommitmentContext(agent_id="agent-1")
        
        await adapter.create_tool_commitment(
            context=context,
            tool_name="search",
            expected_outcome="Search",
        )
        await adapter.create_tool_commitment(
            context=context,
            tool_name="read_file",
            expected_outcome="Read",
        )
        await adapter.create_commitment(
            context=context,
            action_type="task",
            description="Task",
            success_condition="true",
        )
        
        tool_commitments = adapter.get_tool_commitments("agent-1")
        assert len(tool_commitments) == 2
        
        search_commitments = adapter.get_tool_commitments("agent-1", "search")
        assert len(search_commitments) == 1
    
    @pytest.mark.asyncio
    async def test_cancel_commitment(self, adapter):
        """Test cancelling a commitment."""
        context = CommitmentContext(agent_id="agent-1")
        
        commitment = await adapter.create_commitment(
            context=context,
            action_type="task",
            description="Task",
            success_condition="true",
        )
        
        cancelled = await adapter.cancel_commitment(commitment.id)
        
        assert cancelled is not None
        assert cancelled.status == CommitmentStatus.CANCELLED


# =============================================================================
# Verification Hook Tests
# =============================================================================

class TestVerificationHooks:
    """Tests for verification hooks."""
    
    @pytest.mark.asyncio
    async def test_custom_hook(self):
        """Test custom verification hook."""
        events = []
        
        class TestHook(VerificationHook):
            async def before_verify(self, commitment, context):
                events.append(("before", commitment.id))
            
            async def after_verify(self, commitment, context, result):
                events.append(("after", commitment.id, result.success))
        
        verifier = RuntimeVerifier(hooks=[TestHook()])
        
        commitment = RuntimeCommitment(
            agent_id="agent-1",
            action_type="task",
            description="Task",
            success_condition="true",
        )
        
        await verifier.verify(commitment, VerificationContext())
        
        assert len(events) == 2
        assert events[0][0] == "before"
        assert events[1][0] == "after"
        assert events[1][2] is True
