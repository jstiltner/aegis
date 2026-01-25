"""
Verification engine for runtime commitments.

This module provides verification capabilities for commitments,
integrating with the GCL verification engine when available.
"""

from __future__ import annotations

import time
import re
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Callable, Awaitable
from dataclasses import dataclass, field

from .models import (
    RuntimeCommitment,
    CommitmentStatus,
    CommitmentResult,
)


@dataclass
class VerificationContext:
    """
    Context for commitment verification.
    
    Attributes:
        pre_state: State before the action
        post_state: State after the action
        action: The action that was performed
        metadata: Additional context information
    """
    
    pre_state: dict[str, Any] = field(default_factory=dict)
    post_state: dict[str, Any] = field(default_factory=dict)
    action: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_eval_context(self) -> dict[str, Any]:
        """
        Build evaluation context for condition checking.
        
        Returns:
            Combined context dictionary
        """
        context: dict[str, Any] = {}
        
        # Add pre-state with prefix
        for key, value in self.pre_state.items():
            context[f"pre_{key}"] = value
        
        # Add post-state (current values)
        context.update(self.post_state)
        
        # Add action parameters
        context["action"] = self.action
        for key, value in self.action.items():
            context[f"action_{key}"] = value
        
        # Add metadata
        context.update(self.metadata)
        
        return context


class VerificationHook(ABC):
    """
    Abstract base class for verification hooks.
    
    Hooks allow custom verification logic to be injected into
    the verification process.
    """
    
    @abstractmethod
    async def before_verify(
        self,
        commitment: RuntimeCommitment,
        context: VerificationContext,
    ) -> None:
        """Called before verification starts."""
        ...
    
    @abstractmethod
    async def after_verify(
        self,
        commitment: RuntimeCommitment,
        context: VerificationContext,
        result: CommitmentResult,
    ) -> None:
        """Called after verification completes."""
        ...


class LoggingVerificationHook(VerificationHook):
    """Hook that logs verification events."""
    
    def __init__(self, logger: Any = None) -> None:
        self.logger = logger
    
    async def before_verify(
        self,
        commitment: RuntimeCommitment,
        context: VerificationContext,
    ) -> None:
        """Log verification start."""
        if self.logger:
            await self.logger.info(
                f"Verifying commitment {commitment.id}",
                data={
                    "commitment_id": commitment.id,
                    "action_type": commitment.action_type,
                },
            )
    
    async def after_verify(
        self,
        commitment: RuntimeCommitment,
        context: VerificationContext,
        result: CommitmentResult,
    ) -> None:
        """Log verification result."""
        if self.logger:
            await self.logger.info(
                f"Verification complete: {result.status.value}",
                data={
                    "commitment_id": commitment.id,
                    "success": result.success,
                    "failure_mode": result.failure_mode,
                },
            )


class ExpressionEvaluator:
    """
    Simple expression evaluator for commitment conditions.
    
    Supports basic comparison and logical operators.
    """
    
    # Safe operators for evaluation
    SAFE_OPERATORS = {
        "==", "!=", "<", ">", "<=", ">=",
        "and", "or", "not", "in", "is",
        "+", "-", "*", "/", "%",
        "(", ")", "[", "]",
        "true", "false", "True", "False",
        "None", "none",
    }
    
    def __init__(self) -> None:
        self._cache: dict[str, Any] = {}
    
    def evaluate(self, expression: str, context: dict[str, Any]) -> bool:
        """
        Evaluate an expression in the given context.
        
        Args:
            expression: The expression to evaluate
            context: Variable bindings for evaluation
            
        Returns:
            Boolean result of the expression
            
        Raises:
            ValueError: If the expression is invalid or unsafe
        """
        if not expression or expression.strip().lower() in ("true", "1"):
            return True
        if expression.strip().lower() in ("false", "0"):
            return False
        
        # Validate expression safety
        self._validate_expression(expression)
        
        try:
            # Create safe evaluation context
            safe_context = {
                "true": True,
                "false": False,
                "True": True,
                "False": False,
                "None": None,
                "none": None,
                **context,
            }
            
            # Evaluate the expression
            result = eval(expression, {"__builtins__": {}}, safe_context)
            return bool(result)
            
        except Exception as e:
            raise ValueError(f"Failed to evaluate expression '{expression}': {e}")
    
    def _validate_expression(self, expression: str) -> None:
        """
        Validate that an expression is safe to evaluate.
        
        Args:
            expression: The expression to validate
            
        Raises:
            ValueError: If the expression contains unsafe constructs
        """
        # Check for dangerous patterns
        dangerous_patterns = [
            r"__\w+__",  # Dunder methods
            r"\bimport\b",  # Import statements
            r"\bexec\b",  # Exec function
            r"\beval\b",  # Nested eval
            r"\bcompile\b",  # Compile function
            r"\bopen\b",  # File operations
            r"\bos\.",  # OS module
            r"\bsys\.",  # Sys module
            r"\bsubprocess\b",  # Subprocess
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, expression, re.IGNORECASE):
                raise ValueError(f"Unsafe expression pattern detected: {pattern}")


class RuntimeVerifier:
    """
    Verifier for runtime commitments.
    
    The RuntimeVerifier checks whether commitments have been fulfilled
    by evaluating their conditions against the verification context.
    
    Example:
        >>> verifier = RuntimeVerifier()
        >>> context = VerificationContext(
        ...     pre_state={"counter": 0},
        ...     post_state={"counter": 5, "tool_success": True},
        ...     action={"type": "increment"},
        ... )
        >>> result = await verifier.verify(commitment, context)
        >>> print(result.success)
        True
    """
    
    def __init__(
        self,
        evaluator: ExpressionEvaluator | None = None,
        hooks: list[VerificationHook] | None = None,
    ) -> None:
        """
        Initialize the verifier.
        
        Args:
            evaluator: Expression evaluator (creates default if None)
            hooks: Verification hooks
        """
        self.evaluator = evaluator or ExpressionEvaluator()
        self.hooks = hooks or []
        
        # Try to import GCL verification engine
        self._gcl_engine = None
        try:
            from gcl.core.verification import VerificationEngine
            self._gcl_engine = VerificationEngine()
        except ImportError:
            pass
    
    def add_hook(self, hook: VerificationHook) -> None:
        """Add a verification hook."""
        self.hooks.append(hook)
    
    def remove_hook(self, hook: VerificationHook) -> None:
        """Remove a verification hook."""
        if hook in self.hooks:
            self.hooks.remove(hook)
    
    async def verify(
        self,
        commitment: RuntimeCommitment,
        context: VerificationContext,
    ) -> CommitmentResult:
        """
        Verify a commitment.
        
        Args:
            commitment: The commitment to verify
            context: The verification context
            
        Returns:
            CommitmentResult indicating success or failure
        """
        start_time = time.perf_counter()
        
        # Run before hooks
        for hook in self.hooks:
            try:
                await hook.before_verify(commitment, context)
            except Exception:
                pass
        
        try:
            result = await self._do_verify(commitment, context, start_time)
        except Exception as e:
            result = CommitmentResult(
                commitment_id=commitment.id,
                status=CommitmentStatus.FAILED,
                success=False,
                failure_mode="verification_error",
                details={"error": str(e)},
                duration_ms=self._elapsed_ms(start_time),
            )
        
        # Run after hooks
        for hook in self.hooks:
            try:
                await hook.after_verify(commitment, context, result)
            except Exception:
                pass
        
        return result
    
    async def _do_verify(
        self,
        commitment: RuntimeCommitment,
        context: VerificationContext,
        start_time: float,
    ) -> CommitmentResult:
        """
        Perform the actual verification.
        
        Args:
            commitment: The commitment to verify
            context: The verification context
            start_time: When verification started
            
        Returns:
            CommitmentResult
        """
        # Check if commitment is expired
        if commitment.is_expired():
            return CommitmentResult(
                commitment_id=commitment.id,
                status=CommitmentStatus.EXPIRED,
                success=False,
                failure_mode="expired",
                details={"reason": "Commitment has expired"},
                duration_ms=self._elapsed_ms(start_time),
            )
        
        # Build evaluation context
        eval_context = context.to_eval_context()
        
        # Check failure conditions first (failure-first approach)
        for failure_name, failure_condition in commitment.failure_conditions.items():
            try:
                if self.evaluator.evaluate(failure_condition, eval_context):
                    return CommitmentResult(
                        commitment_id=commitment.id,
                        status=CommitmentStatus.FAILED,
                        success=False,
                        failure_mode=failure_name,
                        details={
                            "failure_condition": failure_condition,
                            "context": eval_context,
                        },
                        duration_ms=self._elapsed_ms(start_time),
                    )
            except ValueError:
                # If we can't evaluate a failure condition, continue
                pass
        
        # Check success condition
        try:
            if self.evaluator.evaluate(commitment.success_condition, eval_context):
                return CommitmentResult(
                    commitment_id=commitment.id,
                    status=CommitmentStatus.FULFILLED,
                    success=True,
                    details={
                        "success_condition": commitment.success_condition,
                    },
                    duration_ms=self._elapsed_ms(start_time),
                )
            else:
                return CommitmentResult(
                    commitment_id=commitment.id,
                    status=CommitmentStatus.FAILED,
                    success=False,
                    failure_mode="success_condition_not_met",
                    details={
                        "success_condition": commitment.success_condition,
                        "context": eval_context,
                    },
                    duration_ms=self._elapsed_ms(start_time),
                )
        except ValueError as e:
            return CommitmentResult(
                commitment_id=commitment.id,
                status=CommitmentStatus.FAILED,
                success=False,
                failure_mode="evaluation_error",
                details={
                    "error": str(e),
                    "success_condition": commitment.success_condition,
                },
                duration_ms=self._elapsed_ms(start_time),
            )
    
    async def verify_batch(
        self,
        commitments: list[RuntimeCommitment],
        context: VerificationContext,
    ) -> list[CommitmentResult]:
        """
        Verify multiple commitments.
        
        Args:
            commitments: List of commitments to verify
            context: The verification context
            
        Returns:
            List of CommitmentResults
        """
        results = []
        for commitment in commitments:
            result = await self.verify(commitment, context)
            results.append(result)
        return results
    
    def _elapsed_ms(self, start_time: float) -> float:
        """Calculate elapsed time in milliseconds."""
        return (time.perf_counter() - start_time) * 1000


class ToolVerifier:
    """
    Specialized verifier for tool execution commitments.
    
    Provides convenience methods for verifying tool-related commitments.
    """
    
    def __init__(self, verifier: RuntimeVerifier | None = None) -> None:
        """
        Initialize the tool verifier.
        
        Args:
            verifier: Base verifier (creates default if None)
        """
        self.verifier = verifier or RuntimeVerifier()
    
    async def verify_tool_execution(
        self,
        commitment: RuntimeCommitment,
        tool_name: str,
        success: bool,
        duration_seconds: float,
        output: Any = None,
        error: str | None = None,
    ) -> CommitmentResult:
        """
        Verify a tool execution commitment.
        
        Args:
            commitment: The commitment to verify
            tool_name: Name of the executed tool
            success: Whether the tool succeeded
            duration_seconds: Execution duration
            output: Tool output (if any)
            error: Error message (if any)
            
        Returns:
            CommitmentResult
        """
        context = VerificationContext(
            post_state={
                "tool_name": tool_name,
                "tool_success": success,
                "tool_error": error is not None,
                "duration_seconds": duration_seconds,
                "output": output,
            },
            action={
                "type": "tool_execution",
                "tool_name": tool_name,
            },
            metadata={
                "error": error,
            },
        )
        
        return await self.verifier.verify(commitment, context)
    
    async def verify_response(
        self,
        commitment: RuntimeCommitment,
        response_quality: float,
        response_time_seconds: float,
        response_content: str | None = None,
    ) -> CommitmentResult:
        """
        Verify a response quality commitment.
        
        Args:
            commitment: The commitment to verify
            response_quality: Quality score (0.0 to 1.0)
            response_time_seconds: Response generation time
            response_content: The response content
            
        Returns:
            CommitmentResult
        """
        context = VerificationContext(
            post_state={
                "response_quality": response_quality,
                "response_time_seconds": response_time_seconds,
                "response_content": response_content,
            },
            action={
                "type": "response",
            },
        )
        
        return await self.verifier.verify(commitment, context)
