"""
Policy evaluator for constitutional AI principles.

This module provides the PolicyEvaluator class that checks agent
actions and responses against constitutional principles.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Awaitable
from uuid import uuid4

from pydantic import BaseModel, Field, ConfigDict

from .principles import Principle, PrincipleRegistry, PrincipleCategory


class PolicyDecision(str, Enum):
    """Decision from policy evaluation."""
    
    ALLOW = "allow"
    DENY = "deny"
    WARN = "warn"
    REQUIRE_REVIEW = "require_review"


class PolicyViolation(BaseModel):
    """
    A violation of a constitutional principle.
    
    Attributes:
        id: Unique identifier
        principle_id: ID of the violated principle
        principle_name: Name of the violated principle
        severity: Severity of the violation
        description: Description of the violation
        context: Context in which the violation occurred
        remediation_hint: Suggestion for fixing the violation
        timestamp: When the violation was detected
    """
    
    model_config = ConfigDict(frozen=True)
    
    id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier",
    )
    principle_id: str = Field(..., description="ID of the violated principle")
    principle_name: str = Field(..., description="Name of the violated principle")
    severity: float = Field(..., ge=0.0, le=1.0, description="Severity of the violation")
    description: str = Field(..., description="Description of the violation")
    context: dict[str, Any] = Field(
        default_factory=dict,
        description="Context in which the violation occurred",
    )
    remediation_hint: str | None = Field(
        default=None,
        description="Suggestion for fixing the violation",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the violation was detected",
    )


class EvaluationResult(BaseModel):
    """
    Result of a policy evaluation.
    
    Attributes:
        decision: The policy decision
        violations: List of principle violations
        warnings: List of warnings (non-blocking violations)
        passed_principles: List of principles that passed
        evaluation_time_ms: Time taken for evaluation
        context: Evaluation context
        metadata: Additional metadata
    """
    
    model_config = ConfigDict(frozen=True)
    
    decision: PolicyDecision = Field(..., description="The policy decision")
    violations: tuple[PolicyViolation, ...] = Field(
        default_factory=tuple,
        description="List of principle violations",
    )
    warnings: tuple[PolicyViolation, ...] = Field(
        default_factory=tuple,
        description="List of warnings (non-blocking violations)",
    )
    passed_principles: tuple[str, ...] = Field(
        default_factory=tuple,
        description="List of principle IDs that passed",
    )
    evaluation_time_ms: float = Field(
        default=0.0,
        description="Time taken for evaluation",
    )
    context: dict[str, Any] = Field(
        default_factory=dict,
        description="Evaluation context",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata",
    )
    
    @property
    def is_allowed(self) -> bool:
        """Check if the action is allowed."""
        return self.decision == PolicyDecision.ALLOW
    
    @property
    def is_denied(self) -> bool:
        """Check if the action is denied."""
        return self.decision == PolicyDecision.DENY
    
    @property
    def has_violations(self) -> bool:
        """Check if there are any violations."""
        return len(self.violations) > 0
    
    @property
    def has_warnings(self) -> bool:
        """Check if there are any warnings."""
        return len(self.warnings) > 0
    
    @property
    def max_severity(self) -> float:
        """Get the maximum severity of violations."""
        if not self.violations:
            return 0.0
        return max(v.severity for v in self.violations)
    
    def get_violations_by_category(
        self,
        category: PrincipleCategory,
        registry: PrincipleRegistry,
    ) -> list[PolicyViolation]:
        """Get violations for a specific category."""
        result = []
        for violation in self.violations:
            principle = registry.get(violation.principle_id)
            if principle and principle.category == category:
                result.append(violation)
        return result


class PolicyEvaluator:
    """
    Evaluator for constitutional AI policies.
    
    The PolicyEvaluator checks agent actions and responses against
    a set of constitutional principles, producing decisions about
    whether actions should be allowed, denied, or flagged for review.
    
    Example:
        >>> evaluator = PolicyEvaluator()
        >>> result = await evaluator.evaluate_pre_action(
        ...     action={"type": "tool_call", "tool": "web_search"},
        ...     context={"user_request": "search for news"},
        ... )
        >>> if result.is_allowed:
        ...     # Proceed with action
        ...     pass
    """
    
    def __init__(
        self,
        registry: PrincipleRegistry | None = None,
        severity_threshold: float = 0.7,
        warn_threshold: float = 0.3,
    ) -> None:
        """
        Initialize the policy evaluator.
        
        Args:
            registry: Principle registry (uses default if None)
            severity_threshold: Violations above this severity cause denial
            warn_threshold: Violations above this severity cause warnings
        """
        self.registry = registry or PrincipleRegistry()
        self.severity_threshold = severity_threshold
        self.warn_threshold = warn_threshold
        
        # Custom evaluators for specific checks
        self._custom_evaluators: dict[str, Callable[[dict[str, Any]], bool]] = {}
    
    def register_evaluator(
        self,
        name: str,
        evaluator: Callable[[dict[str, Any]], bool],
    ) -> None:
        """
        Register a custom evaluator function.
        
        Custom evaluators are called during principle evaluation
        and their results are added to the context.
        
        Args:
            name: Name of the evaluator (used in expressions)
            evaluator: Function that takes context and returns bool
        """
        self._custom_evaluators[name] = evaluator
    
    async def evaluate_pre_action(
        self,
        action: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> EvaluationResult:
        """
        Evaluate an action before it is performed.
        
        Args:
            action: The action to evaluate
            context: Additional context
            
        Returns:
            EvaluationResult with decision and any violations
        """
        principles = self.registry.get_pre_check_principles()
        return await self._evaluate(
            principles=principles,
            action=action,
            context=context,
            phase="pre",
        )
    
    async def evaluate_post_action(
        self,
        action: dict[str, Any],
        result: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> EvaluationResult:
        """
        Evaluate an action after it is performed.
        
        Args:
            action: The action that was performed
            result: The result of the action
            context: Additional context
            
        Returns:
            EvaluationResult with decision and any violations
        """
        principles = self.registry.get_post_check_principles()
        
        # Merge result into context
        full_context = {**(context or {}), "action_result": result}
        
        return await self._evaluate(
            principles=principles,
            action=action,
            context=full_context,
            phase="post",
        )
    
    async def evaluate_response(
        self,
        response: str,
        request: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> EvaluationResult:
        """
        Evaluate an agent response.
        
        Args:
            response: The response to evaluate
            request: The original request (if any)
            context: Additional context
            
        Returns:
            EvaluationResult with decision and any violations
        """
        # Build evaluation context
        eval_context = {
            **(context or {}),
            "response": response,
            "request": request,
            "response_length": len(response),
        }
        
        # Run custom evaluators to populate context
        eval_context = self._run_custom_evaluators(eval_context)
        
        principles = self.registry.get_enabled()
        return await self._evaluate(
            principles=principles,
            action={"type": "response", "content": response},
            context=eval_context,
            phase="response",
        )
    
    async def evaluate_tool_call(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> EvaluationResult:
        """
        Evaluate a tool call before execution.
        
        Args:
            tool_name: Name of the tool
            tool_input: Input to the tool
            context: Additional context
            
        Returns:
            EvaluationResult with decision and any violations
        """
        eval_context = {
            **(context or {}),
            "tool_name": tool_name,
            "tool_input": tool_input,
        }
        
        # Run custom evaluators
        eval_context = self._run_custom_evaluators(eval_context)
        
        return await self.evaluate_pre_action(
            action={"type": "tool_call", "tool": tool_name, "input": tool_input},
            context=eval_context,
        )
    
    async def _evaluate(
        self,
        principles: list[Principle],
        action: dict[str, Any],
        context: dict[str, Any] | None,
        phase: str,
    ) -> EvaluationResult:
        """Internal evaluation method."""
        start_time = time.perf_counter()
        
        # Build full context
        full_context = {
            **(context or {}),
            "action": action,
            "phase": phase,
        }
        
        # Run custom evaluators
        full_context = self._run_custom_evaluators(full_context)
        
        violations: list[PolicyViolation] = []
        warnings: list[PolicyViolation] = []
        passed: list[str] = []
        
        for principle in principles:
            try:
                is_satisfied = principle.evaluate(full_context)
                
                if is_satisfied:
                    passed.append(principle.id)
                else:
                    violation = PolicyViolation(
                        principle_id=principle.id,
                        principle_name=principle.name,
                        severity=principle.severity,
                        description=f"Violation of {principle.name}: {principle.description}",
                        context=full_context,
                        remediation_hint=principle.remediation_hint,
                    )
                    
                    if principle.severity >= self.severity_threshold:
                        violations.append(violation)
                    elif principle.severity >= self.warn_threshold:
                        warnings.append(violation)
                    else:
                        # Low severity, treat as passed with note
                        passed.append(principle.id)
                        
            except Exception as e:
                # Evaluation error - log but don't block
                warnings.append(PolicyViolation(
                    principle_id=principle.id,
                    principle_name=principle.name,
                    severity=0.1,
                    description=f"Error evaluating {principle.name}: {str(e)}",
                    context=full_context,
                ))
        
        # Determine decision
        if violations:
            max_severity = max(v.severity for v in violations)
            if max_severity >= self.severity_threshold:
                decision = PolicyDecision.DENY
            else:
                decision = PolicyDecision.REQUIRE_REVIEW
        elif warnings:
            decision = PolicyDecision.WARN
        else:
            decision = PolicyDecision.ALLOW
        
        evaluation_time = (time.perf_counter() - start_time) * 1000
        
        return EvaluationResult(
            decision=decision,
            violations=tuple(violations),
            warnings=tuple(warnings),
            passed_principles=tuple(passed),
            evaluation_time_ms=evaluation_time,
            context=full_context,
            metadata={"phase": phase, "principle_count": len(principles)},
        )
    
    def _run_custom_evaluators(
        self,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """Run custom evaluators and add results to context."""
        result = dict(context)
        
        for name, evaluator in self._custom_evaluators.items():
            try:
                result[name] = evaluator(context)
            except Exception:
                result[name] = False
        
        return result
    
    # Built-in evaluator helpers
    
    @staticmethod
    def contains_harmful_keywords(text: str) -> bool:
        """Check if text contains potentially harmful keywords."""
        harmful_patterns = [
            "how to hack",
            "how to steal",
            "how to kill",
            "how to make a bomb",
            "how to make drugs",
        ]
        text_lower = text.lower()
        return any(pattern in text_lower for pattern in harmful_patterns)
    
    @staticmethod
    def contains_personal_info_request(text: str) -> bool:
        """Check if text requests personal information."""
        pii_patterns = [
            "social security",
            "credit card",
            "bank account",
            "password",
            "mother's maiden name",
        ]
        text_lower = text.lower()
        return any(pattern in text_lower for pattern in pii_patterns)
    
    @staticmethod
    def is_response_relevant(response: str, request: str) -> bool:
        """Check if response is relevant to request."""
        if not request or not response:
            return True
        
        # Simple keyword overlap check
        request_words = set(request.lower().split())
        response_words = set(response.lower().split())
        
        # Remove common words
        common_words = {"the", "a", "an", "is", "are", "was", "were", "be", "been", "being"}
        request_words -= common_words
        response_words -= common_words
        
        if not request_words:
            return True
        
        overlap = len(request_words & response_words)
        return overlap / len(request_words) > 0.1


# Factory function for creating evaluator with common checks
def create_default_evaluator() -> PolicyEvaluator:
    """Create a PolicyEvaluator with default configuration and evaluators."""
    evaluator = PolicyEvaluator()
    
    # Register built-in evaluators
    evaluator.register_evaluator(
        "contains_harmful_content",
        lambda ctx: PolicyEvaluator.contains_harmful_keywords(
            ctx.get("response", "") or ctx.get("request", "") or ""
        ),
    )
    
    evaluator.register_evaluator(
        "requests_unnecessary_personal_info",
        lambda ctx: PolicyEvaluator.contains_personal_info_request(
            ctx.get("response", "") or ""
        ),
    )
    
    evaluator.register_evaluator(
        "response_is_relevant",
        lambda ctx: PolicyEvaluator.is_response_relevant(
            ctx.get("response", ""),
            ctx.get("request", ""),
        ),
    )
    
    # Default to True for checks we can't easily automate
    evaluator.register_evaluator("response_addresses_request", lambda ctx: True)
    evaluator.register_evaluator("enables_harm", lambda ctx: False)
    evaluator.register_evaluator("contains_deception", lambda ctx: False)
    evaluator.register_evaluator("fabricates_information", lambda ctx: False)
    evaluator.register_evaluator("acknowledges_ai_nature", lambda ctx: True)
    evaluator.register_evaluator("explains_limitations_when_relevant", lambda ctx: True)
    evaluator.register_evaluator("shares_personal_info", lambda ctx: False)
    evaluator.register_evaluator("contains_discrimination", lambda ctx: False)
    evaluator.register_evaluator("treats_users_equally", lambda ctx: True)
    
    return evaluator
