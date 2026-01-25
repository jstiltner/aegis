"""
Policy engine for tool access control.

This module provides:
- Policy definitions with rules
- Policy evaluation engine
- Policy decisions and violations
"""

from __future__ import annotations

import fnmatch
import re
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable
from uuid import uuid4

from pydantic import BaseModel, Field, ConfigDict


class PolicyDecisionType(str, Enum):
    """Type of policy decision."""
    
    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_APPROVAL = "require_approval"


class PolicyDecision(BaseModel):
    """
    Result of a policy evaluation.
    
    Contains the decision and any constraints or requirements.
    """
    
    model_config = ConfigDict(frozen=True)
    
    decision: PolicyDecisionType = Field(
        ...,
        description="The policy decision",
    )
    policy_name: str = Field(
        ...,
        description="Name of the policy that made this decision",
    )
    rule_name: str | None = Field(
        default=None,
        description="Name of the specific rule that matched",
    )
    reason: str = Field(
        default="",
        description="Explanation for the decision",
    )
    
    # Constraints for allowed actions
    constraints: dict[str, Any] = Field(
        default_factory=dict,
        description="Constraints to apply if allowed",
    )
    
    # Requirements for approval
    approval_type: str | None = Field(
        default=None,
        description="Type of approval required (human, automated, etc.)",
    )
    
    @property
    def is_allowed(self) -> bool:
        """Check if the decision allows the action."""
        return self.decision == PolicyDecisionType.ALLOW
    
    @property
    def is_denied(self) -> bool:
        """Check if the decision denies the action."""
        return self.decision == PolicyDecisionType.DENY
    
    @property
    def requires_approval(self) -> bool:
        """Check if the decision requires approval."""
        return self.decision == PolicyDecisionType.REQUIRE_APPROVAL
    
    @classmethod
    def allow(
        cls,
        policy_name: str,
        rule_name: str | None = None,
        reason: str = "",
        constraints: dict[str, Any] | None = None,
    ) -> PolicyDecision:
        """Create an allow decision."""
        return cls(
            decision=PolicyDecisionType.ALLOW,
            policy_name=policy_name,
            rule_name=rule_name,
            reason=reason,
            constraints=constraints or {},
        )
    
    @classmethod
    def deny(
        cls,
        policy_name: str,
        rule_name: str | None = None,
        reason: str = "",
    ) -> PolicyDecision:
        """Create a deny decision."""
        return cls(
            decision=PolicyDecisionType.DENY,
            policy_name=policy_name,
            rule_name=rule_name,
            reason=reason,
        )
    
    @classmethod
    def require_approval(
        cls,
        policy_name: str,
        approval_type: str = "human",
        rule_name: str | None = None,
        reason: str = "",
    ) -> PolicyDecision:
        """Create a require approval decision."""
        return cls(
            decision=PolicyDecisionType.REQUIRE_APPROVAL,
            policy_name=policy_name,
            rule_name=rule_name,
            reason=reason,
            approval_type=approval_type,
        )


class PolicyViolation(BaseModel):
    """
    Record of a policy violation.
    
    Created when a tool invocation is denied by policy.
    """
    
    model_config = ConfigDict(frozen=True)
    
    violation_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for this violation",
    )
    
    # Context
    agent_id: str = Field(
        ...,
        description="ID of the agent that attempted the action",
    )
    session_id: str = Field(
        ...,
        description="ID of the session",
    )
    tool_name: str = Field(
        ...,
        description="Name of the tool that was denied",
    )
    arguments: dict[str, Any] = Field(
        default_factory=dict,
        description="Arguments that were provided",
    )
    
    # Decision
    decision: PolicyDecision = Field(
        ...,
        description="The policy decision that caused the violation",
    )
    
    # Timing
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the violation occurred",
    )
    
    # Metadata
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata",
    )


class RuleCondition(ABC):
    """Abstract base class for rule conditions."""
    
    @abstractmethod
    def evaluate(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        context: dict[str, Any],
    ) -> bool:
        """
        Evaluate the condition.
        
        Args:
            tool_name: Name of the tool being invoked
            arguments: Arguments to the tool
            context: Additional context (agent_id, session_id, etc.)
            
        Returns:
            True if the condition matches
        """
        ...


class ToolNameCondition(RuleCondition):
    """Condition that matches tool names."""
    
    def __init__(self, patterns: list[str]) -> None:
        """
        Initialize with tool name patterns.
        
        Args:
            patterns: List of glob patterns to match tool names
        """
        self.patterns = patterns
    
    def evaluate(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        context: dict[str, Any],
    ) -> bool:
        """Check if tool name matches any pattern."""
        return any(fnmatch.fnmatch(tool_name, p) for p in self.patterns)


class ArgumentCondition(RuleCondition):
    """Condition that matches argument values."""
    
    def __init__(
        self,
        argument_name: str,
        patterns: list[str] | None = None,
        values: list[Any] | None = None,
        regex: str | None = None,
    ) -> None:
        """
        Initialize with argument matching criteria.
        
        Args:
            argument_name: Name of the argument to check
            patterns: Glob patterns to match string values
            values: Exact values to match
            regex: Regex pattern to match string values
        """
        self.argument_name = argument_name
        self.patterns = patterns
        self.values = values
        self.regex = re.compile(regex) if regex else None
    
    def evaluate(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        context: dict[str, Any],
    ) -> bool:
        """Check if argument matches criteria."""
        if self.argument_name not in arguments:
            return False
        
        value = arguments[self.argument_name]
        
        # Check exact values
        if self.values is not None and value in self.values:
            return True
        
        # Check patterns (for strings)
        if self.patterns is not None and isinstance(value, str):
            if any(fnmatch.fnmatch(value, p) for p in self.patterns):
                return True
        
        # Check regex (for strings)
        if self.regex is not None and isinstance(value, str):
            if self.regex.match(value):
                return True
        
        return False


class ContextCondition(RuleCondition):
    """Condition that matches context values."""
    
    def __init__(
        self,
        context_key: str,
        values: list[Any] | None = None,
        min_value: float | None = None,
        max_value: float | None = None,
    ) -> None:
        """
        Initialize with context matching criteria.
        
        Args:
            context_key: Key in the context to check
            values: Exact values to match
            min_value: Minimum value (for numeric comparisons)
            max_value: Maximum value (for numeric comparisons)
        """
        self.context_key = context_key
        self.values = values
        self.min_value = min_value
        self.max_value = max_value
    
    def evaluate(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        context: dict[str, Any],
    ) -> bool:
        """Check if context matches criteria."""
        if self.context_key not in context:
            return False
        
        value = context[self.context_key]
        
        # Check exact values
        if self.values is not None and value in self.values:
            return True
        
        # Check numeric range
        if isinstance(value, (int, float)):
            if self.min_value is not None and value < self.min_value:
                return False
            if self.max_value is not None and value > self.max_value:
                return False
            if self.min_value is not None or self.max_value is not None:
                return True
        
        return False


class PolicyRule(BaseModel):
    """
    A single rule within a policy.
    
    Rules have conditions and an action (allow, deny, require_approval).
    """
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    name: str = Field(
        ...,
        description="Rule name",
    )
    description: str = Field(
        default="",
        description="Rule description",
    )
    
    # Matching
    tools: list[str] = Field(
        default_factory=lambda: ["*"],
        description="Tool name patterns to match",
    )
    
    # Action
    action: PolicyDecisionType = Field(
        ...,
        description="Action to take when rule matches",
    )
    
    # Conditions (stored as dicts for serialization)
    argument_conditions: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Conditions on argument values",
    )
    context_conditions: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Conditions on context values",
    )
    
    # Constraints (for allow rules)
    constraints: dict[str, Any] = Field(
        default_factory=dict,
        description="Constraints to apply if allowed",
    )
    
    # Approval settings (for require_approval rules)
    approval_type: str = Field(
        default="human",
        description="Type of approval required",
    )
    
    # Priority (higher = evaluated first)
    priority: int = Field(
        default=0,
        description="Rule priority",
    )
    
    # Enabled flag
    enabled: bool = Field(
        default=True,
        description="Whether the rule is enabled",
    )
    
    def matches_tool(self, tool_name: str) -> bool:
        """Check if the rule matches a tool name."""
        return any(fnmatch.fnmatch(tool_name, p) for p in self.tools)
    
    def matches_arguments(self, arguments: dict[str, Any]) -> bool:
        """Check if the rule matches the arguments."""
        for arg_name, conditions in self.argument_conditions.items():
            if arg_name not in arguments:
                # If argument is required by condition but not present
                if conditions.get("required", False):
                    return False
                continue
            
            value = arguments[arg_name]
            
            # Check allowed values
            if "values" in conditions:
                if value not in conditions["values"]:
                    return False
            
            # Check patterns
            if "patterns" in conditions and isinstance(value, str):
                if not any(fnmatch.fnmatch(value, p) for p in conditions["patterns"]):
                    return False
            
            # Check denied values
            if "deny_values" in conditions:
                if value in conditions["deny_values"]:
                    return False
            
            # Check denied patterns
            if "deny_patterns" in conditions and isinstance(value, str):
                if any(fnmatch.fnmatch(value, p) for p in conditions["deny_patterns"]):
                    return False
        
        return True
    
    def matches_context(self, context: dict[str, Any]) -> bool:
        """Check if the rule matches the context."""
        for ctx_key, conditions in self.context_conditions.items():
            if ctx_key not in context:
                if conditions.get("required", False):
                    return False
                continue
            
            value = context[ctx_key]
            
            # Check allowed values
            if "values" in conditions:
                if value not in conditions["values"]:
                    return False
            
            # Check numeric range
            if "min" in conditions and value < conditions["min"]:
                return False
            if "max" in conditions and value > conditions["max"]:
                return False
        
        return True
    
    def evaluate(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        context: dict[str, Any],
    ) -> PolicyDecision | None:
        """
        Evaluate the rule against a tool invocation.
        
        Returns:
            PolicyDecision if rule matches, None otherwise
        """
        if not self.enabled:
            return None
        
        if not self.matches_tool(tool_name):
            return None
        
        if not self.matches_arguments(arguments):
            return None
        
        if not self.matches_context(context):
            return None
        
        # Rule matches - return decision
        match self.action:
            case PolicyDecisionType.ALLOW:
                return PolicyDecision.allow(
                    policy_name="",  # Will be set by policy
                    rule_name=self.name,
                    reason=self.description,
                    constraints=self.constraints,
                )
            case PolicyDecisionType.DENY:
                return PolicyDecision.deny(
                    policy_name="",
                    rule_name=self.name,
                    reason=self.description,
                )
            case PolicyDecisionType.REQUIRE_APPROVAL:
                return PolicyDecision.require_approval(
                    policy_name="",
                    approval_type=self.approval_type,
                    rule_name=self.name,
                    reason=self.description,
                )


class Policy(BaseModel):
    """
    A policy containing multiple rules.
    
    Policies are evaluated in order of rule priority.
    """
    
    model_config = ConfigDict(frozen=True)
    
    name: str = Field(
        ...,
        description="Policy name",
    )
    description: str = Field(
        default="",
        description="Policy description",
    )
    
    # Rules
    rules: tuple[PolicyRule, ...] = Field(
        default_factory=tuple,
        description="Policy rules",
    )
    
    # Default action if no rules match
    default_action: PolicyDecisionType = Field(
        default=PolicyDecisionType.DENY,
        description="Default action if no rules match",
    )
    
    # Enabled flag
    enabled: bool = Field(
        default=True,
        description="Whether the policy is enabled",
    )
    
    # Priority (for ordering multiple policies)
    priority: int = Field(
        default=0,
        description="Policy priority",
    )
    
    def evaluate(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        context: dict[str, Any],
    ) -> PolicyDecision:
        """
        Evaluate the policy against a tool invocation.
        
        Rules are evaluated in priority order (highest first).
        First matching rule determines the decision.
        
        Returns:
            PolicyDecision from matching rule or default
        """
        if not self.enabled:
            return PolicyDecision.allow(
                policy_name=self.name,
                reason="Policy disabled",
            )
        
        # Sort rules by priority (highest first)
        sorted_rules = sorted(self.rules, key=lambda r: r.priority, reverse=True)
        
        for rule in sorted_rules:
            decision = rule.evaluate(tool_name, arguments, context)
            if decision is not None:
                # Update policy name in decision
                return PolicyDecision(
                    decision=decision.decision,
                    policy_name=self.name,
                    rule_name=decision.rule_name,
                    reason=decision.reason,
                    constraints=decision.constraints,
                    approval_type=decision.approval_type,
                )
        
        # No rule matched - use default
        match self.default_action:
            case PolicyDecisionType.ALLOW:
                return PolicyDecision.allow(
                    policy_name=self.name,
                    reason="No matching rule (default allow)",
                )
            case PolicyDecisionType.DENY:
                return PolicyDecision.deny(
                    policy_name=self.name,
                    reason="No matching rule (default deny)",
                )
            case PolicyDecisionType.REQUIRE_APPROVAL:
                return PolicyDecision.require_approval(
                    policy_name=self.name,
                    reason="No matching rule (default require approval)",
                )


class PolicyEngine:
    """
    Engine for evaluating multiple policies.
    
    The PolicyEngine:
    - Manages multiple policies
    - Evaluates them in priority order
    - Combines decisions (most restrictive wins)
    - Tracks violations
    """
    
    def __init__(self) -> None:
        self._policies: dict[str, Policy] = {}
        self._violations: list[PolicyViolation] = []
    
    @property
    def policies(self) -> list[Policy]:
        """Get all registered policies."""
        return list(self._policies.values())
    
    @property
    def violations(self) -> list[PolicyViolation]:
        """Get all recorded violations."""
        return self._violations.copy()
    
    def add_policy(self, policy: Policy) -> None:
        """Add a policy to the engine."""
        self._policies[policy.name] = policy
    
    def remove_policy(self, policy_name: str) -> bool:
        """Remove a policy from the engine."""
        if policy_name in self._policies:
            del self._policies[policy_name]
            return True
        return False
    
    def get_policy(self, policy_name: str) -> Policy | None:
        """Get a policy by name."""
        return self._policies.get(policy_name)
    
    def clear_policies(self) -> None:
        """Remove all policies."""
        self._policies.clear()
    
    def clear_violations(self) -> None:
        """Clear violation history."""
        self._violations.clear()
    
    def evaluate(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        context: dict[str, Any],
    ) -> PolicyDecision:
        """
        Evaluate all policies against a tool invocation.
        
        Policies are evaluated in priority order.
        The most restrictive decision wins:
        - DENY beats everything
        - REQUIRE_APPROVAL beats ALLOW
        - ALLOW is the least restrictive
        
        Returns:
            Combined PolicyDecision
        """
        if not self._policies:
            # No policies - allow by default
            return PolicyDecision.allow(
                policy_name="default",
                reason="No policies configured",
            )
        
        # Sort policies by priority (highest first)
        sorted_policies = sorted(
            self._policies.values(),
            key=lambda p: p.priority,
            reverse=True,
        )
        
        decisions: list[PolicyDecision] = []
        
        for policy in sorted_policies:
            decision = policy.evaluate(tool_name, arguments, context)
            decisions.append(decision)
            
            # Short-circuit on deny
            if decision.is_denied:
                return decision
        
        # Check for require_approval
        approval_decisions = [d for d in decisions if d.requires_approval]
        if approval_decisions:
            return approval_decisions[0]
        
        # All policies allow - combine constraints
        combined_constraints: dict[str, Any] = {}
        for decision in decisions:
            combined_constraints.update(decision.constraints)
        
        return PolicyDecision.allow(
            policy_name="combined",
            reason="All policies allow",
            constraints=combined_constraints,
        )
    
    def check_and_record(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        agent_id: str,
        session_id: str,
        context: dict[str, Any] | None = None,
    ) -> PolicyDecision:
        """
        Evaluate policies and record any violations.
        
        Args:
            tool_name: Name of the tool
            arguments: Tool arguments
            agent_id: ID of the agent
            session_id: ID of the session
            context: Additional context
            
        Returns:
            PolicyDecision
        """
        full_context = {
            "agent_id": agent_id,
            "session_id": session_id,
            **(context or {}),
        }
        
        decision = self.evaluate(tool_name, arguments, full_context)
        
        # Record violation if denied
        if decision.is_denied:
            violation = PolicyViolation(
                agent_id=agent_id,
                session_id=session_id,
                tool_name=tool_name,
                arguments=arguments,
                decision=decision,
            )
            self._violations.append(violation)
        
        return decision
    
    def get_violations_for_agent(self, agent_id: str) -> list[PolicyViolation]:
        """Get violations for a specific agent."""
        return [v for v in self._violations if v.agent_id == agent_id]
    
    def get_violations_for_session(self, session_id: str) -> list[PolicyViolation]:
        """Get violations for a specific session."""
        return [v for v in self._violations if v.session_id == session_id]


# Convenience functions for creating common policies


def create_allowlist_policy(
    name: str,
    allowed_tools: list[str],
    description: str = "",
) -> Policy:
    """
    Create a policy that only allows specific tools.
    
    Args:
        name: Policy name
        allowed_tools: List of allowed tool name patterns
        description: Policy description
        
    Returns:
        Policy that allows only the specified tools
    """
    return Policy(
        name=name,
        description=description or f"Allow only: {', '.join(allowed_tools)}",
        rules=(
            PolicyRule(
                name="allow_listed",
                description="Allow listed tools",
                tools=allowed_tools,
                action=PolicyDecisionType.ALLOW,
            ),
        ),
        default_action=PolicyDecisionType.DENY,
    )


def create_denylist_policy(
    name: str,
    denied_tools: list[str],
    description: str = "",
) -> Policy:
    """
    Create a policy that denies specific tools.
    
    Args:
        name: Policy name
        denied_tools: List of denied tool name patterns
        description: Policy description
        
    Returns:
        Policy that denies the specified tools
    """
    return Policy(
        name=name,
        description=description or f"Deny: {', '.join(denied_tools)}",
        rules=(
            PolicyRule(
                name="deny_listed",
                description="Deny listed tools",
                tools=denied_tools,
                action=PolicyDecisionType.DENY,
            ),
        ),
        default_action=PolicyDecisionType.ALLOW,
    )


def create_rate_limit_policy(
    name: str,
    tools: list[str],
    max_calls_per_minute: int,
    description: str = "",
) -> Policy:
    """
    Create a policy with rate limiting constraints.
    
    Note: Actual rate limiting enforcement is done by the gateway.
    This policy just adds the constraint.
    
    Args:
        name: Policy name
        tools: Tool patterns to rate limit
        max_calls_per_minute: Maximum calls per minute
        description: Policy description
        
    Returns:
        Policy with rate limit constraints
    """
    return Policy(
        name=name,
        description=description or f"Rate limit: {max_calls_per_minute}/min",
        rules=(
            PolicyRule(
                name="rate_limited",
                description=f"Rate limit to {max_calls_per_minute}/min",
                tools=tools,
                action=PolicyDecisionType.ALLOW,
                constraints={"rate_limit_per_minute": max_calls_per_minute},
            ),
        ),
        default_action=PolicyDecisionType.ALLOW,
    )


def create_approval_policy(
    name: str,
    tools: list[str],
    approval_type: str = "human",
    description: str = "",
) -> Policy:
    """
    Create a policy requiring approval for certain tools.
    
    Args:
        name: Policy name
        tools: Tool patterns requiring approval
        approval_type: Type of approval required
        description: Policy description
        
    Returns:
        Policy requiring approval
    """
    return Policy(
        name=name,
        description=description or f"Require {approval_type} approval",
        rules=(
            PolicyRule(
                name="require_approval",
                description=f"Require {approval_type} approval",
                tools=tools,
                action=PolicyDecisionType.REQUIRE_APPROVAL,
                approval_type=approval_type,
            ),
        ),
        default_action=PolicyDecisionType.ALLOW,
    )
