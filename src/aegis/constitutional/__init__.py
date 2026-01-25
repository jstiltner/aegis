"""
Constitutional AI policy rules for the agent runtime.

This package provides:
- Constitutional principles definition
- Policy evaluation and enforcement
- Self-critique and revision mechanisms
- Integration with tool gateway and commitments
"""

from __future__ import annotations

from .principles import (
    Principle,
    PrincipleCategory,
    PrincipleRegistry,
    # Built-in principles
    HELPFULNESS,
    HARMLESSNESS,
    HONESTY,
    TRANSPARENCY,
    PRIVACY,
    FAIRNESS,
    get_default_principles,
)
from .evaluator import (
    PolicyEvaluator,
    PolicyDecision,
    PolicyViolation,
    EvaluationResult,
)
from .critique import (
    CritiqueEngine,
    Critique,
    Revision,
    CritiqueResult,
)

__all__ = [
    # Principles
    "Principle",
    "PrincipleCategory",
    "PrincipleRegistry",
    "HELPFULNESS",
    "HARMLESSNESS",
    "HONESTY",
    "TRANSPARENCY",
    "PRIVACY",
    "FAIRNESS",
    "get_default_principles",
    # Evaluator
    "PolicyEvaluator",
    "PolicyDecision",
    "PolicyViolation",
    "EvaluationResult",
    # Critique
    "CritiqueEngine",
    "Critique",
    "Revision",
    "CritiqueResult",
]
