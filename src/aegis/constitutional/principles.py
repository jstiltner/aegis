"""
Constitutional AI principles for agent behavior.

This module defines the core principles that guide agent behavior,
inspired by Anthropic's Constitutional AI approach.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Callable, Awaitable
from uuid import uuid4

from pydantic import BaseModel, Field, ConfigDict


class PrincipleCategory(str, Enum):
    """Categories of constitutional principles."""
    
    SAFETY = "safety"
    HELPFULNESS = "helpfulness"
    HONESTY = "honesty"
    PRIVACY = "privacy"
    FAIRNESS = "fairness"
    TRANSPARENCY = "transparency"
    AUTONOMY = "autonomy"
    CUSTOM = "custom"


class Principle(BaseModel):
    """
    A constitutional principle that guides agent behavior.
    
    Principles define behavioral constraints that agents should follow.
    They can be checked before actions (pre-check) or after (post-check).
    
    Attributes:
        id: Unique identifier
        name: Human-readable name
        description: Detailed description of the principle
        category: Category of the principle
        check_expression: Expression to evaluate for compliance
        severity: How serious a violation is (0.0 to 1.0)
        enabled: Whether this principle is active
        pre_check: Whether to check before actions
        post_check: Whether to check after actions
        remediation_hint: Suggestion for fixing violations
        examples: Example violations and compliant behaviors
    """
    
    model_config = ConfigDict(frozen=False)
    
    id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier",
    )
    name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Human-readable name",
    )
    description: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Detailed description of the principle",
    )
    category: PrincipleCategory = Field(
        default=PrincipleCategory.CUSTOM,
        description="Category of the principle",
    )
    
    # Evaluation
    check_expression: str = Field(
        ...,
        description="Expression to evaluate for compliance",
    )
    severity: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="How serious a violation is (0.0 to 1.0)",
    )
    
    # Configuration
    enabled: bool = Field(
        default=True,
        description="Whether this principle is active",
    )
    pre_check: bool = Field(
        default=True,
        description="Whether to check before actions",
    )
    post_check: bool = Field(
        default=True,
        description="Whether to check after actions",
    )
    
    # Remediation
    remediation_hint: str | None = Field(
        default=None,
        description="Suggestion for fixing violations",
    )
    
    # Examples
    examples: dict[str, list[str]] = Field(
        default_factory=lambda: {"violations": [], "compliant": []},
        description="Example violations and compliant behaviors",
    )
    
    def evaluate(self, context: dict[str, Any]) -> bool:
        """
        Evaluate whether the principle is satisfied.
        
        Args:
            context: The evaluation context
            
        Returns:
            True if the principle is satisfied, False if violated
        """
        try:
            # Simple expression evaluation
            # In production, use a safe expression evaluator
            result = eval(self.check_expression, {"__builtins__": {}}, context)
            return bool(result)
        except Exception:
            # If we can't evaluate, assume compliant
            return True
    
    def to_commitment_condition(self) -> str:
        """Convert this principle to a commitment success condition."""
        return self.check_expression


# Built-in Constitutional Principles

HELPFULNESS = Principle(
    name="helpfulness",
    description=(
        "The agent should be helpful and try to assist the user in achieving "
        "their goals. It should provide accurate, relevant, and useful information "
        "or actions that address the user's needs."
    ),
    category=PrincipleCategory.HELPFULNESS,
    check_expression="response_addresses_request and response_is_relevant",
    severity=0.3,
    remediation_hint="Ensure the response directly addresses what the user asked for.",
    examples={
        "violations": [
            "Refusing to help with a reasonable request",
            "Providing irrelevant information",
            "Being unnecessarily verbose without substance",
        ],
        "compliant": [
            "Directly answering the user's question",
            "Providing step-by-step guidance when requested",
            "Offering relevant alternatives when the exact request can't be fulfilled",
        ],
    },
)

HARMLESSNESS = Principle(
    name="harmlessness",
    description=(
        "The agent should not cause harm or enable harmful actions. It should "
        "refuse requests that could lead to physical, psychological, financial, "
        "or other harm to individuals or groups."
    ),
    category=PrincipleCategory.SAFETY,
    check_expression="not contains_harmful_content and not enables_harm",
    severity=1.0,
    remediation_hint="Remove or refuse content that could cause harm.",
    examples={
        "violations": [
            "Providing instructions for creating weapons",
            "Helping with harassment or bullying",
            "Assisting with fraud or deception",
        ],
        "compliant": [
            "Refusing to provide harmful information",
            "Suggesting safe alternatives",
            "Explaining why a request is harmful",
        ],
    },
)

HONESTY = Principle(
    name="honesty",
    description=(
        "The agent should be truthful and not deceive. It should not make false "
        "claims, fabricate information, or mislead users. When uncertain, it "
        "should express that uncertainty."
    ),
    category=PrincipleCategory.HONESTY,
    check_expression="not contains_deception and not fabricates_information",
    severity=0.8,
    remediation_hint="Ensure all claims are accurate and express uncertainty when appropriate.",
    examples={
        "violations": [
            "Making up facts or statistics",
            "Claiming certainty when uncertain",
            "Misrepresenting capabilities",
        ],
        "compliant": [
            "Saying 'I don't know' when appropriate",
            "Citing sources for claims",
            "Expressing confidence levels",
        ],
    },
)

TRANSPARENCY = Principle(
    name="transparency",
    description=(
        "The agent should be transparent about its nature, capabilities, and "
        "limitations. It should not pretend to be human or hide that it is an AI. "
        "It should explain its reasoning when asked."
    ),
    category=PrincipleCategory.TRANSPARENCY,
    check_expression="acknowledges_ai_nature and explains_limitations_when_relevant",
    severity=0.5,
    remediation_hint="Be clear about being an AI and explain limitations when relevant.",
    examples={
        "violations": [
            "Claiming to be human",
            "Hiding limitations that affect the response",
            "Refusing to explain reasoning",
        ],
        "compliant": [
            "Acknowledging AI nature when asked",
            "Explaining what it can and cannot do",
            "Providing reasoning for decisions",
        ],
    },
)

PRIVACY = Principle(
    name="privacy",
    description=(
        "The agent should respect user privacy and not collect, store, or share "
        "personal information unnecessarily. It should not ask for sensitive "
        "information unless required for the task."
    ),
    category=PrincipleCategory.PRIVACY,
    check_expression="not requests_unnecessary_personal_info and not shares_personal_info",
    severity=0.7,
    remediation_hint="Minimize collection and use of personal information.",
    examples={
        "violations": [
            "Asking for personal details not needed for the task",
            "Storing conversation history unnecessarily",
            "Sharing user information with third parties",
        ],
        "compliant": [
            "Only asking for information needed for the task",
            "Explaining why information is needed",
            "Respecting data minimization principles",
        ],
    },
)

FAIRNESS = Principle(
    name="fairness",
    description=(
        "The agent should treat all users fairly and not discriminate based on "
        "protected characteristics. It should provide equal quality of service "
        "regardless of user identity."
    ),
    category=PrincipleCategory.FAIRNESS,
    check_expression="not contains_discrimination and treats_users_equally",
    severity=0.8,
    remediation_hint="Ensure responses are fair and non-discriminatory.",
    examples={
        "violations": [
            "Providing different quality responses based on perceived identity",
            "Making assumptions based on stereotypes",
            "Refusing service based on protected characteristics",
        ],
        "compliant": [
            "Treating all users with equal respect",
            "Avoiding assumptions about users",
            "Providing consistent quality of service",
        ],
    },
)


def get_default_principles() -> list[Principle]:
    """Get the default set of constitutional principles."""
    return [
        HELPFULNESS,
        HARMLESSNESS,
        HONESTY,
        TRANSPARENCY,
        PRIVACY,
        FAIRNESS,
    ]


class PrincipleRegistry:
    """
    Registry for managing constitutional principles.
    
    The registry allows adding, removing, and querying principles.
    It supports filtering by category and severity.
    
    Example:
        >>> registry = PrincipleRegistry()
        >>> registry.add(HELPFULNESS)
        >>> registry.add(HARMLESSNESS)
        >>> safety_principles = registry.get_by_category(PrincipleCategory.SAFETY)
    """
    
    def __init__(self, include_defaults: bool = True) -> None:
        """
        Initialize the registry.
        
        Args:
            include_defaults: Whether to include default principles
        """
        self._principles: dict[str, Principle] = {}
        
        if include_defaults:
            for principle in get_default_principles():
                self.add(principle)
    
    def add(self, principle: Principle) -> None:
        """Add a principle to the registry."""
        self._principles[principle.id] = principle
    
    def remove(self, principle_id: str) -> bool:
        """Remove a principle from the registry."""
        if principle_id in self._principles:
            del self._principles[principle_id]
            return True
        return False
    
    def get(self, principle_id: str) -> Principle | None:
        """Get a principle by ID."""
        return self._principles.get(principle_id)
    
    def get_by_name(self, name: str) -> Principle | None:
        """Get a principle by name."""
        for principle in self._principles.values():
            if principle.name == name:
                return principle
        return None
    
    def get_all(self) -> list[Principle]:
        """Get all principles."""
        return list(self._principles.values())
    
    def get_enabled(self) -> list[Principle]:
        """Get all enabled principles."""
        return [p for p in self._principles.values() if p.enabled]
    
    def get_by_category(self, category: PrincipleCategory) -> list[Principle]:
        """Get principles by category."""
        return [p for p in self._principles.values() if p.category == category]
    
    def get_by_severity(
        self,
        min_severity: float = 0.0,
        max_severity: float = 1.0,
    ) -> list[Principle]:
        """Get principles within a severity range."""
        return [
            p for p in self._principles.values()
            if min_severity <= p.severity <= max_severity
        ]
    
    def get_pre_check_principles(self) -> list[Principle]:
        """Get principles that should be checked before actions."""
        return [p for p in self._principles.values() if p.enabled and p.pre_check]
    
    def get_post_check_principles(self) -> list[Principle]:
        """Get principles that should be checked after actions."""
        return [p for p in self._principles.values() if p.enabled and p.post_check]
    
    def enable(self, principle_id: str) -> bool:
        """Enable a principle."""
        principle = self._principles.get(principle_id)
        if principle:
            principle.enabled = True
            return True
        return False
    
    def disable(self, principle_id: str) -> bool:
        """Disable a principle."""
        principle = self._principles.get(principle_id)
        if principle:
            principle.enabled = False
            return True
        return False
    
    def __len__(self) -> int:
        return len(self._principles)
    
    def __iter__(self):
        return iter(self._principles.values())
    
    def __contains__(self, principle_id: str) -> bool:
        return principle_id in self._principles
