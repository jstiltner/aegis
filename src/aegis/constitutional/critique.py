"""
Self-critique and revision mechanism for Constitutional AI.

This module provides the CritiqueEngine that enables agents to
critique their own responses and revise them to better align
with constitutional principles.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Awaitable, TYPE_CHECKING
from uuid import uuid4

from pydantic import BaseModel, Field, ConfigDict

from .principles import Principle, PrincipleRegistry
from .evaluator import PolicyEvaluator, EvaluationResult, PolicyViolation

if TYPE_CHECKING:
    from ..llm import LLMClient


class CritiqueSeverity(str, Enum):
    """Severity of a critique."""
    
    MINOR = "minor"
    MODERATE = "moderate"
    MAJOR = "major"
    CRITICAL = "critical"


class Critique(BaseModel):
    """
    A critique of an agent response or action.
    
    Attributes:
        id: Unique identifier
        target: What is being critiqued (response, action, etc.)
        principle_id: ID of the principle being evaluated
        principle_name: Name of the principle
        severity: Severity of the issue
        issue: Description of the issue
        suggestion: Suggested improvement
        confidence: Confidence in the critique (0.0 to 1.0)
        timestamp: When the critique was generated
    """
    
    model_config = ConfigDict(frozen=True)
    
    id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier",
    )
    target: str = Field(..., description="What is being critiqued")
    principle_id: str | None = Field(
        default=None,
        description="ID of the principle being evaluated",
    )
    principle_name: str | None = Field(
        default=None,
        description="Name of the principle",
    )
    severity: CritiqueSeverity = Field(
        default=CritiqueSeverity.MODERATE,
        description="Severity of the issue",
    )
    issue: str = Field(..., description="Description of the issue")
    suggestion: str = Field(..., description="Suggested improvement")
    confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Confidence in the critique",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the critique was generated",
    )


class Revision(BaseModel):
    """
    A revision of an agent response or action.
    
    Attributes:
        id: Unique identifier
        original: The original content
        revised: The revised content
        critiques_addressed: IDs of critiques addressed by this revision
        changes_made: Description of changes made
        improvement_score: Estimated improvement (0.0 to 1.0)
        timestamp: When the revision was made
    """
    
    model_config = ConfigDict(frozen=True)
    
    id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier",
    )
    original: str = Field(..., description="The original content")
    revised: str = Field(..., description="The revised content")
    critiques_addressed: tuple[str, ...] = Field(
        default_factory=tuple,
        description="IDs of critiques addressed",
    )
    changes_made: str = Field(..., description="Description of changes made")
    improvement_score: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Estimated improvement",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the revision was made",
    )


class CritiqueResult(BaseModel):
    """
    Result of a critique operation.
    
    Attributes:
        critiques: List of critiques generated
        revisions: List of revisions made
        original_evaluation: Evaluation of original content
        final_evaluation: Evaluation of final content
        iterations: Number of critique-revise iterations
        total_time_ms: Total time taken
        success: Whether the revision process succeeded
    """
    
    model_config = ConfigDict(frozen=True)
    
    critiques: tuple[Critique, ...] = Field(
        default_factory=tuple,
        description="List of critiques generated",
    )
    revisions: tuple[Revision, ...] = Field(
        default_factory=tuple,
        description="List of revisions made",
    )
    original_evaluation: EvaluationResult | None = Field(
        default=None,
        description="Evaluation of original content",
    )
    final_evaluation: EvaluationResult | None = Field(
        default=None,
        description="Evaluation of final content",
    )
    iterations: int = Field(
        default=0,
        description="Number of critique-revise iterations",
    )
    total_time_ms: float = Field(
        default=0.0,
        description="Total time taken",
    )
    success: bool = Field(
        default=False,
        description="Whether the revision process succeeded",
    )
    
    @property
    def final_content(self) -> str | None:
        """Get the final revised content."""
        if self.revisions:
            return self.revisions[-1].revised
        return None
    
    @property
    def improvement(self) -> float:
        """Calculate overall improvement."""
        if not self.revisions:
            return 0.0
        return sum(r.improvement_score for r in self.revisions) / len(self.revisions)


class CritiqueEngine:
    """
    Engine for self-critique and revision of agent outputs.
    
    The CritiqueEngine implements the Constitutional AI pattern of:
    1. Generate initial response
    2. Critique the response against principles
    3. Revise the response based on critiques
    4. Repeat until satisfactory or max iterations reached
    
    Example:
        >>> engine = CritiqueEngine(evaluator=evaluator)
        >>> result = await engine.critique_and_revise(
        ...     content="Here's how to hack a computer...",
        ...     context={"request": "help with security"},
        ... )
        >>> if result.success:
        ...     print(result.final_content)
    """
    
    def __init__(
        self,
        evaluator: PolicyEvaluator | None = None,
        llm_client: "LLMClient | None" = None,
        max_iterations: int = 3,
        improvement_threshold: float = 0.1,
    ) -> None:
        """
        Initialize the critique engine.
        
        Args:
            evaluator: Policy evaluator for checking principles
            llm_client: LLM client for generating critiques and revisions
            max_iterations: Maximum critique-revise iterations
            improvement_threshold: Minimum improvement to continue iterating
        """
        self.evaluator = evaluator or PolicyEvaluator()
        self.llm_client = llm_client
        self.max_iterations = max_iterations
        self.improvement_threshold = improvement_threshold
    
    async def critique(
        self,
        content: str,
        context: dict[str, Any] | None = None,
    ) -> list[Critique]:
        """
        Generate critiques for content.
        
        Args:
            content: The content to critique
            context: Additional context
            
        Returns:
            List of critiques
        """
        # Evaluate against principles
        evaluation = await self.evaluator.evaluate_response(
            response=content,
            request=context.get("request") if context else None,
            context=context,
        )
        
        critiques: list[Critique] = []
        
        # Convert violations to critiques
        for violation in evaluation.violations:
            severity = self._violation_to_severity(violation.severity)
            critiques.append(Critique(
                target=content[:100] + "..." if len(content) > 100 else content,
                principle_id=violation.principle_id,
                principle_name=violation.principle_name,
                severity=severity,
                issue=violation.description,
                suggestion=violation.remediation_hint or "Revise to comply with this principle.",
                confidence=0.9,
            ))
        
        # Convert warnings to minor critiques
        for warning in evaluation.warnings:
            critiques.append(Critique(
                target=content[:100] + "..." if len(content) > 100 else content,
                principle_id=warning.principle_id,
                principle_name=warning.principle_name,
                severity=CritiqueSeverity.MINOR,
                issue=warning.description,
                suggestion=warning.remediation_hint or "Consider revising.",
                confidence=0.7,
            ))
        
        # If we have an LLM client, generate additional critiques
        if self.llm_client and not critiques:
            llm_critiques = await self._generate_llm_critiques(content, context)
            critiques.extend(llm_critiques)
        
        return critiques
    
    async def revise(
        self,
        content: str,
        critiques: list[Critique],
        context: dict[str, Any] | None = None,
    ) -> Revision:
        """
        Revise content based on critiques.
        
        Args:
            content: The content to revise
            critiques: Critiques to address
            context: Additional context
            
        Returns:
            Revision with the revised content
        """
        if not critiques:
            return Revision(
                original=content,
                revised=content,
                changes_made="No changes needed",
                improvement_score=0.0,
            )
        
        # If we have an LLM client, use it for revision
        if self.llm_client:
            revised = await self._generate_llm_revision(content, critiques, context)
        else:
            # Simple rule-based revision
            revised = self._apply_simple_revisions(content, critiques)
        
        # Calculate improvement score
        improvement = self._calculate_improvement(content, revised, critiques)
        
        return Revision(
            original=content,
            revised=revised,
            critiques_addressed=tuple(c.id for c in critiques),
            changes_made=self._describe_changes(content, revised, critiques),
            improvement_score=improvement,
        )
    
    async def critique_and_revise(
        self,
        content: str,
        context: dict[str, Any] | None = None,
    ) -> CritiqueResult:
        """
        Perform iterative critique and revision.
        
        Args:
            content: The content to critique and revise
            context: Additional context
            
        Returns:
            CritiqueResult with all critiques and revisions
        """
        start_time = time.perf_counter()
        
        all_critiques: list[Critique] = []
        all_revisions: list[Revision] = []
        current_content = content
        
        # Get initial evaluation
        original_evaluation = await self.evaluator.evaluate_response(
            response=content,
            request=context.get("request") if context else None,
            context=context,
        )
        
        for iteration in range(self.max_iterations):
            # Generate critiques
            critiques = await self.critique(current_content, context)
            all_critiques.extend(critiques)
            
            # If no critiques, we're done
            if not critiques:
                break
            
            # Filter to significant critiques
            significant_critiques = [
                c for c in critiques
                if c.severity in (CritiqueSeverity.MAJOR, CritiqueSeverity.CRITICAL)
            ]
            
            if not significant_critiques:
                # Only minor issues, we're done
                break
            
            # Revise
            revision = await self.revise(current_content, significant_critiques, context)
            all_revisions.append(revision)
            
            # Check if improvement is sufficient
            if revision.improvement_score < self.improvement_threshold:
                break
            
            current_content = revision.revised
        
        # Get final evaluation
        final_evaluation = await self.evaluator.evaluate_response(
            response=current_content,
            request=context.get("request") if context else None,
            context=context,
        )
        
        total_time = (time.perf_counter() - start_time) * 1000
        
        # Determine success
        success = (
            final_evaluation.is_allowed or
            (original_evaluation.has_violations and not final_evaluation.has_violations)
        )
        
        return CritiqueResult(
            critiques=tuple(all_critiques),
            revisions=tuple(all_revisions),
            original_evaluation=original_evaluation,
            final_evaluation=final_evaluation,
            iterations=len(all_revisions),
            total_time_ms=total_time,
            success=success,
        )
    
    def _violation_to_severity(self, severity: float) -> CritiqueSeverity:
        """Convert violation severity to critique severity."""
        if severity >= 0.9:
            return CritiqueSeverity.CRITICAL
        elif severity >= 0.7:
            return CritiqueSeverity.MAJOR
        elif severity >= 0.4:
            return CritiqueSeverity.MODERATE
        else:
            return CritiqueSeverity.MINOR
    
    async def _generate_llm_critiques(
        self,
        content: str,
        context: dict[str, Any] | None,
    ) -> list[Critique]:
        """Generate critiques using LLM."""
        if not self.llm_client:
            return []
        
        # Build prompt for critique
        principles_text = "\n".join(
            f"- {p.name}: {p.description}"
            for p in self.evaluator.registry.get_enabled()
        )
        
        prompt = f"""Critique the following response against these principles:

{principles_text}

Response to critique:
{content}

For each issue found, provide:
1. Which principle is violated
2. What the specific issue is
3. How to fix it

If the response is acceptable, say "No issues found."
"""
        
        try:
            response = await self.llm_client.complete(prompt=prompt)
            # Parse response into critiques (simplified)
            if "no issues found" in response.get_text().lower():
                return []
            
            # For now, return a generic critique if issues mentioned
            return [Critique(
                target=content[:100],
                severity=CritiqueSeverity.MODERATE,
                issue="LLM identified potential issues",
                suggestion=response.get_text(),
            )]
        except Exception:
            return []
    
    async def _generate_llm_revision(
        self,
        content: str,
        critiques: list[Critique],
        context: dict[str, Any] | None,
    ) -> str:
        """Generate revision using LLM."""
        if not self.llm_client:
            return content
        
        critiques_text = "\n".join(
            f"- {c.issue}: {c.suggestion}"
            for c in critiques
        )
        
        prompt = f"""Revise the following response to address these issues:

Issues to address:
{critiques_text}

Original response:
{content}

Provide a revised response that addresses all the issues while maintaining helpfulness.
"""
        
        try:
            response = await self.llm_client.complete(prompt=prompt)
            return response.get_text()
        except Exception:
            return content
    
    def _apply_simple_revisions(
        self,
        content: str,
        critiques: list[Critique],
    ) -> str:
        """Apply simple rule-based revisions."""
        revised = content
        
        for critique in critiques:
            if critique.severity == CritiqueSeverity.CRITICAL:
                # For critical issues, add a disclaimer
                revised = (
                    "I apologize, but I cannot provide that information as it may "
                    "violate safety guidelines. Let me offer an alternative approach:\n\n"
                    + revised
                )
                break
            elif "harmful" in critique.issue.lower():
                revised = (
                    "I want to be helpful while ensuring safety. "
                    + revised
                )
        
        return revised
    
    def _calculate_improvement(
        self,
        original: str,
        revised: str,
        critiques: list[Critique],
    ) -> float:
        """Calculate improvement score."""
        if original == revised:
            return 0.0
        
        # Simple heuristic: more changes = more improvement (up to a point)
        change_ratio = abs(len(revised) - len(original)) / max(len(original), 1)
        
        # Weight by critique severity
        severity_weight = sum(
            0.9 if c.severity == CritiqueSeverity.CRITICAL else
            0.7 if c.severity == CritiqueSeverity.MAJOR else
            0.4 if c.severity == CritiqueSeverity.MODERATE else
            0.2
            for c in critiques
        ) / max(len(critiques), 1)
        
        return min(1.0, change_ratio * severity_weight * 2)
    
    def _describe_changes(
        self,
        original: str,
        revised: str,
        critiques: list[Critique],
    ) -> str:
        """Describe the changes made."""
        if original == revised:
            return "No changes made"
        
        changes = []
        
        if len(revised) > len(original):
            changes.append("Added content")
        elif len(revised) < len(original):
            changes.append("Removed content")
        
        for critique in critiques:
            changes.append(f"Addressed {critique.principle_name or 'issue'}")
        
        return "; ".join(changes) if changes else "Modified content"
