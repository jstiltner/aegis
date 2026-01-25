"""
Recovery module for commitment violation detection and recovery.

This module provides:
- Violation detection for commitments
- Recovery strategies for handling violations
- Recovery orchestration for automated recovery
"""

from .detector import (
    ViolationType,
    Violation,
    ViolationDetector,
)
from .strategies import (
    RecoveryAction,
    RecoveryStrategy,
    RetryStrategy,
    EscalateStrategy,
    CompensateStrategy,
    RenegotiateStrategy,
    StrategySelector,
)
from .orchestrator import (
    RecoveryResult,
    RecoveryOrchestrator,
)

__all__ = [
    # Detector
    "ViolationType",
    "Violation",
    "ViolationDetector",
    # Strategies
    "RecoveryAction",
    "RecoveryStrategy",
    "RetryStrategy",
    "EscalateStrategy",
    "CompensateStrategy",
    "RenegotiateStrategy",
    "StrategySelector",
    # Orchestrator
    "RecoveryResult",
    "RecoveryOrchestrator",
]
