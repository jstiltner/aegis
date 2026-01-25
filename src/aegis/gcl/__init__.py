"""
GCL (Grounded Commitment Learning) integration for the agent runtime.

This package provides:
- Commitment adapter for bridging agent runtime with GCL
- Full GCL library integration via bridge module
- Verification integration
- Commitment-aware tool execution
- Audit logging for commitments
"""

from __future__ import annotations

from .adapter import (
    GCLAdapter,
    CommitmentConfig,
    CommitmentContext,
)
from .models import (
    RuntimeCommitment,
    CommitmentStatus,
    CommitmentResult,
    ToolCommitment,
)
from .verifier import (
    RuntimeVerifier,
    VerificationHook,
)
from .manager import (
    CommitmentManager,
    CommitmentPolicy,
)
from .bridge import (
    GCLBridge,
    get_bridge,
    is_gcl_available,
)

__all__ = [
    # Adapter
    "GCLAdapter",
    "CommitmentConfig",
    "CommitmentContext",
    # Models
    "RuntimeCommitment",
    "CommitmentStatus",
    "CommitmentResult",
    "ToolCommitment",
    # Verifier
    "RuntimeVerifier",
    "VerificationHook",
    # Manager
    "CommitmentManager",
    "CommitmentPolicy",
    # Bridge (full GCL integration)
    "GCLBridge",
    "get_bridge",
    "is_gcl_available",
]
