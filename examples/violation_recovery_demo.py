#!/usr/bin/env python3
"""
Commitment Violation Detection and Recovery Demo.

This interactive demo showcases the violation detection and recovery
capabilities of the autonomous agent infrastructure.

Features demonstrated:
- Violation detection (deadline, policy, condition violations)
- Recovery strategies (retry, escalate, compensate, renegotiate)
- Recovery orchestration
- Real-time monitoring

Usage:
    python examples/violation_recovery_demo.py
"""

import asyncio
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aegis.commitments import RuntimeCommitment, CommitmentStatus
from aegis.recovery import (
    ViolationType,
    Violation,
    ViolationDetector,
    RecoveryAction,
    RecoveryStrategy,
    RetryStrategy,
    EscalateStrategy,
    CompensateStrategy,
    RenegotiateStrategy,
    StrategySelector,
    RecoveryResult,
    RecoveryOrchestrator,
)


def print_header(title: str) -> None:
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_section(title: str) -> None:
    """Print a section header."""
    print(f"\n--- {title} ---")


def print_violation(violation: Violation) -> None:
    """Print violation details."""
    print(f"  ID: {violation.id[:8]}...")
    print(f"  Type: {violation.type.value}")
    print(f"  Severity: {violation.severity:.2f}")
    print(f"  Description: {violation.description}")
    print(f"  Agent: {violation.agent_id}")
    print(f"  Resolved: {violation.resolved}")


def print_result(result: RecoveryResult) -> None:
    """Print recovery result details."""
    print(f"  Status: {result.status.value}")
    print(f"  Strategy: {result.strategy_used or 'None'}")
    if result.error:
        print(f"  Error: {result.error}")
    if result.result:
        print(f"  Result: {result.result}")


async def demo_violation_detection() -> None:
    """Demonstrate violation detection."""
    print_header("VIOLATION DETECTION DEMO")
    
    detector = ViolationDetector()
    
    # Create a commitment with a past deadline
    print_section("Creating commitment with past deadline")
    commitment = RuntimeCommitment(
        debtor="worker-agent",
        creditor="manager-agent",
        action="process_customer_data",
        condition="customer_data_available",
        deadline=datetime.now(timezone.utc) - timedelta(hours=2),
    )
    print(f"  Commitment ID: {commitment.id[:8]}...")
    print(f"  Action: {commitment.action}")
    print(f"  Deadline: {commitment.deadline}")
    print(f"  Status: {commitment.status.value}")
    
    # Check for violations
    print_section("Checking for violations")
    violations = detector.check_commitment(commitment)
    
    print(f"  Found {len(violations)} violation(s):")
    for v in violations:
        print()
        print_violation(v)
    
    # Demonstrate policy violation detection
    print_section("Detecting policy violation")
    policy_violation = detector.check_policy_violation(
        commitment,
        action="access_external_api",
        policy_result={
            "allowed": False,
            "reason": "External API access not permitted",
            "severity": 0.85,
        },
    )
    if policy_violation:
        print_violation(policy_violation)
    
    # Demonstrate resource violation detection
    print_section("Detecting resource violation")
    resource_violation = detector.check_resource_violation(
        commitment,
        resource="api_calls",
        used=150,
        limit=100,
    )
    if resource_violation:
        print_violation(resource_violation)
    
    # Show statistics
    print_section("Violation Statistics")
    stats = detector.get_statistics()
    print(f"  Total violations: {stats['total']}")
    print(f"  Resolved: {stats['resolved']}")
    print(f"  Unresolved: {stats['unresolved']}")
    print(f"  Average severity: {stats['average_severity']:.2f}")
    print(f"  By type: {stats['by_type']}")


async def demo_recovery_strategies() -> None:
    """Demonstrate recovery strategies."""
    print_header("RECOVERY STRATEGIES DEMO")
    
    # Create test commitment and violation
    commitment = RuntimeCommitment(
        debtor="worker-agent",
        creditor="manager-agent",
        action="analyze_data",
        deadline=datetime.now(timezone.utc) - timedelta(minutes=30),
    )
    
    deadline_violation = Violation(
        type=ViolationType.DEADLINE_EXCEEDED,
        commitment_id=commitment.id,
        agent_id=commitment.debtor,
        description="Deadline exceeded by 30 minutes",
        severity=0.5,
        context={"overdue_seconds": 1800},
    )
    
    # Test Retry Strategy
    print_section("Retry Strategy")
    retry = RetryStrategy(max_retries=3, base_delay=1.0)
    print(f"  Can handle: {retry.can_handle(deadline_violation, commitment)}")
    
    if retry.can_handle(deadline_violation, commitment):
        plan = retry.create_plan(deadline_violation, commitment)
        print(f"  Plan action: {plan.action.value}")
        print(f"  Retry count: {plan.parameters.get('retry_count')}")
        print(f"  Delay: {plan.parameters.get('delay')}s")
        
        result = await retry.execute(plan, commitment)
        print(f"  Execution result: {result}")
    
    # Test Renegotiate Strategy
    print_section("Renegotiate Strategy")
    renegotiate = RenegotiateStrategy()
    print(f"  Can handle: {renegotiate.can_handle(deadline_violation, commitment)}")
    
    if renegotiate.can_handle(deadline_violation, commitment):
        plan = renegotiate.create_plan(deadline_violation, commitment)
        print(f"  Plan action: {plan.action.value}")
        print(f"  New deadline: {plan.parameters.get('new_deadline')}")
        print(f"  Extension: {plan.parameters.get('extension_seconds')}s")
        
        result = await renegotiate.execute(plan, commitment)
        print(f"  Execution result: {result}")
    
    # Test Escalate Strategy with high-severity violation
    print_section("Escalate Strategy")
    policy_violation = Violation(
        type=ViolationType.POLICY_VIOLATED,
        commitment_id=commitment.id,
        agent_id=commitment.debtor,
        description="Attempted unauthorized data access",
        severity=0.95,
    )
    
    escalate = EscalateStrategy()
    print(f"  Can handle: {escalate.can_handle(policy_violation, commitment)}")
    
    if escalate.can_handle(policy_violation, commitment):
        plan = escalate.create_plan(policy_violation, commitment)
        print(f"  Plan action: {plan.action.value}")
        print(f"  Requires human: {plan.parameters.get('requires_human')}")
        print(f"  Priority: {plan.priority}")
        
        result = await escalate.execute(plan, commitment)
        print(f"  Execution result: {result}")
    
    # Test Compensate Strategy
    print_section("Compensate Strategy")
    condition_violation = Violation(
        type=ViolationType.CONDITION_VIOLATED,
        commitment_id=commitment.id,
        agent_id=commitment.debtor,
        description="Required condition no longer holds",
        severity=0.6,
    )
    
    compensate = CompensateStrategy()
    print(f"  Can handle: {compensate.can_handle(condition_violation, commitment)}")
    
    if compensate.can_handle(condition_violation, commitment):
        plan = compensate.create_plan(condition_violation, commitment)
        print(f"  Plan action: {plan.action.value}")
        
        result = await compensate.execute(plan, commitment)
        print(f"  Execution result: {result}")


async def demo_strategy_selection() -> None:
    """Demonstrate strategy selection."""
    print_header("STRATEGY SELECTION DEMO")
    
    selector = StrategySelector()
    selector.register(RetryStrategy())
    selector.register(EscalateStrategy())
    selector.register(CompensateStrategy())
    selector.register(RenegotiateStrategy())
    
    commitment = RuntimeCommitment(
        debtor="worker-agent",
        creditor="manager-agent",
        action="process_request",
    )
    
    # Test different violation types
    test_cases = [
        ("Low severity deadline violation", Violation(
            type=ViolationType.DEADLINE_EXCEEDED,
            commitment_id=commitment.id,
            agent_id=commitment.debtor,
            description="Slightly overdue",
            severity=0.3,
            context={"overdue_seconds": 300},
        )),
        ("High severity policy violation", Violation(
            type=ViolationType.POLICY_VIOLATED,
            commitment_id=commitment.id,
            agent_id=commitment.debtor,
            description="Security policy breach",
            severity=0.95,
        )),
        ("Medium severity condition violation", Violation(
            type=ViolationType.CONDITION_VIOLATED,
            commitment_id=commitment.id,
            agent_id=commitment.debtor,
            description="Precondition failed",
            severity=0.5,
        )),
        ("Resource exceeded", Violation(
            type=ViolationType.RESOURCE_EXCEEDED,
            commitment_id=commitment.id,
            agent_id=commitment.debtor,
            description="Memory limit exceeded",
            severity=0.4,
        )),
    ]
    
    for name, violation in test_cases:
        print_section(name)
        print(f"  Type: {violation.type.value}")
        print(f"  Severity: {violation.severity}")
        
        selected = selector.select(violation, commitment)
        print(f"  Selected strategy: {selected.name if selected else 'None'}")
        
        all_strategies = selector.select_all(violation, commitment)
        print(f"  All applicable: {[s.name for s in all_strategies]}")


async def demo_recovery_orchestration() -> None:
    """Demonstrate recovery orchestration."""
    print_header("RECOVERY ORCHESTRATION DEMO")
    
    orchestrator = RecoveryOrchestrator()
    
    # Add event handler
    async def on_recovery(result: RecoveryResult) -> None:
        print(f"  [Event] Recovery {result.status.value}: {result.strategy_used}")
    
    orchestrator.add_event_handler(on_recovery)
    
    # Create commitment
    commitment = RuntimeCommitment(
        debtor="data-processor",
        creditor="orchestrator",
        action="transform_dataset",
        deadline=datetime.now(timezone.utc) - timedelta(hours=1),
    )
    
    print_section("Registering commitment for monitoring")
    orchestrator.register_commitment(commitment)
    print(f"  Commitment: {commitment.id[:8]}...")
    
    # Create and recover from violations
    print_section("Creating deadline violation")
    deadline_violation = Violation(
        type=ViolationType.DEADLINE_EXCEEDED,
        commitment_id=commitment.id,
        agent_id=commitment.debtor,
        description="Processing deadline exceeded",
        severity=0.6,
        context={"overdue_seconds": 3600},
    )
    
    print_section("Attempting recovery")
    result = await orchestrator.recover(deadline_violation, commitment)
    print_result(result)
    
    # Create policy violation
    print_section("Creating policy violation")
    policy_violation = Violation(
        type=ViolationType.POLICY_VIOLATED,
        commitment_id=commitment.id,
        agent_id=commitment.debtor,
        description="Attempted to access restricted resource",
        severity=0.9,
    )
    
    print_section("Attempting recovery (should escalate)")
    result = await orchestrator.recover(policy_violation, commitment)
    print_result(result)
    
    # Show statistics
    print_section("Recovery Statistics")
    stats = orchestrator.get_statistics()
    print(f"  Total attempts: {stats['total_attempts']}")
    print(f"  Succeeded: {stats['succeeded']}")
    print(f"  Failed: {stats['failed']}")
    print(f"  Escalated: {stats['escalated']}")
    print(f"  Success rate: {stats['success_rate']:.1%}")
    print(f"  By strategy: {stats['by_strategy']}")


async def demo_full_recovery_scenario() -> None:
    """Demonstrate a full recovery scenario."""
    print_header("FULL RECOVERY SCENARIO")
    
    print("""
    Scenario: An agent is processing a batch of customer records.
    The agent encounters multiple issues during processing:
    1. Deadline is exceeded due to slow processing
    2. A policy violation occurs when trying to access external data
    3. A condition violation when required data becomes unavailable
    
    The recovery system will handle each violation appropriately.
    """)
    
    orchestrator = RecoveryOrchestrator()
    
    # Create the commitment
    commitment = RuntimeCommitment(
        debtor="batch-processor",
        creditor="job-scheduler",
        action="process_customer_batch",
        condition="customer_data_available AND processing_resources_available",
        deadline=datetime.now(timezone.utc) + timedelta(hours=2),
    )
    
    print_section("Initial Commitment")
    print(f"  ID: {commitment.id[:8]}...")
    print(f"  Action: {commitment.action}")
    print(f"  Condition: {commitment.condition}")
    print(f"  Deadline: {commitment.deadline}")
    
    # Simulate processing with violations
    violations_and_recoveries = []
    
    # Phase 1: Processing starts, deadline approaches
    print_section("Phase 1: Processing starts")
    print("  Agent begins processing customer records...")
    await asyncio.sleep(0.5)
    
    # Phase 2: Deadline exceeded
    print_section("Phase 2: Deadline exceeded")
    commitment.deadline = datetime.now(timezone.utc) - timedelta(minutes=15)
    
    deadline_violation = Violation(
        type=ViolationType.DEADLINE_EXCEEDED,
        commitment_id=commitment.id,
        agent_id=commitment.debtor,
        description="Batch processing deadline exceeded",
        severity=0.5,
        context={"overdue_seconds": 900, "retry_count": 0},
    )
    
    print("  Violation detected: Deadline exceeded")
    result = await orchestrator.recover(deadline_violation, commitment)
    violations_and_recoveries.append(("Deadline exceeded", result))
    print(f"  Recovery: {result.strategy_used} -> {result.status.value}")
    
    # Phase 3: Policy violation
    print_section("Phase 3: Policy violation")
    print("  Agent attempts to fetch external enrichment data...")
    
    policy_violation = Violation(
        type=ViolationType.POLICY_VIOLATED,
        commitment_id=commitment.id,
        agent_id=commitment.debtor,
        description="External API access blocked by policy",
        severity=0.8,
        context={"policy": "no_external_api", "attempted_action": "fetch_enrichment"},
    )
    
    print("  Violation detected: Policy violation")
    result = await orchestrator.recover(policy_violation, commitment)
    violations_and_recoveries.append(("Policy violation", result))
    print(f"  Recovery: {result.strategy_used} -> {result.status.value}")
    
    # Phase 4: Condition violation
    print_section("Phase 4: Condition violation")
    print("  Source data becomes unavailable...")
    
    condition_violation = Violation(
        type=ViolationType.CONDITION_VIOLATED,
        commitment_id=commitment.id,
        agent_id=commitment.debtor,
        description="customer_data_available condition no longer holds",
        severity=0.7,
        context={"condition": "customer_data_available", "reason": "source_offline"},
    )
    
    print("  Violation detected: Condition violation")
    result = await orchestrator.recover(condition_violation, commitment)
    violations_and_recoveries.append(("Condition violation", result))
    print(f"  Recovery: {result.strategy_used} -> {result.status.value}")
    
    # Summary
    print_section("Recovery Summary")
    print(f"  Total violations: {len(violations_and_recoveries)}")
    for name, result in violations_and_recoveries:
        status_icon = "✓" if result.status.value in ["succeeded", "escalated"] else "✗"
        print(f"  {status_icon} {name}: {result.strategy_used} -> {result.status.value}")
    
    # Final statistics
    print_section("Final Statistics")
    stats = orchestrator.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")


async def interactive_demo() -> None:
    """Run an interactive demo menu."""
    print_header("VIOLATION RECOVERY DEMO")
    print("""
    This demo showcases the commitment violation detection and
    recovery capabilities of the autonomous agent infrastructure.
    
    Select a demo to run:
    """)
    
    demos = [
        ("Violation Detection", demo_violation_detection),
        ("Recovery Strategies", demo_recovery_strategies),
        ("Strategy Selection", demo_strategy_selection),
        ("Recovery Orchestration", demo_recovery_orchestration),
        ("Full Recovery Scenario", demo_full_recovery_scenario),
        ("Run All Demos", None),
    ]
    
    for i, (name, _) in enumerate(demos, 1):
        print(f"    {i}. {name}")
    
    print("\n    0. Exit")
    
    while True:
        try:
            choice = input("\n  Enter choice (0-6): ").strip()
            
            if choice == "0":
                print("\n  Goodbye!")
                break
            
            choice_num = int(choice)
            if choice_num < 1 or choice_num > len(demos):
                print("  Invalid choice. Please try again.")
                continue
            
            if choice_num == len(demos):
                # Run all demos
                for name, demo_func in demos[:-1]:
                    await demo_func()
                    print("\n  Press Enter to continue...")
                    input()
            else:
                name, demo_func = demos[choice_num - 1]
                await demo_func()
            
            print("\n  Demo complete!")
            
        except ValueError:
            print("  Invalid input. Please enter a number.")
        except KeyboardInterrupt:
            print("\n\n  Interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"  Error: {e}")


async def main() -> None:
    """Main entry point."""
    if len(sys.argv) > 1 and sys.argv[1] == "--all":
        # Run all demos non-interactively
        await demo_violation_detection()
        await demo_recovery_strategies()
        await demo_strategy_selection()
        await demo_recovery_orchestration()
        await demo_full_recovery_scenario()
    else:
        # Run interactive demo
        await interactive_demo()


if __name__ == "__main__":
    asyncio.run(main())
