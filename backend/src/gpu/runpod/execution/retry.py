"""RunPod launch retry policy.

A single training "attempt" maps to one ``run_id``. When a pod fails before
``99_done`` reaches S3 we want to relaunch — but only when the failure is
infrastructural (the pod never had a chance to run user code), not when
the user code itself is broken.

Retryable failure reasons:

  * ``scheduling_failed``  — RunPod never assigned a node
  * ``image_pull_stuck``   — container image download stalled
  * ``driver_mismatch``    — onstart's CUDA self-test exited non-zero

Non-retryable (user code or accounting):

  * ``exit_1`` … ``exit_N`` — training process returned non-zero
  * ``oom``                — kernel killed the process for memory pressure
  * ``cost_limit_hit``     — we don't want to compound an over-budget run

Each retry uses a *new* ``run_id`` so S3 marker / artifact prefixes never
collide and the original failure remains audit-able. The chain is recorded
through ``run.json.retry_of = <previous_run_id>`` (linked list).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class FailureReason(str, Enum):
    SCHEDULING_FAILED = "scheduling_failed"
    IMAGE_PULL_STUCK = "image_pull_stuck"
    DRIVER_MISMATCH = "driver_mismatch"
    OOM = "oom"
    COST_LIMIT_HIT = "cost_limit_hit"
    TIMEOUT = "timeout"
    NON_ZERO_EXIT = "non_zero_exit"
    UNKNOWN = "unknown"


_RETRYABLE: frozenset[FailureReason] = frozenset(
    {
        FailureReason.SCHEDULING_FAILED,
        FailureReason.IMAGE_PULL_STUCK,
        FailureReason.DRIVER_MISMATCH,
    }
)


def is_retryable(reason: FailureReason) -> bool:
    return reason in _RETRYABLE


@dataclass(frozen=True)
class RetryDecision:
    """The outcome of a retry policy lookup."""

    should_retry: bool
    reason: FailureReason
    attempt_after_this: int  # how many attempts will have been made if we DO retry
    explanation: str


def decide(
    *,
    failure_reason: FailureReason,
    attempts_so_far: int,
    max_retries: int,
) -> RetryDecision:
    """Pure decision: given the current failure + history, should we retry?

    ``attempts_so_far`` counts the failed attempt that just finished. So the
    very first failure is ``attempts_so_far=1``. Retries are bounded by
    ``max_retries`` (e.g. ``max_retries=2`` permits a maximum of 3 total
    attempts).
    """
    next_attempt = attempts_so_far + 1
    if not is_retryable(failure_reason):
        return RetryDecision(
            should_retry=False,
            reason=failure_reason,
            attempt_after_this=attempts_so_far,
            explanation=f"reason={failure_reason.value} is not in the retryable set",
        )
    if attempts_so_far > max_retries:
        return RetryDecision(
            should_retry=False,
            reason=failure_reason,
            attempt_after_this=attempts_so_far,
            explanation=(
                f"already at attempts_so_far={attempts_so_far} > "
                f"max_retries={max_retries}"
            ),
        )
    return RetryDecision(
        should_retry=True,
        reason=failure_reason,
        attempt_after_this=next_attempt,
        explanation=(
            f"reason={failure_reason.value} retryable; will attempt "
            f"#{next_attempt} (cap {max_retries + 1})"
        ),
    )


@dataclass(frozen=True)
class RetryChain:
    """Linked-list view of a retry sequence. Persisted via run.json.retry_of."""

    head_run_id: str
    attempts: list[str] = field(default_factory=list)

    @property
    def length(self) -> int:
        return len(self.attempts)


def append_attempt(chain: RetryChain, run_id: str) -> RetryChain:
    return RetryChain(
        head_run_id=chain.head_run_id,
        attempts=[*chain.attempts, run_id],
    )
