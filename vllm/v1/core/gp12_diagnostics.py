# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded rate limiting for downstream GP-12 scheduler diagnostics."""

from collections import OrderedDict
from dataclasses import dataclass
from typing import Hashable, Literal


@dataclass
class _Observation:
    signature: Hashable
    attempts: int = 1
    last_reported_attempt: int = 1


@dataclass(frozen=True)
class DiagnosticEvent:
    kind: Literal["first", "repeat", "state_change"]
    attempts: int
    suppressed_repeats: int = 0
    previous_attempts: int = 0
    previous_signature: Hashable | None = None
    signature: Hashable | None = None


class GP12DiagnosticLimiter:
    """Keep per-request diagnostic state bounded and suppress duplicate logs.

    Repeated unchanged events are summarized at exponentially increasing
    intervals. State changes are emitted immediately, and callers can flush
    any remaining suppressed count when a request is admitted or removed.
    """

    def __init__(self, max_entries: int = 4096):
        if max_entries < 2:
            raise ValueError("max_entries must be at least two")
        self.max_entries = max_entries
        self._once_limit = max_entries // 2
        self._observation_limit = max_entries - self._once_limit
        self._once: OrderedDict[tuple[str, str], None] = OrderedDict()
        self._observations: OrderedDict[tuple[str, str], _Observation] = (
            OrderedDict()
        )

    @property
    def entry_count(self) -> int:
        """Number of retained request/event keys across both stores."""
        return len(self._once) + len(self._observations)

    def should_log_once(self, event: str, request_id: str) -> bool:
        key = (event, request_id)
        if key in self._once:
            self._once.move_to_end(key)
            return False
        self._once[key] = None
        self._trim(self._once, self._once_limit)
        return True

    def observe(
        self, event: str, request_id: str, signature: Hashable
    ) -> DiagnosticEvent | None:
        key = (event, request_id)
        state = self._observations.get(key)
        if state is None:
            self._observations[key] = _Observation(signature)
            self._trim(self._observations, self._observation_limit)
            return DiagnosticEvent("first", attempts=1, signature=signature)

        self._observations.move_to_end(key)
        if state.signature != signature:
            suppressed = state.attempts - state.last_reported_attempt
            previous_signature = state.signature
            self._observations[key] = _Observation(signature)
            return DiagnosticEvent(
                "state_change",
                attempts=1,
                suppressed_repeats=suppressed,
                previous_attempts=state.attempts,
                previous_signature=previous_signature,
                signature=signature,
            )

        state.attempts += 1
        if state.attempts & (state.attempts - 1):
            return None

        suppressed = state.attempts - state.last_reported_attempt - 1
        state.last_reported_attempt = state.attempts
        return DiagnosticEvent(
            "repeat",
            attempts=state.attempts,
            suppressed_repeats=suppressed,
            signature=signature,
        )

    def clear_request(self, request_id: str) -> list[tuple[str, DiagnosticEvent]]:
        """Drop all state for a request and return unsummarized repeat counts."""
        pending: list[tuple[str, DiagnosticEvent]] = []
        for key, state in list(self._observations.items()):
            event, key_request_id = key
            if key_request_id != request_id:
                continue
            suppressed = state.attempts - state.last_reported_attempt
            if suppressed:
                pending.append(
                    (
                        event,
                        DiagnosticEvent(
                            "repeat",
                            attempts=state.attempts,
                            suppressed_repeats=suppressed,
                            signature=state.signature,
                        ),
                    )
                )
            del self._observations[key]

        for key in list(self._once):
            if key[1] == request_id:
                del self._once[key]
        return pending

    @staticmethod
    def _trim(entries: OrderedDict, limit: int) -> None:
        while len(entries) > limit:
            entries.popitem(last=False)
