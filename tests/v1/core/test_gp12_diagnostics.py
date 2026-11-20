# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression coverage for bounded GP-12 scheduler diagnostics."""

import logging

import pytest

from vllm.v1.core.gp12_diagnostics import GP12DiagnosticLimiter
from vllm.v1.outputs import ModelRunnerOutput

from .utils import create_requests, create_scheduler

pytestmark = pytest.mark.cpu_test


def test_limiter_summarizes_repeats_and_logs_state_changes():
    limiter = GP12DiagnosticLimiter(max_entries=2)
    assert limiter.observe("WAIT_REJECT", "req", ("capacity", 4)).kind == "first"

    emitted = [
        limiter.observe("WAIT_REJECT", "req", ("capacity", 4))
        for _ in range(7)
    ]
    repeats = [event for event in emitted if event is not None]
    assert [event.attempts for event in repeats] == [2, 4, 8]
    assert [event.suppressed_repeats for event in repeats] == [0, 1, 3]

    assert limiter.observe("WAIT_REJECT", "req", ("capacity", 4)) is None
    assert limiter.observe("WAIT_REJECT", "req", ("capacity", 4)) is None
    changed = limiter.observe("WAIT_REJECT", "req", ("capacity", 8))
    assert changed is not None
    assert changed.kind == "state_change"
    assert changed.previous_signature == ("capacity", 4)
    assert changed.signature == ("capacity", 8)
    assert changed.previous_attempts == 10
    assert changed.suppressed_repeats == 2


def test_limiter_is_bounded_and_flushes_state_on_request_clear():
    limiter = GP12DiagnosticLimiter(max_entries=2)
    limiter.should_log_once("WAIT_ENTRY", "req")
    limiter.observe("ALLOCATOR_REJECT", "req", ("full", (2, 1)))
    for request_id in ("one", "two", "three"):
        limiter.should_log_once("WAIT_ENTRY", request_id)
        limiter.observe("ALLOCATOR_REJECT", request_id, ("full", (2, 1)))
    assert limiter.entry_count <= 2

    limiter.should_log_once("WAIT_ENTRY", "flush")
    limiter.observe("ALLOCATOR_REJECT", "flush", ("full", (2, 1)))
    for _ in range(6):
        limiter.observe("ALLOCATOR_REJECT", "flush", ("full", (2, 1)))
    pending = limiter.clear_request("flush")
    assert [(name, event.suppressed_repeats) for name, event in pending] == [
        ("ALLOCATOR_REJECT", 3)
    ]
    assert limiter.should_log_once("WAIT_ENTRY", "flush")


def test_repeated_wait_rejection_logs_are_rate_limited(
    caplog: pytest.LogCaptureFixture,
):
    caplog.set_level(logging.WARNING, logger="vllm.v1.core.sched.scheduler")
    scheduler = create_scheduler(
        max_num_seqs=2,
        max_num_batched_tokens=512,
        max_model_len=1024,
        num_blocks=4,
        block_size=16,
        enable_chunked_prefill=True,
        long_prefill_token_threshold=512,
        prefill_batch_barrier=True,
    )
    (blocked,) = create_requests(
        num_requests=1,
        num_tokens=1024,
        max_tokens=1,
        req_ids=["blocked"],
    )
    scheduler.add_request(blocked)

    for _ in range(64):
        assert scheduler.schedule().num_scheduled_tokens == {}

    messages = [record.getMessage() for record in caplog.records]
    rejects = [message for message in messages if message.startswith("GP12_WAIT_REJECT")]
    assert len(rejects) <= 7
    assert any(message.startswith("GP12_WAIT_REJECT ") for message in rejects)
    assert (
        sum(
            message.startswith("GP12_WAIT_ENTRY id=blocked ")
            for message in messages
        )
        == 1
    )
    assert sum(message.startswith("GP12_WAIT_POST ") for message in messages) == 1


def test_capacity_wait_diagnostics_remain_once_per_request(caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.WARNING, logger="vllm.v1.core.sched.scheduler")
    scheduler = create_scheduler(
        max_num_seqs=1,
        max_num_batched_tokens=512,
        max_model_len=1024,
        prefill_batch_barrier=True,
    )
    running, waiting = create_requests(
        num_requests=2,
        num_tokens=8,
        max_tokens=128,
        req_ids=["running", "waiting"],
    )
    scheduler.add_request(running)
    first = scheduler.schedule()
    scheduler.update_from_output(
        first,
        ModelRunnerOutput(
            req_ids=["running"],
            req_id_to_index={"running": 0},
            sampled_token_ids=[[0]],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )
    scheduler.add_request(waiting)

    for _ in range(16):
        output = scheduler.schedule()
        assert set(output.num_scheduled_tokens) == {"running"}
        scheduler.update_from_output(
            output,
            ModelRunnerOutput(
                req_ids=["running"],
                req_id_to_index={"running": 0},
                sampled_token_ids=[[0]],
                logprobs=None,
                prompt_logprobs_dict={},
                pooler_output=[],
            ),
        )

    messages = [record.getMessage() for record in caplog.records]
    assert sum(
        message.startswith("GP12_WAIT_ENTRY id=waiting ") for message in messages
    ) == 1
    assert (
        sum(message.startswith("GP12_CAPACITY_BLOCK ") for message in messages)
        == 1
    )
    assert sum(message.startswith("GP12_WAIT_POST ") for message in messages) == 1
