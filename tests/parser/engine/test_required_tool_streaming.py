# Recovered Experimental B downstream test; see PATCHES.md.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for required-tool partial JSON streaming."""

from vllm.tool_parsers.streaming import extract_required_tool_call_streaming


def test_required_tool_stream_tolerates_unmatched_partial_closer():
    # Streaming can expose a temporarily malformed prefix. The fast partial
    # JSON repairer currently underflows its delimiter stack on this prefix.
    delta, name_returned = extract_required_tool_call_streaming(
        previous_text="",
        current_text="}",
        delta_text="}",
        function_name_returned=False,
        tool_call_idx=None,
        tool_call_id_type="random",
    )

    assert delta is None
    assert name_returned is False


def test_required_tool_stream_still_parses_valid_json():
    import json

    output = '[{"name":"lookup_city","parameters":{"city":"Oslo"}}]'
    delta, name_returned = extract_required_tool_call_streaming(
        previous_text="",
        current_text=output,
        delta_text=output,
        function_name_returned=False,
        tool_call_idx=None,
        tool_call_id_type="random",
    )

    assert name_returned is True
    assert delta is not None
    assert delta.tool_calls[0].function.name == "lookup_city"
    assert json.loads(delta.tool_calls[0].function.arguments) == {"city": "Oslo"}
