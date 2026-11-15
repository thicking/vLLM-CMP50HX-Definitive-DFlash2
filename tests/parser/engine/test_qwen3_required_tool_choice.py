# Recovered Experimental B downstream test; see PATCHES.md.
from types import SimpleNamespace
from unittest.mock import MagicMock

from tests.parser.engine.conftest import make_mock_tokenizer
from vllm.parser.abstract_parser import DelegatingParser
from vllm.parser.qwen3 import TOOL_CALL_END, TOOL_CALL_START
from vllm.tool_parsers.qwen3_engine_tool_parser import Qwen3EngineToolParser


def test_qwen3_required_json_array_becomes_openai_tool_call():
    tokenizer = make_mock_tokenizer({TOOL_CALL_START: 100, TOOL_CALL_END: 101})
    tool = {
        "type": "function",
        "function": {
            "name": "lookup_city",
            "description": "Look up a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
    tool_parser = Qwen3EngineToolParser(tokenizer, tools=[tool])
    request = MagicMock()
    request.tool_choice = "required"
    request.tools = [tool]
    request.include_reasoning = True
    raw = '[{"name":"lookup_city","parameters":{"city":"Oslo"}}]'
    parser = SimpleNamespace(
        _tool_parser=tool_parser,
        _engine_based=True,
        _make_tool_call_id=lambda name: "call_" + name,
        extract_tool_calls=lambda content, request: tool_parser.extract_tool_calls(
            content, request
        ),
    )

    calls, content = DelegatingParser._extract_tool_calls(
        parser, raw, request, enable_auto_tools=True
    )

    assert calls is not None
    assert [(call.name, call.arguments) for call in calls] == [
        ("lookup_city", '{"city": "Oslo"}')
    ]
    assert content is None
