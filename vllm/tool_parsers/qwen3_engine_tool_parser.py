# Downstream changes for vLLM-CMP50HX-Definitive-DFlash2; see PATCHES.md.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.parser.engine.registered_adapters import Qwen3ParserToolAdapter


class Qwen3EngineToolParser(Qwen3ParserToolAdapter):  # type: ignore[valid-type, misc]
    # Qwen3 auto tool calls use XML and are handled by ParserEngine. In
    # required/named mode, structured generation emits a JSON function array
    # or arguments object, which the generic OpenAI parser must decode.
    supports_required_and_named = True
    structural_tag_model = "qwen_3_coder"
