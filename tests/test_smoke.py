# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""No-GPU smoke tests for the Xing4_0 plugin.

Run:  python -m pytest tests/test_smoke.py -q

The sentinel strings (tool-call / think markers) are imported from the
vendored modules instead of being spelled out in this file.
"""

import json
from types import SimpleNamespace

import pytest

import vllm_xing4_0_plugin
from vllm_xing4_0_plugin import (
    ARCHITECTURE,
    OVERRIDE_ENV,
    PARSER_NAME,
)
from vllm_xing4_0_plugin.xing4_0_reasoning_parser import (
    END_THINK,
    START_THINK,
    Xing4_0ReasoningParser,
)
from vllm_xing4_0_plugin.xing4_0_tool_parser import (
    FORWARD_UNKNOWN_TOOLS_ENV,
    TOOL_END_TOKEN,
    TOOL_START_TOKEN,
    Xing4_0ToolParser,
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "days": {"type": "integer"},
                },
            },
        },
    }
]


class FakeTokenizer:
    """Minimal TokenizerLike: the parsers only need get_vocab()."""

    def __init__(self, extra_vocab=None):
        self._vocab = {START_THINK: 100, END_THINK: 101}
        self._vocab.update(extra_vocab or {})

    def get_vocab(self):
        return self._vocab

    def __call__(self, *args, **kwargs):
        return []


def _chat_request(tools=TOOLS):
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
    )

    return ChatCompletionRequest(model="m", messages=[], tools=tools)


@pytest.fixture(autouse=True)
def _register_plugin(monkeypatch):
    """Force registration even on vLLM builds that ship Xing4_0 in tree."""
    monkeypatch.setenv(OVERRIDE_ENV, "1")
    vllm_xing4_0_plugin.register()


def test_entry_point_discoverable():
    from importlib.metadata import entry_points

    eps = entry_points(group="vllm.general_plugins")
    matches = [e for e in eps if e.name == PARSER_NAME]
    assert matches, f"no '{PARSER_NAME}' entry point in vllm.general_plugins"
    assert matches[0].value == "vllm_xing4_0_plugin:register"


def test_model_registered():
    from vllm import ModelRegistry

    assert ARCHITECTURE in ModelRegistry.get_supported_archs()
    target = ModelRegistry.models[ARCHITECTURE]
    assert (target.module_name, target.class_name) == (
        "vllm_xing4_0_plugin.xing4_0",
        ARCHITECTURE,
    )


def test_hf_config_registered():
    from transformers import DeepseekV3Config
    from vllm.transformers_utils.config import _CONFIG_REGISTRY

    assert PARSER_NAME in _CONFIG_REGISTRY
    assert _CONFIG_REGISTRY[PARSER_NAME] is DeepseekV3Config


def test_parsers_registered():
    from vllm.reasoning.abs_reasoning_parsers import ReasoningParserManager
    from vllm.tool_parsers.abstract_tool_parser import ToolParserManager

    assert ReasoningParserManager.lazy_parsers[PARSER_NAME] == (
        "vllm_xing4_0_plugin.xing4_0_reasoning_parser",
        "Xing4_0ReasoningParser",
    )
    assert ToolParserManager.lazy_parsers[PARSER_NAME] == (
        "vllm_xing4_0_plugin.xing4_0_tool_parser",
        "Xing4_0ToolParser",
    )


class _FakeHfConfig:
    """Duck-typed PretrainedConfig for the speculative-config override."""

    def __init__(self, model_type, architectures=None, **kwargs):
        self.model_type = model_type
        self.architectures = architectures or ["Whatever"]
        for key, value in kwargs.items():
            setattr(self, key, value)

    def update(self, values):
        for key, value in values.items():
            setattr(self, key, value)


def test_speculative_override_maps_xing4_0_mtp():
    from vllm.config.speculative import SpeculativeConfig

    cfg = _FakeHfConfig(
        PARSER_NAME,
        architectures=["Xing4_0ForCausalLM"],
        num_nextn_predict_layers=1,
    )
    out = SpeculativeConfig.hf_config_override(cfg)
    assert out.model_type == "deepseek_mtp"
    assert out.architectures == ["DeepSeekMTPModel"]
    assert out.n_predict == 1


def test_speculative_override_leaves_other_models_alone():
    from vllm.config.speculative import SpeculativeConfig

    cfg = _FakeHfConfig("llama", architectures=["LlamaForCausalLM"])
    out = SpeculativeConfig.hf_config_override(cfg)
    assert out.model_type == "llama"
    assert out.architectures == ["LlamaForCausalLM"]


def _convertor(model_type, **text_attrs):
    from vllm.transformers_utils.model_arch_config_convertor import (
        ModelArchConfigConvertorBase,
    )

    hf_text_config = _FakeHfConfig(model_type, **text_attrs)
    return ModelArchConfigConvertorBase(hf_text_config, hf_text_config)


def test_is_deepseek_mla_true_for_xing4_0():
    assert _convertor(PARSER_NAME, kv_lora_rank=512).is_deepseek_mla()
    assert not _convertor(PARSER_NAME).is_deepseek_mla()  # no kv_lora_rank


def test_is_deepseek_mla_unchanged_for_other_models():
    assert not _convertor("llama").is_deepseek_mla()
    assert _convertor("deepseek_v3", kv_lora_rank=512).is_deepseek_mla()


def test_tool_parser_json_call():
    parser = Xing4_0ToolParser(FakeTokenizer())
    output = (
        TOOL_START_TOKEN
        + json.dumps({"name": "get_weather", "arguments": {"city": "Paris"}})
        + TOOL_END_TOKEN
    )
    result = parser.extract_tool_calls(output, _chat_request())
    assert result.tools_called
    assert len(result.tool_calls) == 1
    call = result.tool_calls[0]
    assert call.function.name == "get_weather"
    assert json.loads(call.function.arguments) == {"city": "Paris"}


def test_tool_parser_tag_call():
    parser = Xing4_0ToolParser(FakeTokenizer())
    output = (
        TOOL_START_TOKEN
        + "get_weather\n<param_key>city</param_key>\n"
        + "<param_value>Paris</param_value>\n"
        + TOOL_END_TOKEN
    )
    result = parser.extract_tool_calls(output, _chat_request())
    assert result.tools_called
    call = result.tool_calls[0]
    assert call.function.name == "get_weather"
    # days is declared as integer in the tool schema, so it is deserialized.
    assert json.loads(call.function.arguments) == {"city": "Paris"}


def test_tool_parser_no_tool_call():
    parser = Xing4_0ToolParser(FakeTokenizer())
    result = parser.extract_tool_calls("plain answer", _chat_request())
    assert not result.tools_called
    assert result.content == "plain answer"


def _unknown_tool_output(name="delete_everything"):
    return (
        TOOL_START_TOKEN
        + json.dumps({"name": name, "arguments": {"city": "Paris"}})
        + TOOL_END_TOKEN
    )


def test_tool_parser_unknown_tool_skipped_by_default(monkeypatch):
    monkeypatch.delenv(FORWARD_UNKNOWN_TOOLS_ENV, raising=False)
    parser = Xing4_0ToolParser(FakeTokenizer())
    result = parser.extract_tool_calls(_unknown_tool_output(), _chat_request())
    assert not result.tools_called
    assert result.tool_calls == []


def test_tool_parser_unknown_tool_forwarded_when_enabled(monkeypatch):
    monkeypatch.setenv(FORWARD_UNKNOWN_TOOLS_ENV, "1")
    parser = Xing4_0ToolParser(FakeTokenizer())
    result = parser.extract_tool_calls(_unknown_tool_output(), _chat_request())
    assert result.tools_called
    assert len(result.tool_calls) == 1
    call = result.tool_calls[0]
    assert call.function.name == "delete_everything"
    assert json.loads(call.function.arguments) == {"city": "Paris"}


def test_tool_parser_empty_name_skipped_in_both_modes(monkeypatch):
    for enabled in (False, True):
        if enabled:
            monkeypatch.setenv(FORWARD_UNKNOWN_TOOLS_ENV, "1")
        else:
            monkeypatch.delenv(FORWARD_UNKNOWN_TOOLS_ENV, raising=False)
        parser = Xing4_0ToolParser(FakeTokenizer())
        output = (
            TOOL_START_TOKEN
            + json.dumps({"name": "", "arguments": {"city": "Paris"}})
            + TOOL_END_TOKEN
        )
        result = parser.extract_tool_calls(output, _chat_request())
        assert not result.tools_called
        assert result.tool_calls == []


def test_tool_parser_streaming_drains_skipped_unknown_tool(monkeypatch):
    monkeypatch.delenv(FORWARD_UNKNOWN_TOOLS_ENV, raising=False)
    parser = Xing4_0ToolParser(FakeTokenizer())
    output = "answer " + _unknown_tool_output() + "done"
    result = parser.extract_tool_calls_streaming(
        previous_text="",
        current_text=output,
        delta_text=output,
        previous_token_ids=[],
        current_token_ids=[],
        delta_token_ids=[],
        request=_chat_request(),
    )
    assert result is not None
    assert not result.tool_calls
    # The skipped call is drained, not leaked into the content.
    assert result.content == "answer done"


def test_tool_parser_streaming_drains_empty_name(monkeypatch):
    monkeypatch.setenv(FORWARD_UNKNOWN_TOOLS_ENV, "1")
    parser = Xing4_0ToolParser(FakeTokenizer())
    output = TOOL_START_TOKEN + json.dumps({"name": "", "arguments": {}}) + TOOL_END_TOKEN
    result = parser.extract_tool_calls_streaming(
        previous_text="",
        current_text=output,
        delta_text=output,
        previous_token_ids=[],
        current_token_ids=[],
        delta_token_ids=[],
        request=_chat_request(),
    )
    # Empty-name calls are drained even when the switch is on.
    assert result is None


def test_tool_parser_streaming_forwards_unknown_tool_when_enabled(monkeypatch):
    monkeypatch.setenv(FORWARD_UNKNOWN_TOOLS_ENV, "1")
    parser = Xing4_0ToolParser(FakeTokenizer())
    output = _unknown_tool_output()
    result = parser.extract_tool_calls_streaming(
        previous_text="",
        current_text=output,
        delta_text=output,
        previous_token_ids=[],
        current_token_ids=[],
        delta_token_ids=[],
        request=_chat_request(),
    )
    assert result is not None
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].function.name == "delete_everything"
    assert json.loads(result.tool_calls[0].function.arguments) == {"city": "Paris"}


def test_reasoning_parser_extraction():
    parser = Xing4_0ReasoningParser(FakeTokenizer())
    request = SimpleNamespace(chat_template_kwargs=None)
    reasoning, content = parser.extract_reasoning(
        "step by step" + END_THINK + "final answer", request
    )
    assert reasoning == "step by step"
    assert content == "final answer"


def test_reasoning_parser_thinking_disabled():
    parser = Xing4_0ReasoningParser(FakeTokenizer())
    request = SimpleNamespace(chat_template_kwargs={"enable_thinking": False})
    reasoning, content = parser.extract_reasoning("final answer only", request)
    assert reasoning is None
    assert content == "final answer only"


def test_reasoning_parser_thinking_enabled_without_end_marker():
    parser = Xing4_0ReasoningParser(FakeTokenizer())
    request = SimpleNamespace(chat_template_kwargs=None)
    reasoning, content = parser.extract_reasoning("still thinking...", request)
    assert reasoning == "still thinking..."
    assert content is None
