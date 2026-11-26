# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""No-GPU smoke tests for the Xing4.1-Preview compatibility module.

Run:  python -m pytest tests/test_xing4_1_preview.py -q

The tests cover the compatibility layers contributed by
``vllm_xing4_0_plugin.xing4_1_preview`` (model-type recognition, reasoning /
tool-call parser registration, MLA recognition, MTP draft remapping,
query-only RoPE math). The model implementation itself comes from vLLM's
in-tree ``DeepseekV32ForCausalLM``; the parsers reuse the vendored Xing4_0
implementations (Xing4.1 emits the same sentinel-tagged formats).
"""

from types import SimpleNamespace

import pytest

import vllm_xing4_0_plugin.xing4_1_preview as xing4_1
from vllm_xing4_0_plugin.xing4_1_preview import (
    PARSER_ALIAS,
    PARSER_NAME,
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


@pytest.fixture(autouse=True)
def _register_plugin():
    # Idempotent: a second registration must not double-wrap.
    xing4_1.register()
    xing4_1.register()


def test_entry_point_discoverable():
    from importlib.metadata import entry_points

    eps = entry_points(group="vllm.general_plugins")
    matches = [e for e in eps if e.name == "xing4_1_preview"]
    assert matches, "no 'xing4_1_preview' entry point in vllm.general_plugins"
    assert matches[0].value == "vllm_xing4_0_plugin.xing4_1_preview:register"


def test_config_registry_includes_xing41_preview():
    from transformers import DeepseekV3Config
    from vllm.transformers_utils.config import _CONFIG_REGISTRY

    assert "Xing4.1-Preview" in _CONFIG_REGISTRY
    assert _CONFIG_REGISTRY["Xing4.1-Preview"] is DeepseekV3Config


def test_parsers_registered_under_both_names():
    """Both parser names must exist with only this entry point allowlisted.

    ``VLLM_PLUGINS=xing4_1_preview`` keeps ``--reasoning-parser xing4_0``
    and ``--tool-call-parser xing4_0`` valid while also offering the
    ``xing4_1_preview`` name.
    """
    from vllm.reasoning.abs_reasoning_parsers import ReasoningParserManager
    from vllm.tool_parsers.abstract_tool_parser import ToolParserManager

    for name in (PARSER_NAME, PARSER_ALIAS):
        assert ReasoningParserManager.lazy_parsers[name] == (
            "vllm_xing4_0_plugin.xing4_0_reasoning_parser",
            "Xing4_0ReasoningParser",
        ), name
        assert ToolParserManager.lazy_parsers[name] == (
            "vllm_xing4_0_plugin.xing4_0_tool_parser",
            "Xing4_0ToolParser",
        ), name


def _convertor(model_type, **text_attrs):
    from vllm.transformers_utils.model_arch_config_convertor import (
        ModelArchConfigConvertorBase,
    )

    hf_text_config = _FakeHfConfig(model_type, **text_attrs)
    return ModelArchConfigConvertorBase(hf_text_config, hf_text_config)


def test_is_deepseek_mla_true_for_xing41_preview():
    assert _convertor("Xing4.1-Preview", kv_lora_rank=512).is_deepseek_mla()
    assert not _convertor("Xing4.1-Preview").is_deepseek_mla()  # no kv_lora_rank


def test_is_deepseek_mla_unchanged_for_other_models():
    assert not _convertor("llama").is_deepseek_mla()
    assert _convertor("deepseek_v3", kv_lora_rank=512).is_deepseek_mla()


def test_speculative_override_maps_xing41_preview_mtp():
    from vllm.config.speculative import SpeculativeConfig

    cfg = _FakeHfConfig(
        "Xing4.1-Preview",
        architectures=["DeepseekV32ForCausalLM"],
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


def test_speculative_override_leaves_xing4_0_alone():
    """The sibling xing4_0 module owns that model_type remap."""
    from vllm.config.speculative import SpeculativeConfig

    cfg = _FakeHfConfig("xing4_0", architectures=["Xing4_0ForCausalLM"])
    # When only this module's patch is installed, xing4_0 configs are passed
    # through to vLLM's original override untouched.
    out = SpeculativeConfig.hf_config_override(cfg)
    assert out.model_type == "xing4_0"


def test_extra_model_types_env_extends_aliases(monkeypatch):
    monkeypatch.setenv("XING4_PLUGIN_MODEL_TYPES", "Xing4.2-Preview, foo")

    assert xing4_1._known_model_types() >= {
        "Xing4.1-Preview",
        "Xing4.2-Preview",
        "foo",
    }
    # Env-injected aliases are honored without re-registering the plugin.
    assert _convertor("Xing4.2-Preview", kv_lora_rank=512).is_deepseek_mla()
    # Unknown names still fall through to vLLM's original logic.
    assert not _convertor("bar", kv_lora_rank=512).is_deepseek_mla()


def _rope_embedding(head_size, rotary_dim, max_positions, neox_style=True):
    """A fake DeepseekScalingRotaryEmbedding with a cosine/sine cache only.

    Only the attributes used by the query-only branch (``cos_sin_cache``,
    ``is_neox_style``, ``rotary_dim``) are provided, so the tests do not
    depend on vLLM's embedding constructor signature.
    """
    import torch

    return SimpleNamespace(
        head_size=head_size,
        use_flashinfer=False,  # stay on the torch reference branch
        # cos_sin_cache holds cos||sin concatenated; each half is
        # rotary_dim / 2 wide, so the last dimension is rotary_dim.
        cos_sin_cache=torch.randn(max_positions, rotary_dim),
        is_neox_style=neox_style,
        rotary_dim=rotary_dim,
    )


def test_query_only_rope_patch_registered():
    from vllm.model_executor.layers.rotary_embedding.deepseek_scaling_rope import (
        DeepseekScalingRotaryEmbedding,
    )

    current = DeepseekScalingRotaryEmbedding.__dict__["forward_cuda"]
    assert getattr(current, "_xing4_1_patch", False)


def test_query_only_rope_rotates_without_key():
    """The wrapper must rotate the query when key is None (DSA indexer)."""
    import torch

    head_size, rotary_dim, max_pos = 8, 6, 16
    nt, nhead = 4, 2
    positions = torch.arange(nt)
    query = torch.randn(nt, nhead, head_size)
    fake = _rope_embedding(head_size, rotary_dim, max_pos, neox_style=True)

    q_rot, key = xing4_1._query_only_rope_forward_cuda(fake, positions, query, None)

    assert key is None
    assert q_rot.shape == query.shape

    # Independent reference: DeepseekScalingRotaryEmbedding.forward_static
    # math (first rotary_dim lanes, neox half-split rotation).
    cs = fake.cos_sin_cache[positions]
    cos, sin = cs.chunk(2, dim=-1)
    cos = torch.cat((cos, cos), dim=-1).unsqueeze(-2)
    sin = torch.cat((sin, sin), dim=-1).unsqueeze(-2)
    d2 = rotary_dim // 2
    x1 = query[..., :d2]
    x2 = query[..., d2:rotary_dim]
    ref_rot = torch.cat(
        (
            x1 * cos[..., :d2] - x2 * sin[..., :d2],
            x2 * cos[..., d2:] + x1 * sin[..., d2:],
        ),
        dim=-1,
    )
    expected = torch.cat((ref_rot, query[..., rotary_dim:]), dim=-1)
    assert torch.allclose(q_rot, expected)


def test_query_only_rope_delegates_when_key_present():
    """A real key is handed to the installed forward_cuda unchanged."""
    import torch

    head_size, rotary_dim, max_pos = 8, 6, 16
    nt, nhead = 4, 2
    positions = torch.arange(nt)
    query = torch.randn(nt, nhead, head_size)
    key = torch.randn(nt, nhead, head_size)
    fake = _rope_embedding(head_size, rotary_dim, max_pos, neox_style=True)

    calls = []

    def orig(self, positions_, query_, key_, offsets_):
        calls.append((self, offsets_))
        return query_, key_

    saved = xing4_1._ORIG_ROPE_FORWARD_CUDA
    xing4_1._ORIG_ROPE_FORWARD_CUDA = orig
    try:
        q_rot, k_rot = xing4_1._query_only_rope_forward_cuda(
            fake, positions, query, key
        )
    finally:
        xing4_1._ORIG_ROPE_FORWARD_CUDA = saved

    assert calls == [(fake, None)]
    assert q_rot is query and k_rot is key
