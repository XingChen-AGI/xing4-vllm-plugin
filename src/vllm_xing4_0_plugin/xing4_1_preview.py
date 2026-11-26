# SPDX-License-Identifier: Apache-2.0
"""Out-of-tree Xing4.1-Preview compatibility for vLLM.

Xing4.1-Preview checkpoints are DeepSeek-V3.2 style models (MLA attention +
DSA indexer + MoE + optional MTP) that ship a custom ``model_type``
(``"Xing4.1-Preview"``) in config.json while reusing the in-tree
``DeepseekV32ForCausalLM`` architecture. The model implementation therefore
comes from vLLM itself; this module contributes the compatibility layers,
registered through its own ``vllm.general_plugins`` entry point so it works
standalone (``VLLM_PLUGINS=xing4_1_preview``) or alongside the Xing4_0
support above:

- ``model_type`` recognition (``_CONFIG_REGISTRY`` entries);
- ``xing4_1_preview`` and ``xing4_0`` names for the Xing-family reasoning
  and tool-call parsers (reuse of the vendored Xing4_0 parsers; Xing4.1
  emits the same sentinel-tagged formats);
- MLA recognition in the model-arch convertor (``ModelConfig.use_mla``);
- MTP draft remapping in ``SpeculativeConfig.hf_config_override``;
- query-only RoPE support on ``DeepseekScalingRotaryEmbedding.forward_cuda``
  for the DSA indexer path.

The module imports only the standard library at import time. vLLM loads
general plugins in the API process, engine core, and worker processes, so
registration must stay cheap, lazy, and safe to call more than once. All
monkey-patches chain onto whatever is installed (including the Xing4_0
patches from this same package) and are idempotent per module.
"""

from __future__ import annotations

import logging
import os

__all__ = [
    "MODEL_TYPE",
    "PARSER_ALIAS",
    "PARSER_NAME",
    "TESTED_VLLM",
    "register",
]

logger = logging.getLogger(__name__)

MODEL_TYPE = "Xing4.1-Preview"
TESTED_VLLM = "2e09247c2d7b6b97d13af6e71a85bf8d1271deb6"

# The Xing-family vendored parsers live in the sibling modules of the same
# package. Xing4.1-Preview emits the same sentinel-tagged reasoning and
# tool-call formats as Xing4_0, so the implementations are reused directly
# instead of being re-vendored.
_TOOL_PARSER_MODULE = "vllm_xing4_0_plugin.xing4_0_tool_parser"
_TOOL_PARSER_CLASS = "Xing4_0ToolParser"
_REASONING_PARSER_MODULE = "vllm_xing4_0_plugin.xing4_0_reasoning_parser"
_REASONING_PARSER_CLASS = "Xing4_0ReasoningParser"

# Parser registry names. ``xing4_0`` keeps existing serving configs (e.g.
# ``--tool-call-parser xing4_0``) working when only this entry point is
# allowlisted; ``xing4_1_preview`` is the forward-looking name.
PARSER_NAME = "xing4_1_preview"
PARSER_ALIAS = "xing4_0"

# Further Xing4-family ``model_type`` aliases (comma/space separated) can be
# added at runtime through ``XING4_PLUGIN_MODEL_TYPES`` without changing the
# plugin. Aliases resolve at each use, so new names take effect even without
# re-registering the plugin (except ``_CONFIG_REGISTRY`` entries, which are
# written once per process at registration time). ``xing4_0`` itself is
# owned by ``vllm_xing4_0_plugin.__init__`` and is deliberately not listed
# here.
_EXTRA_MODEL_TYPES_ENV = "XING4_PLUGIN_MODEL_TYPES"

_TRUE = frozenset({"1", "true", "yes", "on"})

# Originals captured at registration time. Kept in module globals so the
# patched wrappers below stay picklable by reference (the draft-model
# hf_overrides callable is sent to spawned engine-core processes) and still
# work if unpickled before register() ran.
_ORIG_HF_CONFIG_OVERRIDE = None
_ORIG_IS_DEEPSEEK_MLA = None
_ORIG_ROPE_FORWARD_CUDA = None


def _known_model_types() -> "frozenset[str]":
    """Model types recognized as Xing4.1-style checkpoints."""
    extras = os.environ.get(_EXTRA_MODEL_TYPES_ENV, "")
    extra_types = {t.strip() for t in extras.replace(",", " ").split() if t.strip()}
    return frozenset({MODEL_TYPE, *extra_types})


def _register_config() -> None:
    """Expose Xing4.1-Preview through vLLM's model-type registry.

    Xing4.1-Preview checkpoints are DeepSeek-V3 style configs (MLA attention
    + MoE), so the existing ``DeepseekV3Config`` (from HuggingFace
    Transformers) is reused, mirroring the Xing4_0 registration. The value
    is a class name resolved lazily by ``LazyConfigDict``.
    """
    from vllm.transformers_utils.config import _CONFIG_REGISTRY

    for model_type in _known_model_types():
        _CONFIG_REGISTRY[model_type] = "DeepseekV3Config"


def _register_reasoning_parser() -> None:
    """Register the Xing-family reasoning parser under both names.

    Registers lazily (the vendored parser is only imported when a request
    actually uses it), mirroring the Xing4_0 registration so the same
    ``--reasoning-parser`` value works whether Xing4_0 support is loaded
    or this module is allowlisted on its own.
    """
    from vllm.reasoning.abs_reasoning_parsers import ReasoningParserManager

    for name in (PARSER_NAME, PARSER_ALIAS):
        ReasoningParserManager.register_lazy_module(
            name=name,
            module_path=_REASONING_PARSER_MODULE,
            class_name=_REASONING_PARSER_CLASS,
        )


def _register_tool_parser() -> None:
    """Register the Xing-family tool-call parser under both names.

    Registers lazily like the Xing4_0 entry point; Xing4.1-Preview emits
    the same sentinel-tagged tool-call formats. If an upstream parser was
    already resolved (eager cache) under either name, drop it so this lazy
    registration takes effect.
    """
    from vllm.tool_parsers.abstract_tool_parser import ToolParserManager

    for name in (PARSER_NAME, PARSER_ALIAS):
        ToolParserManager.tool_parsers.pop(name, None)
        ToolParserManager.register_lazy_module(
            name=name,
            module_path=_TOOL_PARSER_MODULE,
            class_name=_TOOL_PARSER_CLASS,
        )


def hf_config_override_xing4_1(hf_config, *args, **kwargs):
    """Wrap ``SpeculativeConfig.hf_config_override`` for Xing4.1 MTP drafts.

    Xing4.1 MTP draft checkpoints keep ``model_type: "Xing4.1-Preview"``;
    remap them to the DeepSeek MTP path and delegate, so vLLM derives
    ``n_predict`` and the ``DeepSeekMTPModel`` architecture as for
    DeepSeek-V3.

    Newer vLLM (e.g. 0.34.x) passes a second ``transformers_mtp_overrides``
    argument, either positionally or as a keyword (and may bind it with
    ``functools.partial``). Accept and forward it verbatim; when it is set,
    vLLM's own override applies those keys and returns early, so the
    ``Xing4.1-Preview`` -> ``deepseek_mtp`` remap must be skipped. Other
    model types (including ``xing4_0``, owned by the sibling module) are
    passed through untouched.
    """
    from vllm.config.speculative import SpeculativeConfig

    mtp_overrides = kwargs.get("transformers_mtp_overrides", None)
    if mtp_overrides is None and args:
        mtp_overrides = args[0]
    if mtp_overrides is None and (
        getattr(hf_config, "model_type", None) in _known_model_types()
    ):
        hf_config.model_type = "deepseek_mtp"

    orig = _ORIG_HF_CONFIG_OVERRIDE
    if orig is None:
        # register() has not run in this process (e.g. the override callable
        # was unpickled in a fresh subprocess); whatever is installed on the
        # class is the correct next hop.
        orig = SpeculativeConfig.__dict__["hf_config_override"]
        orig = getattr(orig, "__func__", orig)
    if args:
        return orig(hf_config, *args)
    if kwargs:
        # Forward positionally: the parameter name is stable today but the
        # wrapper should not depend on it.
        return orig(hf_config, kwargs.get("transformers_mtp_overrides"))
    return orig(hf_config)


hf_config_override_xing4_1._xing4_1_patch = True


def is_deepseek_mla_xing4_1(self):
    """Wrap ``ModelArchConfigConvertorBase.is_deepseek_mla`` for Xing4.1.

    ``"Xing4.1-Preview"`` is not in vLLM's model_type allowlist, so without
    this patch ``ModelConfig.use_mla`` stays False and both the MLA
    attention and the DSA indexer path stay disabled (the convertor would
    also miscompute ``head_size``, which must be ``kv_lora_rank +
    qk_rope_head_dim`` for MLA). Xing4.1 is a DeepSeek-V3.2 style MLA model
    without ``compress_ratios``. Other model types fall through to whatever
    was installed before (the sibling Xing4_0 patch or vLLM's original).
    """
    from vllm.transformers_utils.model_arch_config_convertor import (
        ModelArchConfigConvertorBase,
    )

    if getattr(self.hf_text_config, "model_type", None) in _known_model_types():
        return getattr(self.hf_text_config, "kv_lora_rank", None) is not None

    orig = _ORIG_IS_DEEPSEEK_MLA
    if orig is None:
        orig = ModelArchConfigConvertorBase.__dict__["is_deepseek_mla"]
    return orig(self)


is_deepseek_mla_xing4_1._xing4_1_patch = True


def _query_only_rope_forward_cuda(self, positions, query, key=None, offsets=None):
    """Wrap ``DeepseekScalingRotaryEmbedding.forward_cuda`` for query-only RoPE.

    vLLM's DSA indexer path rotates only the query tensor and calls
    ``rotary_emb(positions, q_pe)`` with ``key=None``. ``forward_cuda``
    forwards that straight into ``flashinfer_rotary_embedding`` (which
    unconditionally does ``key.view(...)``), or falls back to
    ``forward_native`` (whose first statement is ``assert key is not
    None``), so the call crashes with::

        AttributeError: 'NoneType' object has no attribute 'view'

    When ``key`` is absent, the query is rotated with exactly the reference
    math of ``DeepseekScalingRotaryEmbedding.forward_static`` (first
    ``rotary_dim`` lanes, ``rotate_neox`` / ``rotate_gptj`` per style,
    pass-through remainder when ``rotary_dim < head_size``) and the key half
    is returned as ``None``; otherwise the installed ``forward_cuda`` runs.

    This class is selected for every MLA/DSA model whose ``rope_parameters``
    lacks ``is_deepseek_v4`` (e.g. model_type ``deepseek_v32`` and Xing4.1);
    native query-only support exists only on
    ``DeepseekV4ScalingRotaryEmbedding``, which rotates the *last*
    ``rotary_dim`` lanes instead and is therefore not a drop-in upgrade for
    DeepSeek-V3.2 style backbones.
    """
    if key is not None:
        orig = _ORIG_ROPE_FORWARD_CUDA
        if orig is None:
            # register() has not run in this process; whatever is installed
            # on the class is the correct next hop.
            from vllm.model_executor.layers.rotary_embedding.deepseek_scaling_rope import (
                DeepseekScalingRotaryEmbedding,
            )

            orig = DeepseekScalingRotaryEmbedding.__dict__["forward_cuda"]
            orig = getattr(orig, "__func__", orig)
        return orig(self, positions, query, key, offsets)

    import torch

    from vllm.model_executor.layers.rotary_embedding.common import (
        rotate_gptj,
        rotate_neox,
    )

    h = query.shape[-1]
    q = query.reshape(positions.shape[0], -1, h)
    pose = torch.add(positions, offsets) if offsets is not None else positions
    cs = self.cos_sin_cache[pose]
    cos, sin = cs.chunk(2, dim=-1)
    if self.is_neox_style:
        cos = torch.cat((cos, cos), dim=-1).unsqueeze(-2).to(query.dtype)
        sin = torch.cat((sin, sin), dim=-1).unsqueeze(-2).to(query.dtype)
        rfn = rotate_neox
    else:
        cos = cos.repeat_interleave(2, dim=-1).unsqueeze(-2).to(query.dtype)
        sin = sin.repeat_interleave(2, dim=-1).unsqueeze(-2).to(query.dtype)
        rfn = rotate_gptj
    rd = self.rotary_dim
    qr = q[..., :rd] * cos + rfn(q[..., :rd]) * sin
    q = torch.cat((qr, q[..., rd:]), dim=-1) if rd < h else qr
    return q.reshape_as(query), None


_query_only_rope_forward_cuda._xing4_1_patch = True


def _patch_speculative_config() -> None:
    """Teach SpeculativeConfig about Xing4.1 MTP draft configs."""
    global _ORIG_HF_CONFIG_OVERRIDE
    from vllm.config.speculative import SpeculativeConfig

    current = SpeculativeConfig.__dict__.get("hf_config_override")
    current = getattr(current, "__func__", current)  # unwrap staticmethod
    if getattr(current, "_xing4_1_patch", False):
        return  # already patched by this module; keep the first original
    _ORIG_HF_CONFIG_OVERRIDE = current
    SpeculativeConfig.hf_config_override = staticmethod(hf_config_override_xing4_1)


def _patch_model_arch_convertor() -> None:
    """Teach the model-arch convertor that Xing4.1 uses DeepSeek MLA."""
    global _ORIG_IS_DEEPSEEK_MLA
    from vllm.transformers_utils.model_arch_config_convertor import (
        ModelArchConfigConvertorBase,
    )

    current = ModelArchConfigConvertorBase.__dict__.get("is_deepseek_mla")
    if getattr(current, "_xing4_1_patch", False):
        return  # already patched by this module; keep the first original
    _ORIG_IS_DEEPSEEK_MLA = current
    ModelArchConfigConvertorBase.is_deepseek_mla = is_deepseek_mla_xing4_1


def _patch_deepseek_rope_forward() -> None:
    """Teach DeepseekScalingRotaryEmbedding to rotate a query-only tensor."""
    global _ORIG_ROPE_FORWARD_CUDA
    from vllm.model_executor.layers.rotary_embedding.deepseek_scaling_rope import (
        DeepseekScalingRotaryEmbedding,
    )

    current = DeepseekScalingRotaryEmbedding.__dict__.get("forward_cuda")
    if getattr(current, "_xing4_1_patch", False):
        return  # already patched by this module; keep the first original
    _ORIG_ROPE_FORWARD_CUDA = current
    DeepseekScalingRotaryEmbedding.forward_cuda = _query_only_rope_forward_cuda


def register() -> None:
    """Register Xing4.1-Preview with vLLM's ``vllm.general_plugins``."""
    import vllm

    _register_config()
    _register_reasoning_parser()
    _register_tool_parser()
    _patch_speculative_config()
    _patch_model_arch_convertor()
    _patch_deepseek_rope_forward()
    logger.info(
        "vllm-xing4_1_preview-plugin: registered support for model_type %s "
        "(tested against vLLM commit %s, running %s, model types: %s, "
        "parsers: %s)",
        MODEL_TYPE,
        TESTED_VLLM,
        vllm.__version__,
        ", ".join(sorted(_known_model_types())),
        f"{PARSER_NAME}/{PARSER_ALIAS}",
    )
