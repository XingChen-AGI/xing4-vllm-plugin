# SPDX-License-Identifier: Apache-2.0
"""Out-of-tree XingChen Xing4_0 support for vLLM.

The entry point imports only the standard library. vLLM loads general plugins
in the API process, engine core, and worker processes, so registration must
stay cheap, lazy, and safe to call more than once.
"""

from __future__ import annotations

import logging
import os

__all__ = [
    "ARCHITECTURE",
    "MODEL_TARGET",
    "OVERRIDE_ENV",
    "PARSER_NAME",
    "TESTED_VLLM",
    "register",
]

logger = logging.getLogger(__name__)

ARCHITECTURE = "Xing4_0ForCausalLM"
PARSER_NAME = "xing4_0"
MODEL_TARGET = f"{__name__}.xing4_0:{ARCHITECTURE}"
TESTED_VLLM = "2e09247c2d7b6b97d13af6e71a85bf8d1271deb6"
OVERRIDE_ENV = "XING4_0_PLUGIN_OVERRIDE"

_TRUE = frozenset({"1", "true", "yes", "on"})

# Originals captured by _patch_speculative_config / _patch_model_arch_convertor.
# Kept in module globals so the patched wrappers below stay picklable by
# reference (the draft-model hf_overrides callable is sent to spawned
# engine-core processes) and still work if unpickled before register() ran.
_ORIG_HF_CONFIG_OVERRIDE = None
_ORIG_IS_DEEPSEEK_MLA = None


def _override_requested() -> bool:
    """Return whether the operator explicitly requested an override."""
    return os.environ.get(OVERRIDE_ENV, "").strip().lower() in _TRUE


def _register_config() -> None:
    """Expose Xing4_0 through vLLM's model-type registry.

    Xing4_0 checkpoints are DeepSeek-V3 style configs (MLA attention + MoE),
    so the existing ``DeepseekV3Config`` (from HuggingFace Transformers) is
    reused. The value is a class name resolved lazily by ``LazyConfigDict``.
    """
    from vllm.transformers_utils.config import _CONFIG_REGISTRY

    _CONFIG_REGISTRY[PARSER_NAME] = "DeepseekV3Config"


def _register_reasoning_parser() -> None:
    """Register the Xing4_0 reasoning parser without importing it until use."""
    from vllm.reasoning.abs_reasoning_parsers import ReasoningParserManager

    ReasoningParserManager.register_lazy_module(
        name=PARSER_NAME,
        module_path=f"{__name__}.xing4_0_reasoning_parser",
        class_name="Xing4_0ReasoningParser",
    )


def _register_tool_parser() -> None:
    """Register the Xing4_0 tool parser without importing it until use."""
    from vllm.tool_parsers.abstract_tool_parser import ToolParserManager

    # If an upstream parser was already resolved before this entry point ran,
    # remove the eager cache so an explicit plugin override takes effect too.
    ToolParserManager.tool_parsers.pop(PARSER_NAME, None)
    ToolParserManager.register_lazy_module(
        name=PARSER_NAME,
        module_path=f"{__name__}.xing4_0_tool_parser",
        class_name="Xing4_0ToolParser",
    )


def hf_config_override_xing4_0(hf_config, *args, **kwargs):
    """Wrap ``SpeculativeConfig.hf_config_override`` for Xing4_0 MTP drafts.

    Xing4_0 MTP draft checkpoints keep ``model_type: "xing4_0"``; remap them
    to the DeepSeek MTP path and delegate, so vLLM derives ``n_predict`` and
    the ``DeepSeekMTPModel`` architecture as for DeepSeek-V3.

    Newer vLLM (e.g. 0.34.x) passes a second ``transformers_mtp_overrides``
    argument, either positionally or as a keyword (and may bind it with
    ``functools.partial``). Accept and forward it verbatim; when it is set,
    vLLM's own override applies those keys and returns early, so the
    ``xing4_0`` -> ``deepseek_mtp`` remap must be skipped.
    """
    from vllm.config.speculative import SpeculativeConfig

    mtp_overrides = kwargs.get("transformers_mtp_overrides", None)
    if mtp_overrides is None and args:
        mtp_overrides = args[0]
    if mtp_overrides is None and getattr(hf_config, "model_type", None) == PARSER_NAME:
        hf_config.model_type = "deepseek_mtp"

    orig = _ORIG_HF_CONFIG_OVERRIDE
    if orig is None:
        # register() has not run in this process (e.g. the override callable
        # was unpickled in a fresh subprocess); vLLM's own override is still
        # in place, so resolve it from the class dict.
        orig = SpeculativeConfig.__dict__["hf_config_override"]
        orig = getattr(orig, "__func__", orig)
    if args:
        return orig(hf_config, *args)
    if kwargs:
        # Forward positionally: the parameter name is stable today but the
        # wrapper should not depend on it.
        return orig(hf_config, kwargs.get("transformers_mtp_overrides"))
    return orig(hf_config)


hf_config_override_xing4_0._xing4_0_patch = True


def is_deepseek_mla_xing4_0(self):
    """Wrap ``ModelArchConfigConvertorBase.is_deepseek_mla`` for Xing4_0.

    Without this, the convertor miscomputes ``head_size`` (which must be
    ``kv_lora_rank + qk_rope_head_dim`` for MLA) and ``ModelConfig.use_mla``
    stays False, disabling the MLA attention and KV-cache paths.
    """
    from vllm.transformers_utils.model_arch_config_convertor import (
        ModelArchConfigConvertorBase,
    )

    if getattr(self.hf_text_config, "model_type", None) == PARSER_NAME:
        # Xing4_0 is a DeepSeek-V3 style MLA model without compress_ratios.
        return getattr(self.hf_text_config, "kv_lora_rank", None) is not None

    orig = _ORIG_IS_DEEPSEEK_MLA
    if orig is None:
        orig = ModelArchConfigConvertorBase.__dict__["is_deepseek_mla"]
    return orig(self)


is_deepseek_mla_xing4_0._xing4_0_patch = True


def _patch_speculative_config() -> None:
    """Teach SpeculativeConfig about Xing4_0 MTP draft configs."""
    global _ORIG_HF_CONFIG_OVERRIDE
    from vllm.config.speculative import SpeculativeConfig

    current = SpeculativeConfig.__dict__.get("hf_config_override")
    current = getattr(current, "__func__", current)  # unwrap staticmethod
    if getattr(current, "_xing4_0_patch", False):
        return  # already patched; keep the first original
    _ORIG_HF_CONFIG_OVERRIDE = current
    SpeculativeConfig.hf_config_override = staticmethod(hf_config_override_xing4_0)


def _patch_model_arch_convertor() -> None:
    """Teach the model-arch convertor that Xing4_0 uses DeepSeek-style MLA."""
    global _ORIG_IS_DEEPSEEK_MLA
    from vllm.transformers_utils.model_arch_config_convertor import (
        ModelArchConfigConvertorBase,
    )

    current = ModelArchConfigConvertorBase.__dict__.get("is_deepseek_mla")
    if getattr(current, "_xing4_0_patch", False):
        return  # already patched; keep the first original
    _ORIG_IS_DEEPSEEK_MLA = current
    ModelArchConfigConvertorBase.is_deepseek_mla = is_deepseek_mla_xing4_0


def register() -> None:
    """Register Xing4_0 with vLLM's ``vllm.general_plugins`` entry point."""
    import vllm
    from vllm import ModelRegistry

    if (
        ARCHITECTURE in ModelRegistry.get_supported_archs()
        and not _override_requested()
    ):
        logger.info(
            "vllm-xing4_0-plugin: %s is already supported in tree; standing "
            "down. Set %s=1 to override.",
            ARCHITECTURE,
            OVERRIDE_ENV,
        )
        return

    _register_config()
    _register_reasoning_parser()
    _register_tool_parser()
    _patch_speculative_config()
    _patch_model_arch_convertor()
    ModelRegistry.register_model(ARCHITECTURE, MODEL_TARGET)
    logger.info(
        "vllm-xing4_0-plugin: registered %s -> %s "
        "(tested against vLLM commit %s, running %s)",
        ARCHITECTURE,
        MODEL_TARGET,
        TESTED_VLLM,
        vllm.__version__,
    )
