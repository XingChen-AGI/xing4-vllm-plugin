# vLLM Xing4_0 Plugin

> This document targets Linux systems and Bash-compatible shells.

This package provides XingChen-Xing4_0 (e.g. `XingChen-AGI/Xing4.0-29B-A4B`)
support for vLLM as an out-of-tree general plugin. It enables loading and
serving Xing4_0 models without modifying the vLLM source tree. The plugin
registers:

- `Xing4_0ForCausalLM`, the Xing4_0 model implementation;
- the `xing4_0` Transformers configuration (`model_type: "xing4_0"`), reusing
  vLLM's existing MLA-attention + MoE config infrastructure;
- the `xing4_0` reasoning parser (think-block extraction);
- the `xing4_0` tool-call parser (JSON, name+JSON and tag-based calls);
- MTP speculative decoding support for `xing4_0` draft checkpoints, and MLA
  recognition in the model-arch convertor.

The implementation is vendored from a vLLM fork based on upstream commit
`2e09247c2d7b6b97d13af6e71a85bf8d1271deb6` and uses vLLM's existing runtime
layers (including `vllm.model_executor.layers.mhc`). It does not contain
custom CUDA kernels or require a separate build.

## Xing4_0 model capabilities

Xing4_0 features MLA (Multi-head Latent Attention), a MoE (Mixture-of-Experts)
block, an optional DSA indexer, and replaces the standard residual connection
with
Manifold-constrained Hyper-Connections (mHC): the residual stream is expanded
into `hc_mult` parallel streams that are mixed by input-dependent,
doubly-stochastic matrices produced by a Sinkhorn-Knopp projection.

Notes:

- mHC mode (`hc_mult > 1`) requires `bfloat16` and does not support pipeline
  parallelism yet; the plugin raises a clear error otherwise.
- Both DSA (`index_topk`/`index_n_heads` present) and non-DSA variants are
  supported.
- Older configs describing RoPE through `rope_scaling` are converted
  automatically.

## Install

Install vLLM first, then install this package into the **same Python
environment**. The editable install is important because vLLM discovers the
plugin through the `vllm.general_plugins` entry-point metadata.

```bash
pip install -e . --no-deps
```

`--no-deps` prevents the plugin from replacing the vLLM installation already
selected for the environment. Verify that the entry point is visible to the
interpreter used to launch vLLM:

```bash
python -c "from importlib.metadata import entry_points; \
print([(e.name, e.value) for e in entry_points(group='vllm.general_plugins')])"
```

The output should include:

```
('xing4_0', 'vllm_xing4_0_plugin:register')
```

Adding `src/` to `PYTHONPATH` alone is **not** sufficient for normal plugin
discovery because entry points come from installed package metadata.

## Usage

Start the OpenAI-compatible server:

```bash
vllm serve /path/to/xing4_0-model \
  --trust-remote-code
```

Reasoning extraction (think blocks):

```bash
vllm serve /path/to/xing4_0-model \
  --trust-remote-code \
  --reasoning-parser xing4_0
```

Tool calling:

```bash
vllm serve /path/to/xing4_0-model \
  --trust-remote-code \
  --enable-auto-tool-choice \
  --tool-call-parser xing4_0 \
  --chat-template /path/to/chat_template.jinja
```

MTP speculative decoding (when the checkpoint ships MTP layers).
`deepseek_mtp` is vLLM's internal method name for MTP-style draft models;
Xing4_0 uses the same MTP architecture, so this identifier is required:

```bash
vllm serve /path/to/xing4_0-model \
  --trust-remote-code \
  --speculative-config '{"method": "deepseek_mtp", "model": "/path/to/xing4_0-model"}'
```

When the server starts, look for the plugin's registration log:

```
vllm-xing4_0-plugin: registered Xing4_0ForCausalLM -> ...
```

If vLLM already provides `Xing4_0ForCausalLM`, the plugin logs a stand-down
message and leaves the in-tree implementation active. To explicitly use this
vendored implementation, set:

```bash
XING4_0_PLUGIN_OVERRIDE=1 vllm serve /path/to/xing4_0-model ...
```

## Reasoning format

The chat template injects the think-start marker into the prompt via
`add_generation_prompt`; the model itself does not emit it again. Generated
output therefore looks like:

- thinking enabled: `<think>reasoning` ... `</think>` `content`
- thinking disabled (via `extra_body={"chat_template_kwargs": {"enable_thinking": False}}`): pure content, no markers.

## Tool-call format

Tool calls are emitted inside the model's tool-call sentinel tags
(`<tool_call>` ... `</tool_call>`, exposed as `TOOL_START_TOKEN` /
`TOOL_END_TOKEN` in `vllm_xing4_0_plugin.xing4_0_tool_parser`). Three payload
forms are recognized:

1. JSON: `{"name": "func", "arguments": {...}}`
2. name + JSON arguments: `func{"arguments": {...}}`
3. tag-based:

```
func
<param_key>arg1</param_key>
<param_value>value1</param_value>
...
```

Values declared as integer/number/boolean/JSON in the tool schema are
converted to their JSON types; string-typed values are kept verbatim.

## Plugin loading and allowlists

General plugins are loaded automatically by vLLM unless `VLLM_PLUGINS` is set
as an allowlist. If an allowlist is used, the entry-point name is `xing4_0`:

```bash
VLLM_PLUGINS=xing4_0 vllm serve /path/to/xing4_0-model ...
```

## Compatibility and maintenance

The model implementation depends on vLLM's internal runtime-layer APIs. The
`TESTED_VLLM` value in `src/vllm_xing4_0_plugin/__init__.py` records the vLLM
revision the implementation was vendored from. When upgrading vLLM, retest the
plugin and re-vendor the model files if an internal API has moved. The plugin
logs both the tested revision and the running vLLM version when it registers,
making version drift visible in the server log.

All vendored imports were re-audited against upstream `main` at
`f147949be8b9e576f80133bee01e717ee7c5db0e` (2026-10); the only drift found
(`make_tool_call_id` moving to `vllm.renderers.chat_utils`) is handled with a
dual-path import in `xing4_0_tool_parser.py`, so the plugin loads on both the
vendored base revision and current upstream `main` / nightly builds. Still
prefer running the smoke tests (`tests/test_smoke.py`) inside your target
image before serving production traffic with a brand-new nightly.

Two monkey-patches are applied at registration time (the only part without an
upstream registration API):

- `SpeculativeConfig.hf_config_override`: maps `xing4_0` MTP draft configs to
  vLLM's `deepseek_mtp` method path (vLLM's internal identifier for MTP-style
  draft models);
- `ModelArchConfigConvertorBase.is_deepseek_mla` (vLLM's internal MLA
  detection method): reports `True` for `xing4_0` configs carrying
  `kv_lora_rank`, enabling MLA attention and the derived `head_size`
  (`kv_lora_rank + qk_rope_head_dim`).

Both wrappers delegate to the originals for every other model type.

## Tests

The plugin smoke tests do not require a GPU or model weights. After installing
the plugin in the test environment, run:

```bash
python -m pytest tests/test_smoke.py -q
```

The tests verify the public registration entry point, the MTP/MLA patches,
and complete reasoning/tool-call extractions.

## Project layout

```
xing4-vllm-plugin/
+-- src/vllm_xing4_0_plugin/
|   +-- __init__.py                  # register() and patch entry point
|   +-- xing4_0.py                    # vendored Xing4_0 model
|   +-- xing4_0_reasoning_parser.py   # vendored reasoning parser
|   +-- xing4_0_tool_parser.py        # vendored tool-call parser
+-- tests/
|   +-- test_smoke.py                # registration and parser smoke tests
+-- pyproject.toml
```

## License

Apache-2.0. The vendored vLLM implementation retains its upstream copyright
headers.
