# Gemma4: Experimental Dense Text Inference

This is a bounded correction of Candle's existing Gemma4 implementation, not a
Gemma3 checkpoint conversion or a complete Gemma4 port. Published under `candle/`
in `ai10xdev/model`, it is based on
<https://github.com/huggingface/candle.git>, at
`c68b24997319b8d76e5a4e775dab197c7f26322a`, which already contains upstream Gemma4
modules and an example. This tree includes the experimental changes; the
Gemma3 implementation remains separate.

**Status (2026-10-18 snapshot, not current runtime state):** the real
`google/gemma-4-31B-it` checkpoint was installed on the hosted
H100; its dense text decoder has run all-layer Q4_0/BF16 LoRA training and trained
adapter inference. The [October 18 report](../../../../h100-server/context-sweep-20261018/README.md)
records synthetic input tests through **262142 input + 2 generated special
tokens** with empty decoded output: capacity evidence, not answer quality or
exact 262144-token input prefill. Full-output maxima remain incomplete in that
snapshot. Older results below are explicitly historical.

## Gemma2 To Gemma4 Trace

`candle-examples/examples/gemma/main.rs` selects Gemma2 2B/9B presets, deserializes
`gemma2::Config`, and builds `gemma2::Model::new` from safetensors via `VarBuilder`.
In `candle-transformers/src/models/gemma2.rs`, `Model::new` loads `model.*` and
ties the head to embeddings; `forward` scales embeddings, runs pre/post-RMSNorm
attention and gated-MLP residual branches, then returns softcapped last-token
logits. Its RMSNorm uses `1 + weight`, attention has separate Q/K/V with standard
RoPE and concatenated KV history, and one optional sliding mask is reused across
layers. Although config parses `query_pre_attn_scalar`, attention scales by
`1/sqrt(head_dim)`. Gemma3's separate path adds Q/K normalization, local/global
RoPE and per-layer rotating/global caches as traced below. Gemma4's separate
`gemma4/main.rs` and `gemma4/{config,text}.rs` implement explicit layer types,
proportional global RoPE, direct-weight RMSNorm, V normalization, global K=V and
layer scalars. This is not a model-ID substitution or Gemma2 checkpoint conversion.

## Gemma3 Trace

All code paths below are relative to the Candle workspace (`candle/` in the
`ai10xdev/model` repository, alongside `h100-server/`).

1. `candle-examples/examples/gemma/main.rs`: `Which::BaseV3_1B` and
   `Which::InstructV3_1B` select `google/gemma-3-1b-pt` and
   `google/gemma-3-1b-it`. `main` retrieves config, tokenizer and safetensors,
   deserializes `gemma3::Config`, creates a memory-mapped `VarBuilder`, and
   constructs `gemma3::Model::new`. This CLI's Gemma3 presets are the text-only
   1B models, not the entire multimodal Gemma3 family.
2. `candle-transformers/src/models/gemma3.rs`: `Config` carries dimensions,
   attention/KV head counts, scalar RoPE frequencies, sliding-window size and
   pattern, context limit, and optional softcaps. `Model::new` loads `model.*`,
   ties the output projection to the embeddings, and chooses global layers with
   `(layer_idx + 1) % sliding_window_pattern == 0`.
3. In that file, `Model::forward` scales embeddings by `sqrt(hidden_size)`, builds
   causal/local masks, runs `DecoderLayer::forward`, and returns last-token
   logits after final normalization and optional softcapping. Each decoder has
   attention and gated GELU MLP residual branches with pre/post RMSNorm.
   `Attention::forward` projects Q/K/V, normalizes Q/K, applies local/global
   `RotaryEmbedding`, updates the cache, repeats KV heads for grouped-query
   attention, and applies scaled attention plus the output projection.
4. `candle-nn/src/kv_cache.rs`: `KvCache` stores global history and
   `RotatingKvCache` retains a local window. Back in the example,
   `TextGeneration::run` prefills the prompt, then forwards one token with its
   position offset, applies repetition penalties, samples through
   `candle-transformers/src/generation/mod.rs::LogitsProcessor`, and streams
   decoded output. The instruction preset uses Gemma3 turn markers and stops at
   `<eos>` or `<end_of_turn>`.
5. The GGUF route is separate:
   `candle-examples/examples/quantized-gemma/main.rs` uses
   `candle-transformers/src/models/quantized_gemma3.rs::ModelWeights` for its
   Gemma3 4B preset. It is not a Gemma4 quantization implementation.

## Architecture Comparison

The shared decoder structure makes a dense upgrade feasible, but merely changing
a model ID or deserializing Gemma4 JSON as Gemma3 is incorrect.

| Area | Traced Candle Gemma3 path | Experimental Gemma4 text path |
| --- | --- | --- |
| RMSNorm | Learned scale is `1 + weight` | Uses checkpoint `weight` directly; normalization and scaling use FP32 for low-precision inputs |
| Attention | Q/K normalization, separate V projection; Candle scales QK by `1/sqrt(head_dim)` | Q/K normalization plus unweighted V RMSNorm; unit QK scaling and FP32 softmax |
| Local/global layout | One head dimension; layer schedule derived from a numeric pattern | Explicit `layer_types`; separate local/global head dimensions and KV-head counts |
| Position encoding | Standard RoPE with local/global scalar frequencies | Nested `rope_parameters`; standard local RoPE and proportional global RoPE, with nonrotated dimensions left unchanged |
| K=V | Independent K and V projections | Where enabled, global attention reuses the raw K projection for V, before K normalization/RoPE; this is not cross-layer KV sharing |
| Decoder/output | Gated GELU MLP, residuals, tied embeddings, optional final softcap | Retains that dense structure, adds checkpoint `layer_scalar` after the decoder block, and supports tied or root-level untied `lm_head` |
| Additional families | No PLE/shared-KV/MoE in this path | Actual Gemma4 variants may require PLE, cross-layer KV sharing, double-wide MLPs, or MoE; these are rejected rather than silently approximated |
| Multimodality | Gemma3's HF wrapper uses a SigLIP vision config; this CLI traces text only | Gemma4 has its own vision encoder and optional audio encoder; this example executes neither |

### Actual Google Variants

These are values from Google's linked **checkpoint configs**, not defaults from
`Gemma4TextConfig()` in Transformers or Candle. Context lengths are configured
limits, not validated capacities of this implementation. All four have a 262144
text vocabulary, local/global head dimensions 256/512, global proportional RoPE
with factor 0.25 and theta 1000000, local theta 10000, tied embeddings, and final
logit softcap 30.

| Instruction checkpoint config | Layers / hidden / MLP intermediate | Q heads; KV local/global | Window / context | Distinguishing requirements and current status |
| --- | --- | --- | --- | --- |
| [E2B][e2b] | 35 / 1536 / 6144 | 8; 1/1 | 512 / 131072 | PLE width 256, 20 shared-KV layers, double-wide MLP on shared layers; rejected |
| [E4B][e4b] | 42 / 2560 / 10240 | 8; 2/2 | 512 / 131072 | PLE width 256, 18 shared-KV layers; rejected |
| [26B-A4B][moe] | 30 / 2816 / 2112 | 16; 8/2 | 1024 / 262144 | Global K=V; MoE with 128 experts, top-8 routing, expert width 704; rejected |
| [31B][dense] | 60 / 5376 / 21504 | 32; 16/4 | 1024 / 262144 | Global K=V, no PLE/shared-KV/MoE; Q4_0/BF16 adapter capacity observed at 262142 input + 2 special tokens (2026-10-18), not quality validation |

E2B uses four local layers per global layer; the other three use five. E2B/E4B
leave the global KV-head override null, so it falls back to their local count.
All four configs include vision; only E2B/E4B include audio. Gemma4 Unified is a
separate architecture, not another name for one of these four, and is rejected.

## Upgrade Scope

- `candle-examples/examples/gemma4/main.rs::parse_text_config` distinguishes a
  flat `gemma4_text` config from a `gemma4` wrapper with `text_config`, checks model
  types, and validates before fetching weights. `Gemma4TextConfig::validate` in
  `candle-transformers/src/models/gemma4/config.rs` rejects unsupported features,
  malformed dimensions, layer schedules and RoPE settings. Bidirectional `all`
  attention is rejected; `vision` is accepted only for this causal text path.
- `candle-transformers/src/models/gemma4/text.rs::TextModel::new` loads a
  standalone `model.*` backbone; `new_for_conditional_generation` loads
  `model.language_model.*`. Both receive the checkpoint root; an untied head is
  `lm_head.weight`, not nested under the backbone. No vision/audio towers are
  instantiated by this loader, although the shard downloader may retrieve all
  shards, including ones containing vision/audio weights.
- In `text.rs`, `RmsNorm`, `v_norm`, `ProportionalRotaryEmbedding`, `Attention`, and
  `DecoderLayer` implement the dense differences above. Local masks exclude keys
  whose distance is at least the window. `TextModel::forward_embeds` enforces
  nonempty shapes, contiguous cache offsets, stable batch size, and the configured
  context limit. Cached multi-token prefill is supported: the CLI forwards chunks
  of at most 256 tokens at contiguous absolute RoPE positions and samples only
  after the full prompt. Local attention preserves up to window-minus-one prior
  keys before updating the rotating cache. Eager attention further chunks queries,
  retaining the chunk's keys and slicing its mask. Call `clear_kv_cache` before a
  new sequence.
- The existing `gemma4/mod.rs::Model`, `vision.rs`, `audio.rs`, and
  `multimodal_embedding.rs` are not evidence of complete multimodal support.
  The CLI rejects `--multimodal`; image/audio preprocessing, embedding placement,
  and multimodal attention still require implementation/review and validation.
  No Gemma4 GGUF, MoE, PLE, shared-KV consumers, or distributed execution
  is added by this dense-path work.

For a separate text-only training pipeline, see
[`gemma4-training`](../gemma4-training/README.md). It adds JSONL preprocessing,
a differentiable full-sequence forward, F32 AdamW, and all-layer Q4_0/BF16 LoRA
with retained real 31B H100 two-update evidence. It does not make full-parameter
31B training fit on a single H100; the training data is toy, not a useful corpus.

## Running

Run all commands in this README from the Candle workspace (`candle/`, not the
`ai10xdev/model` repository root). `--model-id` must be supplied explicitly: there is
no implicit model selection or accidental default download of a large checkpoint.
The following is an example invocation, **not a verified inference result**:

```sh
cargo run -p candle-examples --release --example gemma4 -- \
  --cpu --model-id google/gemma-4-31B-it \
  --prompt "Explain rotary position embeddings briefly." --sample-len 64
```

**Large resource requirement:** allow approximately 61 GB for BF16 or 123 GB for
F32 weights alone, plus KV cache, RoPE tables, activations and runtime overhead.
The example chooses BF16 on CUDA and F32 otherwise; CUDA additionally requires
building with `--features cuda` and omitting `--cpu`. The retained H100 evidence
below uses Q4_0/BF16 adapter mode, not this dense CPU invocation.
This example has no multi-device placement/offloading strategy.

Eager inference bounds attention-score tensors by chunking queries (at most
1024 query rows and, for the 31B probes, 256 MiB per FP32 score tensor). The CLI's
256-token cached prefill also bounds decoder activations and mask rows. Global
masks span the current chunk by all accumulated keys; local masks span the chunk
by at most window-minus-one history keys plus the chunk, not the full prompt
squared. BF16 masks are built directly without an F32 host staging copy. Direct
model callers must chunk their own prefill to obtain these mask bounds. Global
attention still performs quadratic total work. Training attention is unchanged.
The global `Cache::append` allocates the
full configured capacity on first use, so a short prompt does not imply a small
global cache. This is **not optimized
256K inference**. `--use-flash-attn` requires the `flash-attn` build feature, but
local attention and 512-wide global heads fall back to eager attention; it does
not optimize the actual 31B attention path.

The default prompt formatting is one user turn with generation prompt and thinking
disabled, following Google's [chat template][chat]. `--raw-prompt` bypasses that
formatting and automatic special tokens; supply your own `<bos>` and any required
turn markers. Stop tokens are `<eos>`, `<turn|>`, and `<|tool_response>` (the tool
handoff boundary), not Gemma3's `<end_of_turn>` or legacy `</s>`. Stopping at a tool
boundary does not implement tool execution. The CLI does not load sampling
defaults from [generation_config.json][generation]; without `--temperature` it
uses argmax, and its repetition penalty defaults to 1.1.

Use `--revision` to pin a checkpoint revision. Local `--config-file`,
`--tokenizer-file`, and comma-separated `--weight-files` can avoid Hub downloads
when all three are provided; `--model-id` remains explicit. Hub downloads require
network access, disk space, and any access authorization required by the model.

### Trained Q4 LoRA And Capacity Probes

`--adapter-checkpoint DIR` loads a `gemma4-training/checkpoint-N` directory, not
a standalone base model. It requires saved `mode: quantized-lora`,
`quantization: Q4_0` (training CLI `q4-0`), `dtype: bf16`, and F32 adapter masters.
The original absolute `source_checkpoint` directory recorded by training must
still be accessible. Its config, tokenizer, and base weight shards are selected
automatically without Hub downloads. `--model-id` remains required as an explicit
model label; in adapter mode the saved local source, not that Hub ID, selects the
weights. Explicit config/tokenizer overrides must match that base; explicit
weight files must resolve to exactly its original shards. Moved or substituted
base checkpoints are not accepted. This is path/config/tokenizer validation, not
a cryptographic checksum of base weights; do not modify checkpoint files in place.

The loader checks saved source/training architecture and tokenizer consistency,
training context metadata, projection/parameter counts, and the exact set of
finite F32 adapter tensors and shapes, including shared-KV omissions. Training's
shortened context is not used as the base's declared context. Base attention/MLP
weights are loaded one layer at a time and immediately quantized to Q4_0; there
is no retained dense decoder copy. Embeddings, norms, and head remain BF16.
Saved adapters are attached through `add_quantized_lora` using a frozen
safetensors builder, not random initialization or a trainable VarMap.

Additional flags:

| Flag | Meaning |
| --- | --- |
| `--max-context N` | Positive runtime RoPE/cache cap, no larger than the base config's declared context. Defaults to that declared context. Does not change RoPE frequencies. |
| `--probe-prompt-tokens N` | Positive exact prompt token count. Repeat the token IDs of the normally formatted (or `--raw-prompt`) prompt, preserving initial BOS once if other seed tokens exist. Requires `--sample-len >= 2`. |
| `--sample-len N` / `-n N` | Requested generated samples, including any sampled stop token. Existing default remains 10000; use an explicit small value. |
| `--ignore-eos` | Only accepted with `--probe-prompt-tokens`; ignore all three stop tokens so the requested sample count is reached. |
| `--report-file PATH` | Write JSON after successful execution; parent directory must exist. A probe stopping before cached decode is an error and writes no success report. Use a fresh filename per attempt. |

Every run checks `prompt_tokens + requested_sample_tokens <= runtime_context`
before model allocation, even if EOS might stop generation sooner. Oversized
requests fail rather than truncating the normal prompt or extending context.
Without the probe flags, chat formatting, sampling, streamed prompt/output, and
EOS handling are unchanged. Synthetic prompts are not decoded or printed in full.

For an existing CUDA build (commands are examples, not full-model validation):

```sh
./target/release/examples/gemma4 \
  --model-id google/gemma-4-31B-it \
  --adapter-checkpoint /absolute/training-run/checkpoint-100 \
  --max-context 1024 --prompt "Explain rotary position embeddings briefly." \
  --sample-len 64 --report-file normal.json

./target/release/examples/gemma4 \
  --model-id google/gemma-4-31B-it \
  --adapter-checkpoint /absolute/training-run/checkpoint-100 \
  --max-context 4096 --prompt "The quick brown fox jumps over the lazy dog." \
  --probe-prompt-tokens 4080 --sample-len 16 --ignore-eos \
  --seed 299792458 --report-file probe-4096.json
```

For a prompt of P tokens and G > 0 generated samples, the first generation step
prefills all P tokens in cached chunks of at most 256 tokens. The remaining
G - 1 forwards each decode one cached token.
`forwarded_tokens = P + G - 1`, and
`max_forwarded_position_zero_based = P + G - 2`. The last sampled token is **not**
forwarded. Thus the 4096-token budget example above forwards 4095 tokens, through
zero-based position 4094, not position 4095. The conservative budget deliberately
does not allow an extra sample to fill that gap. With zero samples in normal mode,
no forward occurs and the maximum position is null.

Reports include success/mode, prompt/requested/actual sample counts, generated
token IDs, generated `output_text` (special tokens skipped, prompt excluded),
forwarded counts/positions, prefill/cached-decode coverage, EOS status, declared
and runtime contexts, complete adapter metadata, source paths, device/dtype,
sampling parameters, and seed prompt IDs sufficient to reconstruct synthetic
input exactly. Device synchronization brackets model loading and each forward.
Prefill/decode forward timings include input upload but exclude sampling/output;
generation wall time includes sampling and streamed output. No full synthetic
token array is emitted. Argmax is the default; a fixed sampling seed and identical
artifacts/settings help reproducibility but do not guarantee bitwise equality
across hardware/kernel versions.

Run each capacity attempt in a fresh process and vary `--max-context` and
`--probe-prompt-tokens` together, keeping an explicit generation allowance.
A success establishes only that this synthetic prefill and cached decode fit
and ran with those settings, not long-range model quality or a maximum supported
context. Failures exit nonzero without writing a new success report; an older
file at the same path is not removed. Quadratic global attention work and full
global cache allocation remain costs despite Q4 decoder weights and bounded
CLI prefill masks/attention scores.

### Retained H100 Evidence

The latest retained [2026-10-18 report](../../../../h100-server/context-sweep-20261018/README.md)
records successful synthetic inputs of 32768, 65536, 131072, and **262142 tokens**,
each with two generated tokens. The largest test generated two special tokens
with empty decoded text, forwarding 262143 positions in total. It demonstrates
capacity under the tested policy, not long-context quality, an OOM boundary, or
exact 262144-token input prefill. Full-output maxima for the three smaller inputs
were still incomplete at that snapshot; their remaining budgets are not measured
output maxima. The report includes retained hardware telemetry and timings.

The [2026-10-14 historical summary](../../../../h100-server/context-evidence-20261014-v2/summary.json)
records the installed checkpoint at revision
`842da3794eaa0b77d5f08bae87a17459d91ff475`, with both base shards verified against
the local manifest. On one H100 NVL (95830 MiB), the trained all-layer Q4_0/BF16
adapter produced both training-prefix and instruction responses. Synthetic
prefill-plus-cached-decode probes succeeded at 1024, 2048, 4096, and **8192 prompt
tokens**. The last used runtime cap 8194, sampled two tokens, and forwarded 8193
tokens through zero-based position 8192; sampled maximum whole-device usage was
56115 MiB (0.5-second sampling, not instantaneous peak).

The 16384-token prompt failed with `CUDA_ERROR_INVALID_VALUE` (invalid argument),
**not CUDA OOM**. Thus 8192 was the largest successful prompt in that historical
sweep, not a measured maximum; that sweep established no OOM upper bound. Those
runs predate query-chunked eager attention and cached CLI prefill; the October 18
report above supersedes their capacity observations. They establish real
checkpoint/adapter execution, not long-range quality or useful training from the
synthetic corpus.

## Checks And Limits

The existing tests can be run without downloading model weights:

```sh
cargo check -p candle-examples --example gemma4
cargo test -p candle-transformers --lib gemma4::config::tests
cargo test -p candle-transformers --test gemma4_tests
cargo test -p candle-examples --example gemma4
```

Config unit tests cover supported flags, rejection paths and RoPE validation.
`candle-transformers/tests/gemma4_tests.rs` uses tiny synthetic CPU weights to
exercise checkpoint prefixes, tied/untied heads, missing tensors, cache reset and
window wraparound, continuation guards, layer scalars, and final softcapping.
Example tests also cover CLI/probe validation, conservative budgets and overflow,
exact synthetic token counts, adapter metadata/tensor rejection, and frozen
Q4/BF16 safetensors adapter loading with initial prefill, cached decoding, cache
reset, and zero/one/multiple-sample position reporting. The 12 example tests pass
on CPU; this is not a real-checkpoint GPU capacity measurement.

The model tests include 4480 scalar logit comparisons against independently
generated **Transformers 5.19.0 / Torch 2.8.0+cpu** outputs, with absolute tolerance
`2e-5`. Four tiny dense configurations cover tied/untied heads, proportional RoPE,
signed layer scalars, global K=V and separate K/V, and prefill/cached decoding across
local-window boundaries. Both checkpoint prefixes are checked. See
[`gemma4_reference.md`](../../../candle-transformers/tests/gemma4_reference.md)
for provenance and regeneration instructions; Rust tests need no Python runtime.
This establishes parity for those tiny CPU/F32 cases, not real-checkpoint
correctness, BF16 accuracy, GPU kernel correctness, or long-context performance.

## Sources

Architecture references are pinned to **Hugging Face Transformers v5.19.0**:

- [Gemma3 configuration](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/gemma3/configuration_gemma3.py): text config and SigLIP wrapper.
- [Gemma4 configuration](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/gemma4/configuration_gemma4.py): config schema and per-layer overrides. Class defaults are not actual variant specifications.
- [Gemma4 implementation](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/gemma4/modeling_gemma4.py): `Gemma4RMSNorm`, `Gemma4TextAttention`, `Gemma4TextDecoderLayer`, PLE and MoE paths.

Google's actual checkpoint configs, chat template and generation config below
were consulted on 2026-10-13. These Hub links use mutable `main`, unlike the pinned
Transformers references; pin a Hub commit for reproducible experiments.

[e2b]: https://huggingface.co/google/gemma-4-E2B-it/blob/main/config.json
[e4b]: https://huggingface.co/google/gemma-4-E4B-it/blob/main/config.json
[moe]: https://huggingface.co/google/gemma-4-26B-A4B-it/blob/main/config.json
[dense]: https://huggingface.co/google/gemma-4-31B-it/blob/main/config.json
[chat]: https://huggingface.co/google/gemma-4-31B-it/blob/main/chat_template.jinja
[generation]: https://huggingface.co/google/gemma-4-31B-it/blob/main/generation_config.json
