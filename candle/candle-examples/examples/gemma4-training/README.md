# Gemma4 Text Training

An offline example for dense, text-only causal language-model training in Candle.
The fixtures are a **toy architecture and toy tokenizer, not an official Gemma4
checkpoint**. They test the pipeline, not model quality or production-scale
training. No model downloads or remote GPU access are needed for the CPU smoke run.

## Offline CPU Smoke Run

Run these commands from `candle/candle-examples/examples/gemma4-training`.
Cargo locates the workspace through its parent directories. Rust dependencies
must already be available for a fully offline build; add Cargo's `--offline`
before `--` if you need to enforce that restriction.

Both output directories must be new, and their parent `/tmp/opencode` must exist.
For another run, choose different output paths; existing directories are never
overwritten.

```sh
cargo run -p candle-examples --example gemma4-training -- preprocess \
  --input fixtures/train.jsonl \
  --tokenizer fixtures/tokenizer.json \
  --output /tmp/opencode/gemma4-data \
  --sequence-length 8 \
  --vocab-size 16 \
  --validation-fraction 0.25 \
  --seed 42 \
  --bos-token-id 1 \
  --eos-token-id 2

cargo run -p candle-examples --example gemma4-training -- train \
  --config fixtures/config.json \
  --dataset /tmp/opencode/gemma4-data \
  --output /tmp/opencode/gemma4-run \
  --from-scratch \
  --steps 20 \
  --batch-size 2 \
  --learning-rate 0.001 \
  --eval-every 10 \
  --eval-batches 4 \
  --save-every 10 \
  --cpu
```

The standalone `gemma4_text` fixture has hidden size 16, intermediate size 32,
two layers, two query heads, one KV head, and local/global head dimensions of 8.
It alternates sliding attention (window 4) and full attention, with global
proportional RoPE using a partial rotary factor of 0.5. Local RoPE uses full
rotation. It enables `attention_k_eq_v` and tied word embeddings. The source
context limit is 32; training uses the dataset sequence length of 8 as its
effective context to bound RoPE allocations, while preserving the source config.

The whitespace WordLevel tokenizer has exactly 16 entries: unknown ID 0, BOS ID
1, EOS ID 2, and 13 ordinary words. Each of the 12 records has 24 ordinary tokens;
explicit BOS/EOS makes 26 tokens per document. The split has 9 training and 3
validation documents. At sequence length 8 this produces 29 training windows
and 9 validation windows, enough for four validation batches of two. The data
is deliberately repetitive and is not a meaningful held-out benchmark.

## Data Contract

Input is strict JSONL, one object per line with a nonempty string `text` field:

```json
{"text":"the small red cat sees the blue dog"}
```

Malformed JSON, blank lines, missing/non-string `text`, whitespace-only text,
zero-token encodings, and out-of-vocabulary token IDs fail preprocessing rather
than being silently skipped. The vocabulary size must agree with the model.
Use the real checkpoint's tokenizer for real data, never the toy tokenizer.

Preprocessing holds tokenized documents in RAM. It shuffles documents with the
supplied seed and reserves `floor(document_count * validation_fraction)` for
validation **before packing**, so one document cannot cross the split. There is
no deduplication: duplicated or related source documents can still leak content
between splits. Clean and deduplicate your data upstream.

Tokenization disables automatic special-token insertion, truncation, and padding.
Optional explicit BOS and required EOS IDs delimit each document. Documents are
concatenated within each split; windows may cross document boundaries. Both
splits must contain at least one complete window. For sequence length `S`, each
window contains `S + 1` tokens with stride `S`: the first `S` are inputs and the
last `S` are shifted targets. Incomplete tails are discarded without padding.

The dataset directory contains:

- `train.bin` and `valid.bin`: headerless contiguous little-endian `u32` token streams.
- `metadata.json`: format, sequence length, vocabulary, split seed, token/window counts, and discarded-tail counts.
- `tokenizer.json`: a byte-for-byte copy of the supplied tokenizer.

The objective is text-only continual pretraining: next-token cross-entropy on
**all shifted tokens**, including document delimiters and boundary transitions.
There is no assistant-only loss masking and no automatic chat templating. This
is not a chat supervised-fine-tuning pipeline.
Training checks split sizes against metadata, including whole-window truncation.
This is not a content checksum: keep the dataset and tokenizer immutable after
preprocessing.

## Checkpoints And Metrics

Choose exactly one initialization mode: `--from-scratch` or `--checkpoint DIR`.
Checkpoint loading still requires a separately specified `--config` and a
`--dataset`. The checkpoint directory must contain a `tokenizer.json` that is
byte-identical to the dataset tokenizer; equivalent vocabulary or differently
formatted JSON is not sufficient. Weights are automatically resolved from
`model.safetensors` or a safetensors shard index and its referenced shards.

For example, warm-start from the smoke run with a fresh output directory:

```sh
cargo run -p candle-examples --example gemma4-training -- train \
  --config /tmp/opencode/gemma4-run/checkpoint-20/config.json \
  --checkpoint /tmp/opencode/gemma4-run/checkpoint-20 \
  --dataset /tmp/opencode/gemma4-data \
  --output /tmp/opencode/gemma4-warm-start \
  --steps 20 --batch-size 2 --learning-rate 0.001 \
  --eval-every 10 --eval-batches 4 --save-every 10 --cpu
```

Metrics are written to `OUTPUT/metrics.jsonl`. Each checkpoint under
`OUTPUT/checkpoint-{step}/` contains:

- `model.safetensors`: model weights in full-training mode; both quantized modes save only `adapter.safetensors` instead.
- `config.json`: effective training configuration with context set to the dataset sequence length.
- `source-config.json`: the source configuration supplied to this run.
- `tokenizer.json`: the dataset tokenizer.
- `dataset-metadata.json`: dataset provenance and packing metadata.
- `training-state.json`: training progress/settings, not optimizer tensors.

**Loading a checkpoint is a warm start, not exact resume.** AdamW optimizer state
is not saved. The source config is retained for provenance; restoring a longer
context does not establish that the model was trained or validated at that length.
Conditional-generation inputs retain their `model.language_model.*` weight
namespace, but saved weights contain only the text decoder, not a complete
multimodal checkpoint. This example and Candle's text-only Gemma4 inference path
can load that layout.

## Devices And Limits

The default `--mode full --precision f32` uses F32 AdamW for all parameters except
the frozen `layer_scalar`. `--mode quantized-head --precision bf16` instead trains
an output-head LoRA adapter; `--mode quantized-lora --precision bf16` trains LoRA
adapters throughout the frozen quantized decoder, as described below. The CLI
default remains `full`; the hosted training wrapper defaults to `quantized-lora`.
The default global gradient-norm limit is `--max-grad-norm 1`. There is no
distributed training or full-model mixed-precision training in this example.
Scratch embedding/projection weights use a zero-mean normal distribution with
the configured `initializer_range` (default `0.02`); biases are zero and norms
and layer scalars start at one. Nonzero `attention_dropout` is rejected because
dropout is not implemented. PLE/shared-KV consumer, MoE, Unified, and multimodal
training are not supported; a smaller released Gemma4 variant is not necessarily
compatible with this dense path.

`--max-parameters` defaults to `100000000`. It is a hard parameter-count cap
checked with an estimate before allocating the model, not a device-memory
guarantee. F32 weights, gradients, and two AdamW moments alone require roughly
16 bytes per parameter; activations, logits, attention, RoPE, and temporary
buffers need additional memory. A full 31B model would require about **496 GB
for weights/gradients/moments alone**, so full-parameter 31B training on one H100
is not supported. Raising the cap does not make such a run fit.

`--gradient-accumulation-steps` (default `1`) sets microbatches per optimizer
update in all modes. Each microbatch's mean-loss gradients are weighted by its
actual number of windows, including short batches at epoch ends, then divided
by the total windows in the update. Training loss uses the same weighting.
Only detached parameter gradients accumulate; each microbatch's graph is freed
after backward. Global gradient clipping and AdamW run once per update, not per
microbatch. `--steps`, evaluation/save intervals, and checkpoint step numbers
count optimizer updates; `tokens_seen` counts the actual processed tokens.

## Quantized Head Training

This deliberately limited parameter-efficient mode supports the installed dense
`google/gemma-4-31B-it` text decoder on one H100. It is **not full-model training,
QAT, or all-layer QLoRA**. Standard Candle `QMatMul` still has no backward pass
and is unchanged. In `quantized-head`, the entire base is frozen, decoder
features are detached, and only an additive low-rank output projection is trained:

```text
h = stop_gradient(normalized_decoder_features(tokens))
logits = softcap(base_lm_head(h) + (alpha / rank) * (h @ A.T @ B.T))
```

The softcap, when configured, is applied after adding the adapter. The tied base
embedding/output matrix is not modified; the adapter changes only the output.
`--lora-rank 8 --lora-alpha 8` defaults to 2,140,160 trainable parameters for the
31B head-only configuration, not for the all-layer mode below.

## Quantized LoRA Training

`--mode quantized-lora` attaches adapters to **every frozen Q4_0/Q8_0 attention
Q/K/V/O and MLP gate/up/down projection in every decoder layer**. Where
`attention_k_eq_v` omits a separate V projection, there is no V adapter to add.
The default is rank 8 with alpha 8. Each projection adds
`(alpha / rank) * (X @ A.T @ B.T)` to its packed base result. Only adapters are
trained; embeddings, the base output head, norms, biases, scalars, and packed
weights remain frozen. This mode does not add an output-head adapter.

The Gemma4-specific `FrozenQMatMul` custom operation preserves the packed forward
kernel and provides a **first-order input-gradient path** through the frozen
projection: `dX = dY @ dequantize(W)`. Backward dequantizes one projection to a
transient F32 dense weight, computes the F32 input gradient, and releases that
dense weight; it is not retained in the graph or optimizer. No base-weight
gradient or higher-order derivative is provided. Where the forward kernel also
quantizes activations, this is a **straight-through input-gradient approximation**,
not the exact derivative of the activation-quantizing kernel. Standard Candle
`QMatMul` remains unchanged and has no backward pass.

With `--precision bf16`, adapters use F32 master weights and BF16 compute casts
recreated each forward. Adapter gradients and AdamW moments remain F32. This is
parameter-efficient training on a **full-sized checkpoint, not full-parameter
training or QAT**. The retained single-H100 runs below verify this all-layer path
and streamed loader for 20 optimizer updates at sequence length 32 and two
updates at sequence lengths 128 and 256;
historical head-only results do not measure this path or its memory requirements.

## Shared Quantized Policy

Both `quantized-head` and `quantized-lora` require a pretrained `--checkpoint`,
not `--from-scratch`, and share these policies:

- `--quantization q4-0` (default) or `q8-0` uses Candle's GGML block quantization on attention Q/K/V/O and MLP gate/up/down matrices. Both matrix dimensions must be multiples of 32; incompatible configurations fail explicitly.
- Embeddings, output head, norms, and scalars stay dense in the selected compute dtype. No quantized optimizer, NF4, FP8, bitsandbytes, or external quantization library is used.
- `--precision bf16` uses BF16 features, attention, dense head, and adapter matmuls. Quantized matmul inputs are promoted to F32 for Candle CUDA fallback compatibility, with outputs cast back to BF16.
- Norm reductions, attention softmax, loss, adapter master weights, adapter gradients, and AdamW moments use F32. Master-to-BF16 casts are recreated each forward so optimizer updates remain visible. BF16 does not use FP16-style loss scaling; nonfinite loss/gradients fail before updating.
- A starts from normal(0, 0.02); B starts at zero. `--seed` still controls data order only.
- Both modes stream the memory-mapped checkpoint: each decoder layer is loaded dense, then its projections are quantized before the next layer is loaded. Dense projection storage is released as it is replaced; no entire dense base copy is retained. Candle performs this weight conversion on the CPU, transferring each quantized projection back to CUDA. Loading needs the retained quantized layers, dense embeddings/head/norms, one dense decoder layer, and conversion buffers, not the previously required entire BF16 base. Quantized weights are not exported or cached; they are reconstructed from the original checkpoint each run.
- Attention and logits are still materialized. Start with sequence length 32 and batch size 1; this implementation does not promise long-context capacity or provide activation checkpointing. Gradient accumulation does not remove per-microbatch activation memory or the transient F32 projection required for LoRA backward.

After preprocessing with the real checkpoint tokenizer, run the CUDA binary:

```sh
target/release/examples/gemma4-training train \
  --config /home/azureuser/gemma4/checkpoint/config.json \
  --checkpoint /home/azureuser/gemma4/checkpoint \
  --dataset /path/to/prepared-data --output /path/to/new-run \
  --mode quantized-head --quantization q4-0 --precision bf16 \
  --lora-rank 8 --lora-alpha 8 --max-parameters 32000000000 \
  --steps 20 --batch-size 1 --learning-rate 0.001 \
  --weight-decay 0 --eval-every 10 --eval-batches 2 --save-every 10 --cuda
```

Run from the Candle root. The parameter cap covers the base model, not only the
adapter. The dataset must use the real vocabulary size 262144, BOS 2 and EOS 1
for this raw-text objective; it does not insert instruction/chat templates.

For all-layer LoRA, use the same real `google/gemma-4-31B-it` checkpoint and
prepared dataset, with a fresh output directory:

```sh
target/release/examples/gemma4-training train \
  --config /home/azureuser/gemma4/checkpoint/config.json \
  --checkpoint /home/azureuser/gemma4/checkpoint \
  --dataset /path/to/prepared-data --output /path/to/new-lora-run \
  --mode quantized-lora --quantization q4-0 --precision bf16 \
  --lora-rank 8 --lora-alpha 8 --max-parameters 32000000000 \
  --steps 20 --batch-size 1 --gradient-accumulation-steps 4 \
  --learning-rate 0.0001 --weight-decay 0 \
  --eval-every 10 --eval-batches 2 --save-every 10 --cuda
```

`run-config.json` records the initial policy, and checkpoint `training-state.json`
records progress, quantization, precision, adapter dimensions, frozen parameters,
and the canonical base checkpoint path. Both modes save only F32 adapters, not a
complete model or a GGUF file: head mode uses `lora_a`/`lora_b`; all-layer mode
uses `layers.{index}.{projection}.lora_a/lora_b`. **Keep the original base
checkpoint immutable and available at the recorded path.** Its contents are not
cryptographically verified by the trainer on reload.

To warm-start either quantized mode, use the same original `--checkpoint`,
`--mode`, quantization/precision/rank/alpha settings, a fresh output directory, and add
`--adapter-checkpoint /path/to/previous-run/checkpoint-20`. The trainer checks
policy (including an exact mode match), base path, architecture, tokenizer, and
tensor shapes. Head and all-layer adapters are not interchangeable. AdamW/data
order restart. Passing the adapter directory as `--checkpoint` is not supported.
The [`gemma4` inference CLI](../gemma4/README.md#trained-q4-lora-and-capacity-probes)
loads all-layer Q4_0/BF16 adapters with F32 masters via `--adapter-checkpoint`,
using the recorded original base checkpoint. It does not load head-only or Q8_0
adapters; the training path reloads both modes for validation and further training.

For example, after the all-layer run above completes:

```sh
target/release/examples/gemma4-training train \
  --config /home/azureuser/gemma4/checkpoint/config.json \
  --checkpoint /home/azureuser/gemma4/checkpoint \
  --adapter-checkpoint /path/to/new-lora-run/checkpoint-20 \
  --dataset /path/to/prepared-data --output /path/to/new-lora-warm-start \
  --mode quantized-lora --quantization q4-0 --precision bf16 \
  --lora-rank 8 --lora-alpha 8 --max-parameters 32000000000 \
  --steps 1 --batch-size 1 --gradient-accumulation-steps 4 \
  --learning-rate 0.0001 --weight-decay 0 \
  --eval-every 1 --eval-batches 2 --save-every 1 --cuda
```

The hosted build/run wrappers are in `../../../../h100-server/` relative to this
example directory. They capture logs, input/binary/config hashes, base revision
manifest, and sampled GPU memory. The training wrapper now defaults to
`quantized-lora`; select `MODE=quantized-head` for head-only training. See that
directory's README and [historical head-only evidence](../../../../h100-server/training-evidence-20261013/README.md)
for the unchanged 2026-10-13 H100 observations and the distinction between smoke
data and a useful training corpus. Those runs used the old dense-before-quantize
loader; their memory samples do not measure the new streamed loader or all-layer
LoRA.

The original [all-layer 20-update run](../../../../h100-server/lora-training-evidence-20261013/README.md)
was reverified on the VM on 2026-10-16, and its raw metrics/logs were recovered
locally. At sequence length 32, microbatch 1 and accumulation 4, it processed
2560 tokens with initial/final validation loss 8.907012 / 1.717138 and sampled
maximum GPU usage 35125 MiB. All 820 adapter tensors changed between steps 10
and 20. This is the original toy-data smoke run, not a long-context confirmation.

The retained [2026-10-14 training summary](../../../../h100-server/training-context-evidence-20261014/summary.json)
and `screen-128`/`screen-256` checkpoint states record real training on the installed
`google/gemma-4-31B-it` text decoder on one H100 NVL (95830 MiB): 30,697,345,340
base parameters, 410 quantized projections across 60 layers, and 61,214,720
trainable rank-8/alpha-8 LoRA parameters. These are Q4_0/BF16 streamed-loader runs
with F32 masters/gradients/AdamW/loss, microbatch 1, accumulation 4, learning rate
0.0001, weight decay 0, and clipping 1. Each warm-started the same trained adapter
with a fresh optimizer, then completed **two optimizer updates**, validation,
and adapter saving.

| Input sequence length | Tokens seen | Final train / validation loss | Sampled maximum GPU MiB |
| --- | --- | --- | --- |
| 128 | 1024 | 0.507516 / 0.743247 | 51859 |
| 256 | 2048 | 0.291868 / 0.376129 | 83381 |

Memory is whole-device usage sampled every 0.5 seconds, including loading, not
an instantaneous allocation peak. The retained sequence-512 trial reports CUDA
OOM; this does not establish an exact training-context maximum. The synthetic
repeated-text corpus uses the real tokenizer but is **toy data, not useful
training or a model-quality benchmark**. The summary's method describes a planned
20-update boundary confirmation, but its retained successful trials contain only
two-update metrics; the sequence-32 smoke run does not confirm that boundary.

For a suitably small model on a local CUDA machine with the CUDA toolkit and
dependencies installed, build from the same example directory:

```sh
cargo build --release -p candle-examples --example gemma4-training --features cuda
```

With Cargo's default target directory, the binary is
`../../../target/release/examples/gemma4-training` relative to this directory.
Invoke that binary with the same `train` arguments above, replace `--cpu` with
`--cuda`, and choose a fresh output directory. Do not pass both device flags.
If `CARGO_TARGET_DIR` is customized, use the binary under that target directory.
This does not provision a remote GPU or download weights.

Seeds control data randomness only: preprocessing's `--seed` controls document
splitting, and training's `--seed` controls data order. Candle CPU scratch
initialization is not seedable here. Identical CLI seeds therefore do not imply
identical initial weights, losses, or reproducible training runs.

## Verification

From the repository root:

```sh
cargo test -p candle-examples --example gemma4-training --example gemma4
cargo test -p candle-transformers --test gemma4_training_tests --test gemma4_tests
cargo test -p candle-transformers --test gemma4_quantized_decoder_tests --test gemma4_lora_tests
cargo test -p candle-transformers --lib gemma4
```

Coverage includes preprocessing errors and determinism, u32 IDs above 65535,
complete-window truncation, scratch initialization, gradient clipping, real
optimizer updates, warm starts, sharded loading, inference parity, numerical
attention-gradient checks, and nonzero Q/K/V/MLP gradients. Quantized-head tests
also cover F32/BF16 Q4/Q8 forward paths, finite F32 adapter gradients and updates,
numerical adapter gradients, unchanged frozen weights, softcap ordering, policy
rejection and checkpoint warm starts. All-layer LoRA tests cover cross-layer
adapter gradients against a smooth dequantized-weight reference (not exact
derivatives of activation-quantizing kernels), F32/BF16 master updates, frozen
base preservation, shared-KV target selection, and streamed-constructor parity.
Training-example tests cover warm starts and mode-policy rejection for both
quantized modes, plus window-weighted accumulation with a short epoch tail.
With `--features cuda`, the CUDA-specific adapter tests require CUDA device 0
and do not silently skip missing GPUs.
Actual hosted execution evidence is recorded separately in the H100 README;
toy-data execution does not establish model quality or production readiness.
