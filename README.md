# Gemma4 In Candle

Experimental dense-text Gemma4 inference, dataset preprocessing, and all-layer
Q4_0/BF16 LoRA fine-tuning with Rust Candle on a single H100. This repository
publishes the existing implementation and retained hardware evidence; it does
not contain model weights or a hosted inference service.

**Hardware results: [tel.md](tel.md).** The retained input sweep completed at
32,768, 65,536, 131,072, and 262,142 tokens, generating two tokens per trial.
Long-output maximum tests were incomplete at the October 18, 2026 snapshot.
Architectural budgets are not measured output maxima.

## Contents

| Path | Purpose |
| --- | --- |
| [candle/](candle/) | Complete Candle source workspace, including the experimental changes |
| [Gemma4 inference](candle/candle-examples/examples/gemma4/README.md) | Architecture comparison, supported dense path, inference CLI and limitations |
| [Gemma4 training](candle/candle-examples/examples/gemma4-training/README.md) | JSONL preprocessing, CPU toy example, quantized LoRA and checkpoint format |
| [h100-server/](h100-server/README.md) | Historical deployment scripts, capacity probes and regression tests |
| [tel.md](tel.md) | GPU utilization/VRAM, CPU/RAM, context accounting and links to raw evidence |

## Build And Test

Use a current Rust toolchain; the recorded GPU build used Rust 1.96.0 and CUDA
12.8. Run Cargo commands from the `candle/` directory, not this repository root:

```sh
cargo check --locked -p candle-examples --example gemma4 --example gemma4-training
cargo test --locked -p candle-transformers --test gemma4_tests --test gemma4_training_tests --test gemma4_quantized_decoder_tests --test gemma4_lora_tests
cargo test --locked -p candle-examples --example gemma4 --example gemma4-training
```

Run monitor/probe tests from this repository root; these do not access the GPU:

```sh
python3 -B -m unittest discover -s h100-server -p 'test_*.py' -v
```

Follow the training README for an offline toy-model smoke run. Real-checkpoint
execution requires separately obtained base weights, sufficient resources, and
any required model access authorization. Adapter weights referenced in telemetry
are not included. Historical H100 wrappers contain deployment-specific paths and
host checks; inspect and adapt them before use. They are not portable provisioning
commands. Do not launch a competing workload on an occupied GPU.

## Scope And Evidence

- The full-sized checkpoint was fine-tuned with 61,214,720 trainable adapter
  parameters across 60 decoder layers, not by updating all 31B base parameters.
- The verified 20-update run used repetitive toy data. Loss reduction establishes
  pipeline operation, not useful generalization or production readiness.
- The selected dense text path rejects unsupported MoE, PLE and cross-layer
  shared-KV variants; this is not complete multimodal Gemma4 support.
- Context probes use synthetic repeated tokens. The near-262k result generated
  special tokens with empty decoded text, not meaningful long-context answers.
- Evidence is a dated snapshot, not live telemetry. Raw records retain historical
  artifact paths and hashes; executable/checkpoint files are not redistributed.
  Recorded hashes are not a complete source-to-binary build attestation.

## Upstream And Licensing

The `candle/` source is based on
[huggingface/candle](https://github.com/huggingface/candle), commit
`c68b24997319b8d76e5a4e775dab197c7f26322a`, plus the existing Gemma4 implementation,
training, inference and test changes. This repository starts a new publication
history; upstream history remains available at that source commit. The nested
workspace is ordinary source, not a submodule. Its original GitHub workflows are
retained under `candle/.github/`, not enabled as root-level project workflows.

Candle's [Apache-2.0](candle/LICENSE-APACHE) and [MIT](candle/LICENSE-MIT) license
texts and upstream notices are retained. Model weights have separate terms and
are not covered by Candle's source license. No base/adapter checkpoints, access
tokens, local orchestration transcripts, or build artifacts are included.
