# Gemma4 H100 Installation

Published with the existing implementation in `ai10xdev/model`. Runtime and job
status below describe the **2026-10-18 snapshot**, not current VM state; earlier
installation and test results retain their historical dates.

## Input And Output Sweep

The [2026-10-18 measurement snapshot](context-sweep-20261018/README.md) reports
completed input-load measurements at 32,768, 65,536, 131,072 and 262,142 tokens,
with retained telemetry. Full-output maximum tests were running/queued as of
that snapshot, so their maxima remain incomplete in the retained evidence;
the report distinguishes measured output counts from architectural budgets.

[`sweep-inference-output.py`](sweep-inference-output.py) records per-input GPU
utilization/VRAM and host/process CPU/RAM at 32,768, 65,536 and 131,072 tokens,
then attempts the full remaining output budget for each input. It reuses a
validated 262,142-input + 2-output hardware trace rather than repeating it.
All runs use the existing trained adapter and model binary, without retraining
or changing the model's declared 262,144-token context.

```sh
python3 -B /path/to/sweep-inference-output.py \
  /home/azureuser/gemma4/context-optim-1792221640-93574/summary.json \
  /home/azureuser/gemma4/new-input-output-sweep \
  --boundary-summary /home/azureuser/gemma4/context-load-20261018/summary.json \
  --input-timeout 12000 --output-timeout 172800
```

Keep `measure-inference-load.py` beside the runner. Use a new output directory
and an idle GPU; the monitor verifies binary/model/adapter hashes before each
trial. Run detached for long jobs. The runner atomically updates `summary.json`
with the active trial, completed results and maximum-output status. Each trial
retains its command, inference report, log, provenance and approximately
one-second raw `telemetry.jsonl`. A running trial has no final output count yet.

The initial input trials request two outputs. Subsequent output trials request
229,376, 196,608 and 131,072 outputs, respectively, with EOS ignored. Only exact
successful generation establishes the maximum under the existing runtime
policy. These numbers are otherwise **budgets, not measured output maxima**.
Timeouts and other failures leave the maximum inconclusive; this runner does
not bisect failed candidates. Each output trial has a 48-hour timeout by default.

Exact 262,144-token input has zero remaining output budget and is reported
separately as architectural accounting, not a measured prefill. The measured
near-262k case is explicitly 262,142 input plus 2 generated special tokens with
empty decoded text. These are
synthetic capacity tests, not a demonstration of useful long-context answers;
the two special tokens are not a meaningful answer.

At the 2026-10-18 snapshot, the running job was rooted at
`/home/azureuser/gemma4/context-sweep-20261018T0146` on the VM, with results
under `results/`. Current runtime state has not been checked; verify that the
GPU is idle before starting any workload.

```sh
python3 -B -m unittest discover -s h100-server -p 'test_*inference*.py' -v
```

## Verified Training

The existing **31B all-layer Q4_0 LoRA run** was reverified on the VM on
2026-10-16: 20 optimizer updates, sequence length 32, microbatch 1, accumulation
4, BF16 compute and F32 AdamW. All 820 adapter tensors changed across 60 decoder
layers; 61,214,720 parameters were trainable. Initial/final validation loss was
8.907012 / 1.717138 on the repetitive toy fixture, with 35,125 MiB sampled maximum
GPU memory including loading. See the [recovered evidence](lora-training-evidence-20261013/README.md).

This trains adapters on the full-sized checkpoint, **not all 31B parameters**.
Full-parameter F32 AdamW training would need roughly 496 GB before activations.
No production corpus was supplied, so useful domain training and quality
evaluation remain outside the verified smoke run.

## Training Context Measurement

[`probe-gemma4-training.py`](probe-gemma4-training.py) measures **training**
capacity, independently of the inference probe below. It preserves the selected
31B all-layer Q4_0 LoRA policy: BF16 compute, F32 adapter masters/AdamW/loss,
rank/alpha 8/8, microbatch 1, accumulation 4, learning rate 0.0001, gradient
clipping 1, and no weight decay. No FlashAttention or activation checkpointing
is enabled. Each trial warm-starts the same existing step-20 adapter with fresh
optimizer state, without modifying that checkpoint or the base model.

On the VM, choose a new output directory:

```sh
python3 /home/azureuser/gemma4/probe-gemma4-training.py \
  /home/azureuser/gemma4/training-context-evidence-new
```

The probe verifies the pinned base shard hashes, refuses an occupied GPU or an
existing output directory, and records commands, input/dataset/binary/adapter
hashes, logs, metrics, and whole-device memory samples every 0.5 seconds. It
uses the real tokenizer to pack explicitly synthetic repeated text into exact
S-token inputs with S shifted targets (S+1 tokens per source window). This
measures memory capacity, not learning quality or long-context comprehension.

The default search doubles from 128 to at most 4,096 tokens, then refines the
first CUDA OOM bracket to 32-token resolution. Each screening trial performs
two optimizer updates, each containing four forward/backward microbatches;
initial/final evaluation and checkpoint saving are included. The successful
boundary is confirmed with 20 updates and the failing boundary is repeated.
Success requires finite metrics, exact token/step accounting, the selected
training policy, and changed adapter bytes. Only an actual CUDA OOM establishes
an upper bound; timeouts, preprocessing failures, and other errors do not.
Inspect `summary.json` for completed confirmation, not just process exit status.
The per-training-process timeout defaults to 30 minutes.

Report the confirmed success/OOM **interval**, not an exact maximum unless its
bounds are adjacent tokens. The source model's declared context is a separate
architectural cap. Whole-device samples include loading and may miss transient
allocation peaks. A 20-update confirmation is not a guarantee for indefinite
training or different batch sizes, kernels, optimizer settings, or workloads.

CPU-only probe regression tests:

```sh
python3 -B -m unittest discover -s h100-server -p test_probe_gemma4_training.py -v
```

## Trained-Model Inference And Context Measurement

The 2026-10-14 inference CLI supports the **all-layer** Q4 LoRA checkpoint, not
the historical head-only checkpoint. It validates saved adapter policy, shapes,
source config, and tokenizer before attaching frozen F32 adapters to the Q4_0
base with BF16 compute. No retraining or checkpoint modifications are required.

On the VM, generate from the completed training run:

```sh
LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64 \
  /home/azureuser/candle/target/release/examples/gemma4 \
  --model-id google/gemma-4-31B-it \
  --revision 842da3794eaa0b77d5f08bae87a17459d91ff475 \
  --adapter-checkpoint /home/azureuser/gemma4/lora-q4-bf16-smoke-20261013/run/checkpoint-20 \
  --max-context 4096 --sample-len 64 \
  --prompt "Explain in two sentences why the sky appears blue."
```

Reproduce the bounded context sweep with a **new** output directory:

```sh
python3 /home/azureuser/gemma4/probe-gemma4.py \
  /home/azureuser/gemma4/context-evidence-new
```

[`probe-gemma4.py`](probe-gemma4.py) verifies base shard hashes against the saved
installation manifest, records binary/adapter/config/tokenizer hashes, generates
two text samples, then probes exact repeated-token prompt lengths in fresh
processes. It doubles from 1,024 up to 32,768 prompt tokens and refines the first
CUDA OOM bracket to 1,024-token resolution. Each trial has a ten-minute timeout;
timeouts and non-CUDA errors do not establish an OOM upper bound. The script
refuses an occupied GPU or an existing output directory.

Each probe requests two samples and ignores EOS to exercise both prefill and
cached decode. For a prompt of P tokens, the runtime cap is P+2, and P+1 tokens
are actually forwarded; the last sampled token is not fed back. JSON reports
include exact position accounting, synchronized timings, generated IDs/text,
and GPU memory sampled every 0.5 seconds. These are capacity tests, **not**
long-context retrieval-quality tests. Q4 applies to base projections, not KV
caches or activations. The original 262,144-token config and RoPE frequencies
are preserved; a downward-only runtime cap bounds allocations per trial.

Historical results and retained logs are in
[`context-evidence-20261014-v2/summary.json`](context-evidence-20261014-v2/summary.json).
That older binary succeeded at 8,192 prompt tokens and failed at 16,384 with
`CUDA_ERROR_INVALID_VALUE`, not OOM. These results are not an inference memory
maximum.

## Historical Installation

Installed and verified on **2026-10-13** on the existing Azure VM
`gemma-h100-aue` in resource group `RG-GEMMA-H100-AUE` (Australia East).
This completes the model installation task; no HTTP server or persistent
inference service was installed.

## Installed Runtime

- VM size: `Standard_NC40ads_H100_v5`.
- GPU: NVIDIA H100 NVL, 95,830 MiB reported memory.
- NVIDIA driver: `615.71.09`; CUDA toolkit: `12.8` (`nvcc` V12.8.93).
- Rust: `1.96.0`.
- Model: `google/gemma-4-31B-it`.
- Checkpoint revision: `842da3794eaa0b77d5f08bae87a17459d91ff475`.
- Execution: Candle's experimental dense text path, CUDA BF16, one GPU.
- Runtime context: 4,096 tokens, including prompt and generation. The downloaded
  configuration and its RoPE settings remain unchanged; the context override is
  in a separate runtime configuration.

The existing VM already contained prerequisites, source, and downloaded weights.
A checksum-based rsync dry run confirmed that its source matched the local Candle
working tree, including the uncommitted Gemma4 changes, excluding `.git` and
`target`. The missing CUDA executable was built using `build-candle.sh`:

```sh
cargo +1.96.0 build --locked --release -p candle-examples --example gemma4 --features cuda
```

Executable SHA-256:
`e80ace993e721ebe37d0dd627541af99090971bdcfb79b4ee922693d40b5936b`.

## Run

Connect using your SSH key, destination, and a verified known-host entry
(replace the placeholders):

```sh
ssh -i /path/to/ssh-private-key \
  -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes USER@HOST
```

On the VM:

```sh
bash /home/azureuser/gemma4/run-gemma4.sh \
  "Explain rotary position embeddings briefly."
```

The wrapper defaults to at most 64 generated tokens, greedy decoding, and a 1.1
repetition penalty. To change the output limit and sampling options:

```sh
SAMPLE_LEN=128 bash /home/azureuser/gemma4/run-gemma4.sh \
  "Write a short explanation of gravity." --temperature 0.7 --top-p 0.9
```

Each invocation loads the model and releases GPU memory when it exits. It uses
local files and needs no Hub credentials or network access for inference.

## Files On The VM

| Path | Contents |
| --- | --- |
| `/home/azureuser/candle` | Matching Candle working-tree source |
| `/home/azureuser/candle/target/release/examples/gemma4` | CUDA release executable |
| `/home/azureuser/gemma4/checkpoint` | Pinned checkpoint, tokenizer, original config |
| `/home/azureuser/gemma4/config-4096.json` | Bounded runtime config |
| `/home/azureuser/gemma4/checkpoint-manifest.json` | Revision and verified shard hashes |
| `/home/azureuser/gemma4/run-gemma4.sh` | Local inference launcher |
| `/home/azureuser/gemma4/build-candle.sh` | Build script; run from the Candle directory |
| `/home/azureuser/gemma4/install-model.py` | Checkpoint download and verification script |
| `/home/azureuser/gemma4/venv` | Python environment with `huggingface_hub` 2.2.0 |

To reverify the checkpoint against Hub metadata:

```sh
/home/azureuser/gemma4/venv/bin/python /home/azureuser/gemma4/install-model.py
```

This requires Hub access and can resume downloads. Both shards were reverified
successfully against their sizes and SHA-256 metadata during installation:

| Shard | SHA-256 |
| --- | --- |
| `model-00001-of-00002.safetensors` | `eeef8791537bc04f110967c513149e037d2a9ae97d49add7291ebfa62806bbfa` |
| `model-00002-of-00002.safetensors` | `018912220f559f7025d60333e0996183cd538aa77ad6f4988a89ce47be681f10` |

## Verification

The CUDA build and executable `--help` check exited successfully. Two real
checkpoint inference checks also exited successfully:

| Prompt | Result |
| --- | --- |
| `Explain rotary position embeddings briefly.` | Coherent RoPE explanation; 64 tokens at 5.94 tokens/s, stopped at the configured output limit |
| `What is 2 + 2? Reply with only the number.` | Answer `4`; 2 generated tokens including the stop token, stopping before the 32-token limit |

The first model load took 11.63 seconds. During that run, `nvidia-smi` confirmed
the Gemma4 executable on the GPU with 59,272 MiB allocated at the sampled instant;
this is not a peak-memory measurement. GPU allocation returned to zero after exit.
The reported token rates include prefill and are smoke-test observations, not
benchmarks.

Build and first-inference logs are available in the VM journal while retained:

```sh
sudo journalctl -u gemma4-cuda-build --no-pager
sudo journalctl -u gemma4-inference-smoke --no-pager
```

These are completed transient jobs, not serving daemons. This verification adds
full-checkpoint GPU smoke tests to the earlier CPU-only evidence in the Gemma4
example README. It does not establish BF16 reference parity, full 4,096-token
capacity, long-context performance, multimodal support, or production readiness.
No model implementation changes were needed for this installation.

## Historical Head-Only Run

Before the all-layer refactor, **step 7 was only partially satisfied**: the
full-sized 31B text decoder ran on one H100 with frozen Q4_0 attention/MLP weights,
but training updated an output-head adapter only. The results in this section
are historical head-only evidence, not all-layer LoRA results. The completed
steps 1-6 and the saved spec are unchanged.

On 2026-10-13, the existing 20-step run was recovered and verified, followed by a
new one-step checkpoint warm start on the same GPU. No model reinstall or
architecture changes were needed. Logs and metadata are stored locally in
[`training-evidence-20261013/`](training-evidence-20261013/README.md).

| Setting / observation | Value |
| --- | --- |
| Base | `google/gemma-4-31B-it`, same pinned revision as above |
| Text decoder parameters | 30,697,345,340 |
| Quantized projections | 410, Candle GGML Q4_0 |
| Compute / optimizer | BF16 / F32 AdamW |
| Trainable parameters | 2,140,160, rank-8 output-head adapter only |
| Sequence length / batch size | 32 / 1 |
| Original run | 20 steps, 640 training tokens including repeated windows |
| Validation loss | 8.912214 initially; 4.572752 after 20 steps |
| Warm start | Reloaded loss exactly 4.5727516412734985; one additional update completed |
| Warm-start final validation loss | 4.501555 |
| Original sampled maximum GPU allocation | 59,699 MiB, including dense loading/conversion |
| Original final GPU sample | 22,323 MiB; not an exact training peak |
| Runtime including loading and quantization | Original: 2m17.78s; warm start: 2m18.54s |
| Exit status | 0 for both training invocations |

The original input is the repository's repetitive 12-document toy text fixture,
processed with the **real checkpoint tokenizer**, yielding seven training windows
and two validation windows. The losses demonstrate pipeline operation, not
generalization or useful model training. No production corpus was supplied.

### Reproduce The Smoke Run

On the VM, choose a new output directory; the wrapper refuses to overwrite one:

```sh
MODE=quantized-head LEARNING_RATE=0.001 STEPS=20 SEQUENCE_LENGTH=32 bash /home/azureuser/gemma4/train-gemma4.sh \
  /home/azureuser/candle/candle-examples/examples/gemma4-training/fixtures/train.jsonl \
  /home/azureuser/gemma4/head-q4-bf16-smoke-new
```

The original adapter is at
`/home/azureuser/gemma4/head-q4-bf16-smoke-20261013/run/checkpoint-20/adapter.safetensors`.
The new warm-start checkpoint is at
`/home/azureuser/gemma4/head-q4-bf16-smoke-20261013/warm-start/checkpoint-1/adapter.safetensors`.
The exact warm-start invocation is recorded in the copied `warm-start.log`.
These are adapter-only checkpoints: retain the original base weights and use
the training CLI's `--adapter-checkpoint` option, not its `--checkpoint` option,
to reload an adapter. Optimizer state is not restored. The inference CLI does
not currently load these adapters.

### Checks And Remaining Limits

All 24 training-example tests passed with `--features cuda`, including CUDA
numerical-gradient and quantized-adapter update tests. The binary, input, and
configuration matched the original run's SHA-256 manifest before and after the
warm start; the repeated check is saved in `post-run-checks.log`.
A checksum-based source comparison found only the training README differed
between local and remote Candle trees (see `source-comparison.log`). Adapter
hashes differ between steps 10,
20, and the warm-start checkpoint. GPU allocation returned to zero after exit;
no training job was left running.

Embeddings, the base output head, and norms remain dense. In this historical
run, the entire BF16 base was loaded before quantization, with a roughly 58 GiB
loading requirement. The refactored loader quantizes each decoder layer before
loading the next. GPU samples were collected once per second for the
original run only and cannot establish instantaneous peaks or longer-context
capacity.

For full-parameter training, this pipeline's F32 weights, gradients, and AdamW
moments would require roughly 496 GB for 31B parameters before activations.
The head-only mode provides neither transformer-layer adapters nor quantized
input gradients. The new `quantized-lora` mode supplies both without changing
standard Candle `QMatMul`. Useful training still requires an approved, cleaned
corpus and a training/evaluation budget.
