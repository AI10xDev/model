# H100 Telemetry

Publication of retained measurements, not a new benchmark or live VM poll.
The latest sweep snapshot was collected on **2026-10-18 at 02:03:53 UTC**.
See the [detailed report](h100-server/context-sweep-20261018/README.md) and
[sweep state](h100-server/context-sweep-20261018/results/summary.json).

## Environment

| Setting | Recorded value |
| --- | --- |
| Host | Azure Standard_NC40ads_H100_v5, one NVIDIA H100 NVL |
| Hardware capacity | 95,830 MiB VRAM; 40 logical CPUs; 322,237.89 MiB host RAM |
| Driver / toolkit | 615.71.09 / CUDA 12.8 |
| Model | `google/gemma-4-31B-it` |
| Model revision | `842da3794eaa0b77d5f08bae87a17459d91ff475` |
| Inference policy | Q4_0 decoder projections, BF16 compute, step-20 all-layer LoRA, batch one |
| Attention | No FlashAttention; KV caches are not 4-bit |
| Input | Synthetic repeated tokens, exact lengths, EOS ignored |
| Declared context | 262,144 tokens shared by input and generated output |

## Input Sweep

Every row completed successfully with **two generated tokens**. GPU and host CPU
are whole-process sample mean / maximum, including model loading and teardown.
VRAM, RAM and RSS are sampled peaks. Prefill timing excludes loading.

| Exact input tokens | Prefill seconds | GPU mean / max % | Peak VRAM MiB | Host CPU mean / max % | Peak host RAM GiB | Peak process RSS GiB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32,768 | 40.116 | 24.94 / 100 | 25,717 | 2.58 / 4.72 | 6.06 | 58.44 |
| 65,536 | 109.279 | 45.38 / 100 | 28,949 | 2.57 / 4.07 | 6.05 | 58.56 |
| 131,072 | 378.469 | 72.98 / 100 | 34,839 | 2.58 / 4.92 | 6.07 | 58.56 |
| 262,142 | 1,736.924 | 92.25 / 100 | 46,455 | 2.57 / 4.76 | 6.05 | 58.44 |

Sampling was approximately once per second and can miss transient peaks.
Loading took about 132-133 seconds per trial, lowering short-trial GPU averages.
Host CPU is normalized over all 40 logical CPUs; process CPU uses 100% per core
and averaged approximately one core. No swap use was sampled.

Host RAM is `MemTotal - MemAvailable`; process RSS includes resident file-backed
model mappings. Reclaimable mapped pages can count as available to Linux, so
process RSS and host RAM-used are not directly comparable. GiB = 1,024 MiB.

The largest measured input is **262,142**, not 262,144. Two tokens were reserved
for generation. This is successful capacity at the declared total budget, not
an OOM boundary or a long-context quality result. The near-262k run generated two
special tokens with **empty decoded text**.

## Output Lengths

The following statuses describe the retained snapshot, not current VM state.

| Input tokens | Architectural output budget | Retained measured result |
| ---: | ---: | --- |
| 32,768 | 229,376 | Full-output test running at snapshot time; only 2 outputs verified |
| 65,536 | 196,608 | Full-output test pending at snapshot time; only 2 outputs verified |
| 131,072 | 131,072 | Full-output test pending at snapshot time; only 2 outputs verified |
| 262,142 | 2 | 2 outputs verified, the maximum allowed by this runtime policy |
| 262,144 | 0 | Architectural accounting only; exact full-input prefill not measured |

**The first three output maxima are not established by this evidence.** A
requested budget, timeout, or incomplete job is not a measured maximum. With P
input tokens and G generated tokens, forwarded tokens equal `P + G - 1`: the
last sampled token is not forwarded. Input and output do not get separate
262,144-token allowances. Full-output trials can allocate more VRAM than the
two-output input trials.

## Raw Evidence

Each directory contains `summary.json`, `inference.json`, and the raw
`telemetry.jsonl`; summaries retain commands and recorded artifact hashes.

| Input tokens | Evidence directory | Raw samples |
| ---: | --- | ---: |
| 32,768 | [input-32768](h100-server/context-sweep-20261018/results/input-32768/) | 170 |
| 65,536 | [input-65536](h100-server/context-sweep-20261018/results/input-65536/) | 237 |
| 131,072 | [input-131072](h100-server/context-sweep-20261018/results/input-131072/) | 499 |
| 262,142 | [boundary](h100-server/context-sweep-20261018/boundary/) | 1,815 |

The boundary measurement was reused after the original runner checked matching
binary/model/adapter provenance and exact success accounting. Raw evidence is
preserved unchanged; historical remote paths are provenance, not working links.
Active output telemetry was not copied while growing. The empty retained output
log is not completion evidence. Monitor implementation:
[measure-inference-load.py](h100-server/measure-inference-load.py); serial sweep:
[sweep-inference-output.py](h100-server/sweep-inference-output.py).

## Training Snapshot

The [recovered all-layer run](h100-server/lora-training-evidence-20261013/README.md)
records 20 optimizer updates on the full-sized text decoder, using sequence 32,
microbatch one, accumulation four, BF16 compute and F32 AdamW.

| Observation | Value |
| --- | --- |
| Base text-decoder parameters | 30,697,345,340, frozen |
| Trainable adapter parameters | 61,214,720 |
| Coverage | 60 layers, 410 Q4_0 projections, 820 adapter tensors |
| Processed tokens | 2,560 |
| Initial / final validation loss | 8.9070115 / 1.7171376 |
| Final training loss | 0.3049596 |
| Sampled maximum GPU memory | 35,125 MiB, including loading |
| Wall time / exit status | 2m54.99s / 0 |

Raw [metrics](h100-server/lora-training-evidence-20261013/recovered/run/metrics.jsonl),
[GPU samples](h100-server/lora-training-evidence-20261013/recovered/gpu.csv), and
[training log](h100-server/lora-training-evidence-20261013/recovered/train.log)
are included. Historical verification reported changes to all 820 tensors between
steps 10 and 20; the adapter weights are not in this publication, so that check
cannot be repeated from these logs alone.

This is parameter-efficient fine-tuning, **not full-parameter training of 31B
weights**. The repetitive toy fixture does not establish useful generalization.
Separate [training-context screening](h100-server/training-context-evidence-20261014/summary.json)
recorded two-update success at sequence 128 (51,859 MiB peak) and 256 (83,381 MiB),
and CUDA OOM at 512 (94,227 MiB sampled maximum). No completed 20-update boundary
confirmation is retained; an exact maximum training context is not established.

Older dated reports under `h100-server/` used earlier binaries/settings and are
historical evidence, not additional rows in the current inference table.
