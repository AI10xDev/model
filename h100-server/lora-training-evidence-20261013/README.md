# All-Layer Q4 Training Evidence

This directory records the 2026-10-13 verification of the full-sized
`google/gemma-4-31B-it` text decoder with Candle Q4_0 projections and trainable
adapters throughout all decoder layers on `gemma-h100-aue`.

The source checkpoint and completed spec items 1-6 were not modified. This is
parameter-efficient fine-tuning, not full-parameter training. The input is the
existing repetitive toy JSONL fixture, tokenized with the real model tokenizer;
it is not evidence of useful generalization.

## Invocation

From the hosted VM, with a fresh output directory:

```sh
STEPS=20 SEQUENCE_LENGTH=32 GRADIENT_ACCUMULATION_STEPS=4 \
  bash /home/azureuser/gemma4/train-gemma4.sh \
  /home/azureuser/candle/candle-examples/examples/gemma4-training/fixtures/train.jsonl \
  /home/azureuser/gemma4/lora-q4-bf16-smoke-20261013
```

The wrapper defaults to `quantized-lora`, Q4_0, BF16 compute, rank/alpha 8,
microbatch size 1, learning rate 0.0001, no weight decay, and F32 AdamW masters
and moments. It records GPU memory once per second. Sampled memory is not an
instantaneous peak, and the bounded context does not establish long-context
capacity.

Build and run jobs use transient systemd units `gemma4-lora-build-20261013` and
`gemma4-lora-train-20261013`. The training unit has a 30-minute runtime limit.
The artifacts below are retained separately from the historical head-only run.

## Reverified 2026-10-16

The existing VM artifacts were checked again with this directory's `verify.py`;
no training was rerun or checkpoint modified. The verifier checked finite F32
adapter values, all 60 layers, exact step/token accounting, and that **all 820
adapter tensors changed between checkpoints 10 and 20**.

| Observation | Value |
| --- | --- |
| Text decoder / trainable parameters | 30,697,345,340 / 61,214,720 |
| Q4_0 projections / adapter tensors | 410 / 820 |
| Sequence length / microbatch / accumulation | 32 / 1 / 4 |
| Optimizer updates / processed tokens | 20 / 2,560 |
| Initial / final validation loss | 8.90701150894165 / 1.7171375751495361 |
| Sampled maximum GPU memory | 35,125 MiB, 177 samples including loading |
| Wall time / exit status | 2m54.99s / 0 |

Recovered raw logs, metrics, checkpoint metadata, GPU samples, and original
provenance are in [`recovered/`](recovered/). Large adapter tensors and base
weights remain on the VM; they were inspected there, not copied into this
directory. Repeat the verification on the VM with:

```sh
python3 /home/azureuser/gemma4/verify-lora-20261016.py \
  /home/azureuser/gemma4/lora-q4-bf16-smoke-20261013
```

This is a completed pipeline smoke run on repetitive toy text, not a production
training result. The validation loss does not demonstrate generalization.
