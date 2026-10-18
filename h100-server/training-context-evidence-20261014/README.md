# Training Context Capacity Evidence

Measured on 2026-10-14 using the existing Azure `gemma-h100-aue` VM and
installed CUDA Candle training executable. This is training capacity, not the
separate inference context benchmark.

## Reproduction

```sh
python3 /home/azureuser/gemma4/probe-gemma4-training.py \
  /home/azureuser/gemma4/training-context-evidence-new
```

The recorded run used the new output directory
`/home/azureuser/gemma4/training-context-evidence-20261014`, managed by the
transient systemd unit `gemma4-training-context-20261014` with a 90-minute
overall runtime limit and a 30-minute limit for each training process.

Policy: full-sized 31B text decoder, frozen Candle Q4_0 projections, all-layer
rank-8/alpha-8 LoRA, BF16 compute, F32 adapter masters/gradients/AdamW/loss,
microbatch 1, accumulation 4, learning rate 0.0001, weight decay 0, and gradient
clipping 1. Evaluation uses two batches; evaluation and saving occur every ten
updates and at the final update. No FlashAttention or activation checkpointing.
Every trial warm-starts the original step-20 adapter, with a fresh optimizer.
The original checkpoint is not modified.

Eight synthetic repeated-text documents per candidate are preprocessed with
the real checkpoint tokenizer, document-level 75/25 splitting, and exact
S+1-token source windows for S input positions and S shifted targets. The
synthetic corpus establishes allocation capacity only, not quality. Candidate
lengths are doubled from 128 until CUDA OOM, then refined to 32-token resolution.
Screening uses two optimizer updates, followed by a 20-update confirmation at
the successful boundary and a repeated trial at the OOM boundary.

`summary.json` retains commands, hashes, GPU identity, trial outcomes, sampled
memory maxima, validated policy, and completed-update metrics. Trial directories
retain training/preprocessing logs, GPU memory samples, dataset metadata,
metrics, and checkpoint state. Large adapter tensors, tokenizer copies, and
token streams remain on the VM rather than being duplicated in this evidence
directory. Whole-device memory sampling is every 0.5 seconds and is not an
instantaneous allocation peak.
