# Inference Context Measurement

Observed via Azure CLI and read-only SSH on 2026-10-17 at approximately
08:09 UTC. This is an interim measurement report, not a completed maximum-context
search. No training, competing inference job, or utilization monitoring was
started; the existing inference sweep was left unchanged.

## Result

**Largest completed prompt: 131,072 tokens (128 Ki tokens), with two generated
tokens and one cached decode step. The actual maximum remains undetermined.**

The model declares a 262,144-token context. A 262,142-token prompt trial with
a 262,144-token runtime budget was still running at observation time, roughly
28 minutes after its process started. Neither success nor failure has been
established for that trial. Its supervisor permits 12,000 seconds per trial
and requests a subsequent successful-boundary confirmation.

## Completed Trials

| Prompt tokens | Result | Prefill (s) | Cached decode (s) | Total trial (s) |
| ---: | --- | ---: | ---: | ---: |
| 32,768 | Success | 40.26 | 0.094 | 174.98 |
| 65,536 | Success | 109.57 | 0.129 | 245.53 |
| 131,072 | Success | 378.71 | 0.200 | 513.73 |
| 262,142 | Running at observation time | Not available | Not available | Not final |

Total trial times include fresh model loading, approximately 133 seconds.
Both preceding short generation checks also succeeded. The 131,072-token
result has not yet been repeated by this sweep's confirmation phase.

## Conditions

- VM: `gemma-h100-aue`, resource group `RG-GEMMA-H100-AUE`, verified running.
- Hardware: one NVIDIA H100 NVL, 95,830 MiB device memory; VM size
  `Standard_NC40ads_H100_v5`.
- Model: `google/gemma-4-31B-it`, revision
  `842da3794eaa0b77d5f08bae87a17459d91ff475`.
- Adapter: `/home/azureuser/gemma4/lora-q4-bf16-smoke-20261013/run/checkpoint-20`.
- Candle text-only inference: Q4_0 decoder projections, BF16 compute, frozen
  adapters, batch size one. Four-bit weights do not imply four-bit KV caches.
- Executable SHA-256:
  `f87991d7995ea9ee3003fae8716d1e3396a6f1f777fe6762a9de31d092da7f01`.
- Synthetic repeated-token prompts, greedy decoding, EOS ignored, two samples
  requested, fresh process per trial, FlashAttention disabled.
- Forward timings are device-synchronized. The saved probe verifies prefill
  and cached decode, not merely allocation or tokenization.

For prompt length P, these probes request a P+2 context budget but actually
forward P+1 tokens: the final sampled token is not fed back. Thus the largest
completed trial used a 131,074-token budget and actually forwarded 131,073
tokens, through zero-based position 131,072. This is not validation of long
generated responses or long-context retrieval quality.

The October 16 report used a different executable and only confirmed 32,768
prompt tokens. Its timeout at 65,534 is not an upper bound for this newer
binary. Do not merge timings across those sweeps. No OOM upper bound appears
in the current snapshot either.

## Evidence

Unmodified evidence copied from the VM:

- `context-inference-20261017-0809-summary.json`: interim sweep snapshot;
  intentionally has no `finished_utc` or final boundary conclusion.
- `context-inference-20261017-0809-probe-131072.json`: completed largest-prompt
  report, including exact token accounting and inference configuration.

The sweep summary records base shard verification against the local checkpoint
manifest and hashes for the executable, adapter, configuration, and tokenizer.
Those large shards were not rehashed during this read-only inspection.

Authoritative ongoing VM evidence:
`/home/azureuser/gemma4/context-optim-1792221640-93574/summary.json`.
At observation time, supervisor PID 93574 was running the existing
`probe-gemma4.py` with `--start-prompt 32768 --max-prompt 262142 --timeout 12000
--confirm-boundary`; inference PID 98398 was executing `probe-262142`.
Its log showed model loading completed but no completed inference report.

The remaining measurement is to read the 262,142-token result and subsequent
confirmation when this existing job finishes. If it times out, report that
as inconclusive, not OOM or proof of an architectural limit. Until then,
131,072 prompt tokens is a demonstrated lower bound, not a measured maximum.
