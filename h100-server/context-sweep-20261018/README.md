# Input And Output Context Measurements

Historical snapshot from the job on `gemma-h100-aue`, collected on
2026-10-18 after the input sweep completed at 02:03:53 UTC. At collection, the
full-output sweep was incomplete: `output-32768` was active, with the other two
output trials queued. These statuses are not current runtime state; output
maxima remain incomplete in this retained evidence. No model, adapter, or
running benchmark was changed to collect this snapshot.

## Environment And Method

- Azure `Standard_NC40ads_H100_v5`, one NVIDIA H100 NVL, 95,830 MiB VRAM
  (93.58 GiB), 40 logical CPUs, 322,237.89 MiB host RAM (314.69 GiB).
- `google/gemma-4-31B-it`, revision
  `842da3794eaa0b77d5f08bae87a17459d91ff475`, Candle Q4_0 decoder projections,
  BF16 compute, existing step-20 all-layer LoRA adapter, batch one, no
  FlashAttention. KV caches are not 4-bit.
- Synthetic repeated-token inputs, exact token counts, EOS ignored. These are
  capacity tests, not tests of useful answers or long-context retrieval quality.
- Approximately one-second telemetry samples include loading and process exit.
  GPU utilization is device-wide; host CPU is normalized over all 40 logical
  CPUs. Process CPU uses 100% per logical CPU. Peaks may miss transients.
- Host RAM used is `MemTotal - MemAvailable`. Process RSS includes resident
  file-backed model mappings, including reclaimable pages counted as available
  by Linux, so its loading peak can exceed the reported host RAM-used metric.
- Initial input trials generate two tokens and use a runtime cap of input + 2.
  Full-output trials use the full 262,144-token cap and can allocate more VRAM.

## Measured Input Load

All four rows successfully generated two tokens. GPU and CPU columns show
whole-process sample mean / maximum, including loading. Memory columns show
sampled peaks in GiB (1,024 MiB). Prefill excludes loading.

| Exact input tokens | Prefill seconds | GPU mean / max % | Peak VRAM GiB | Host CPU mean / max % | Peak host RAM used GiB | Peak process RSS GiB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32,768 | 40.116 | 24.94 / 100 | 25.11 | 2.58 / 4.72 | 6.06 | 58.44 |
| 65,536 | 109.279 | 45.38 / 100 | 28.27 | 2.57 / 4.07 | 6.05 | 58.56 |
| 131,072 | 378.469 | 72.98 / 100 | 34.02 | 2.58 / 4.92 | 6.07 | 58.56 |
| 262,142 | 1,736.924 | 92.25 / 100 | 45.37 | 2.57 / 4.76 | 6.05 | 58.44 |

The approximately 132-133 seconds of loading per trial strongly lowers GPU
averages in shorter trials. Final-60-second GPU averages are 66.05%, 96.81%,
97.66%, and 99.85%, respectively, but these windows are not exact inference
phases and may include loading or process teardown. The process averaged about
one logical CPU in all trials. No swap use was sampled.

The largest tested input is **262,142**, not exactly 262,144: two tokens were
reserved for generation to exercise cached decode. This reaches the declared
total budget without establishing an OOM limit or measuring exact full-input
prefill.

## Evidence

The VM job root at the 2026-10-18 snapshot was
`/home/azureuser/gemma4/context-sweep-20261018T0146`. Its `results/summary.json`
recorded the active trial and output-test outcomes. The near-262k measurement is
reused from `/home/azureuser/gemma4/context-load-20261018` after validation of
matching binary, model, and adapter hashes and exact successful token accounting.

Local `results/` contains a point-in-time copy of summaries, inference reports,
and logs, not a live view. Local `boundary/` contains the completed near-262k
summary, inference report, and raw telemetry. Raw input telemetry is retained
with each copied input trial. At collection, active output telemetry was left
on the VM rather than copied while growing; its current availability is unverified.

## Output Limits

Statuses below are as of the **2026-10-18 snapshot**, not a live job view.

| Exact input tokens | Architectural output budget | Measured output maximum |
| ---: | ---: | --- |
| 32,768 | 229,376 | Test running; 2 tokens verified in completed input test |
| 65,536 | 196,608 | Pending; 2 tokens verified |
| 131,072 | 131,072 | Pending; 2 tokens verified |
| 262,142 | 2 | 2, verified under the runtime policy |
| 262,144 | 0 | Architectural accounting only; exact full-input prefill not measured |

Input and output share the 262,144-token budget; it is not a separate output
allowance. The final sampled token is not forwarded, so input P and output G
exercise P + G - 1 forwarded positions. The near-262k test generated two special
tokens with empty decoded text, not a meaningful two-token answer.

The serial runner was configured to attempt all three remaining output budgets,
with a 48-hour timeout per trial. Only an exact successful completion establishes a
maximum under this runtime policy. A timeout or failure is inconclusive, not a
measured maximum. The job running at collection was left undisturbed.

## Validation

Retained raw telemetry reproduces the completed trials' sample aggregates.
Successful reports have exact input/output/forwarded token accounting, exit
code zero, no timeout, and no telemetry error. All trials record the same
binary/model/adapter hashes; original remote artifacts were not independently
rehashed during this read-only collection. The older boundary monitor has a
different script hash, but its raw aggregates were also checked.

The local inference-monitor and sweep regression suite passed all 23 tests:

```sh
python3 -B -m unittest discover -s h100-server -p 'test_*inference*.py' -v
```
