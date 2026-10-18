# GPU and Inference Context Sweep Poll

Observed on 2026-10-16 around 05:39 UTC. This directory is a point-in-time
copy of an **unfinished** remote sweep, not its final results.

## Workload

- Azure CLI confirmed `gemma-h100-aue` in `RG-GEMMA-H100-AUE` was running.
- GPU: one NVIDIA H100 NVL, 95,830 MiB total device memory.
- Model: `google/gemma-4-31B-it`, revision `842da3794eaa0b77d5f08bae87a17459d91ff475`.
- Adapter: `/home/azureuser/gemma4/lora-q4-bf16-smoke-20261013/run/checkpoint-20`.
- Service: `gemma4-context-probe-20261016.service`, active/running, supervisor PID 3004.
- Current inference PID: 12129, running `confirm-success-32768`.
- Sweep started at 04:54:09 UTC with a 1,200-second timeout per trial.
- Remote evidence: `/home/azureuser/gemma4/context-evidence-20261016`.

## Live GPU Poll

Read-only `nvidia-smi` polling every two seconds for 60 seconds produced 30
samples from 05:38:39.925 through 05:39:37.953 UTC:

| Metric | Observed range |
| --- | --- |
| GPU utilization | 100% in all 30 samples |
| Device memory used | 40,115-42,803 MiB |
| Memory-controller utilization | 21-38% |
| Power draw | 283.98-322.22 W |
| GPU temperature | 66-76 C |

The bounded monitor exited on its timeout as intended. The inference workload
was not stopped, restarted, or modified. These ranges cover only the polling
interval, not the full sweep. Per-trial GPU JSON files contain the sweep's
separate memory samples, not this live utilization poll.

## Completed Capacity Trials

| Prompt tokens | Status | Sampled peak device memory | Prefill | Total trial time |
| ---: | --- | ---: | ---: | ---: |
| 16,384 | Success | 33,107 MiB | 37.82 s | 171.21 s |
| 32,768 | Success | 44,755 MiB | 150.15 s | 284.41 s |
| 65,534 | Timeout, not recorded as CUDA OOM | 67,701 MiB | Not completed | 1,200.36 s |

Both short generation checks also succeeded. The 32,768-token confirmation
repeat was still running at the final process check; its log in this snapshot
does not establish success or failure. `summary.json` has no finish timestamp.

The largest completed successful prompt is **32,768 tokens**. That trial
generated two tokens, used a runtime context budget of 32,770, and actually
forwarded 32,769 tokens. The final generated token was not forwarded.

The 65,534-token trial exceeded the wall-clock limit, so it does **not** establish
a memory-capacity upper bound. No CUDA-OOM boundary has been measured in this
run, and the model's declared 262,144-token context remains unvalidated on this
setup. Synthetic repeated-token capacity tests do not establish long-context
retrieval quality or capacity for a long generated response.

See `summary.json` for commands, checkpoint verification, binary/adapter hashes,
timings, token accounting, and the timeout excerpt. This newer binary's results
must not be combined with the older 2026-10-14 sweep as if they were one run.
