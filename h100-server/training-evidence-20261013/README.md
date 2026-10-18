# Training Evidence

Logs and metadata copied from the single-H100 output-head training smoke run on
2026-10-13. The input is the existing repetitive toy text fixture, tokenized with
the installed `google/gemma-4-31B-it` tokenizer, not a production training corpus.

The full-sized base is frozen; only the output-head adapter is trained. These
artifacts do not establish full-parameter or all-layer QLoRA training.

The original base and binary hashes/revision are recorded in provenance files.
The adapter weights and full run directories remain on the VM; no pretrained
weights or tokenizer are copied into this evidence directory.

See `../README.md` for invocation, outcome, and limitations.

## Artifacts

| File | Evidence |
| --- | --- |
| `train.log` | Original 20-step invocation, losses, timing, exit status 0 |
| `run/metrics.jsonl` | Initial validation and all 20 optimization steps |
| `run/run-config.json` | Full parameter count, Q4_0/BF16 policy, frozen base and adapter scope |
| `run/checkpoint-{10,20}/training-state.json` | Saved checkpoint progress and policy |
| `data/metadata.json`, `preprocess.log` | Real-tokenizer preprocessing, fixture input, split/window counts |
| `gpu.csv` | Original run's one-second GPU samples, including dense-load memory |
| `provenance.sha256` | Original binary, input and source-config hashes; rechecked before warm start |
| `base-checkpoint-manifest.json` | Installed base revision and shard hashes |
| `cuda-tests.log` | Fresh CUDA-enabled training-example test run: 24 passed |
| `warm-start.log` | Fresh full-checkpoint adapter reload, one update, exit status 0 |
| `warm-start/metrics.jsonl` | Reloaded validation matches original final loss exactly |
| `warm-start/run-config.json`, `warm-start/checkpoint-1/training-state.json` | Warm-start policy and saved progress |
| `adapter-checkpoints.sha256` | Remote adapter hashes, different at each recorded checkpoint |
| `post-run-checks.log` | Repeated provenance checksum verification, zero GPU allocation and no compute processes after exit |
| `source-comparison.log` | Checksum-based rsync dry run excluding `.git` and `target`; only the training README differs |

All remote paths in the evidence refer to the original VM. The warm start resets
AdamW and data order; its step 1 is not an exact optimizer-resumed step 21. The
original run predates this verification; it was not rerun or relabeled as new.
