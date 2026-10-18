#!/usr/bin/env python3
"""Measure single-H100 all-layer Q4 LoRA training capacity using real optimizer steps."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def gpu_query(query):
    return subprocess.check_output(
        ["nvidia-smi", "-i", "0", query, "--format=csv,noheader,nounits"], text=True
    ).strip()


def validate_run(run, length, steps, initial_adapter_hash):
    """A zero exit alone does not prove the requested training workload completed."""
    state = json.loads((run / f"checkpoint-{steps}" / "training-state.json").read_text())
    expected = {
        "mode": "quantized-lora", "quantization": "Q4_0", "dtype": "bf16",
        "master_dtype": "f32", "optimizer_dtype": "f32", "loss_dtype": "f32",
        "optimizer": "AdamW", "lora_rank": 8, "lora_alpha": 8,
        "batch_size": 1, "gradient_accumulation_steps": 4,
        "learning_rate": 0.0001, "weight_decay": 0, "max_grad_norm": 1,
        "sequence_length": length, "step": steps, "steps_requested": steps,
        "tokens_seen": steps * 4 * length, "quantized_projections": 410,
        "eval_batches": 2,
    }
    for key, value in expected.items():
        if state.get(key) != value:
            raise ValueError(f"training policy/completion mismatch: {key}={state.get(key)!r}, expected {value!r}")
    metrics = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    if [row["step"] for row in metrics] != list(range(steps + 1)):
        raise ValueError("missing optimizer updates")
    for step, row in enumerate(metrics):
        required = ["validation_loss"] if step == 0 else ["train_loss", "gradient_norm"]
        if step and (step % 10 == 0 or step == steps):
            required.append("validation_loss")
        if any(not isinstance(row.get(key), (int, float)) or not math.isfinite(row[key]) for key in required):
            raise ValueError("missing or nonfinite training/evaluation metrics")
        if step and (row["tokens_seen"] != step * 4 * length or row["gradient_norm"] <= 0):
            raise ValueError("invalid token accounting or zero gradient norm")
    final_hash = sha256(run / f"checkpoint-{steps}" / "adapter.safetensors")
    if final_hash == initial_adapter_hash:
        raise ValueError("adapter did not change")
    return {"final_metrics": metrics[-1], "adapter_sha256": final_hash, "policy_verified": expected}


def search(probe, start, maximum, resolution):
    """Only actual CUDA OOM supplies an upper bound; all other failures stop search."""
    lower, upper = 0, None
    length = start
    while True:
        status = probe(length)
        if status == "success":
            lower = length
            if length == maximum:
                break
            length = min(length * 2, maximum)
        elif status == "cuda_oom":
            upper = length
            break
        else:
            return lower, upper, False
    while upper is not None and upper - lower > resolution:
        mid = ((lower + upper) // (2 * resolution)) * resolution
        if mid <= lower:
            break
        status = probe(mid)
        if status == "success":
            lower = mid
        elif status == "cuda_oom":
            upper = mid
        else:
            return lower, upper, False
    return lower, upper, True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="new evidence directory; parent must exist")
    parser.add_argument("--binary", type=Path, default=Path("/home/azureuser/candle/target/release/examples/gemma4-training"))
    parser.add_argument("--adapter", type=Path, default=Path("/home/azureuser/gemma4/lora-q4-bf16-smoke-20261013/run/checkpoint-20"))
    parser.add_argument("--start", type=int, default=128)
    parser.add_argument("--max-context", type=int, default=4096)
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--timeout", type=int, default=1800, help="seconds per training process")
    args = parser.parse_args()
    if not (0 < args.resolution <= args.start <= args.max_context) or args.timeout <= 0:
        parser.error("require 0 < resolution <= start <= max-context and positive timeout")
    if args.start % args.resolution or args.max_context % args.resolution:
        parser.error("start and max-context must be multiples of resolution")
    if gpu_query("--query-compute-apps=pid"):
        parser.error("GPU 0 is occupied; refusing to start")
    gpu = gpu_query("--query-gpu=name,memory.total,driver_version")
    if "H100" not in gpu:
        parser.error("GPU 0 is not an H100")
    args.output = args.output.resolve()
    args.adapter = args.adapter.resolve()
    args.binary = args.binary.resolve()
    state = json.loads((args.adapter / "training-state.json").read_text())
    source = Path(state["source_checkpoint"])
    config = json.loads((source / "config.json").read_text())
    text_config = config.get("text_config", config)
    if args.max_context > text_config["max_position_embeddings"]:
        parser.error("max-context exceeds source model context")
    args.output.mkdir(exist_ok=False)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0")
    env["LD_LIBRARY_PATH"] = "/usr/local/cuda-12.8/lib64:" + env.get("LD_LIBRARY_PATH", "")
    summary = {
        "format_version": 1, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu": gpu, "declared_model_context": text_config["max_position_embeddings"],
        "method": "Fresh process and warm start from the same trained adapter per trial; optimizer resets. Two optimizer updates for screening, then 20-update boundary confirmation and repeat upper-bound OOM. Synthetic repeated text, not a quality test. Context is model input tokens; each window also has one shifted target token. No architecture or RoPE-frequency changes.",
        "memory_convention": "Whole-device memory.used sampled every 0.5 seconds, including loading; not instantaneous peak.",
        "resolution_tokens": args.resolution, "search_cap_tokens": args.max_context,
        "provenance_sha256": {}, "trials": [],
    }

    def save():
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    for path in [args.binary, Path(__file__), args.adapter / "adapter.safetensors",
                 args.adapter / "training-state.json", args.adapter / "config.json",
                 source / "config.json", source / "tokenizer.json"]:
        summary["provenance_sha256"][str(path)] = sha256(path)
    initial_hash = summary["provenance_sha256"][str(args.adapter / "adapter.safetensors")]
    manifest = source.parent / "checkpoint-manifest.json"
    provenance = json.loads(manifest.read_text())
    if provenance["model_id"] != "google/gemma-4-31B-it" or provenance["revision"] != "842da3794eaa0b77d5f08bae87a17459d91ff475":
        raise SystemExit("Base manifest model/revision mismatch")
    index = source / "model.safetensors.index.json"
    for path in [manifest, index]:
        summary["provenance_sha256"][str(path)] = sha256(path)
    for shard in sorted(set(json.loads(index.read_text())["weight_map"].values())):
        if Path(shard).name != shard:
            raise SystemExit("Invalid shard filename")
        actual = sha256(source / shard)
        if actual != provenance["sha256"][shard]:
            raise SystemExit(f"Base shard hash mismatch: {shard}")
        summary["provenance_sha256"][str(source / shard)] = actual
    summary["base_checkpoint_manifest"] = provenance
    summary["base_shards_verified_against_local_manifest"] = True
    save()

    def trial(length, steps=2, label="screen"):
        if gpu_query("--query-compute-apps=pid"):
            raise RuntimeError("GPU occupied before trial; refusing to interfere")
        directory = args.output / f"{label}-{length}"
        directory.mkdir()
        # Real tokenizer and preprocessing; enough text in both splits for exact S+1 windows.
        input_path = directory / "synthetic.jsonl"
        with input_path.open("w") as stream:
            for i in range(8):
                stream.write(json.dumps({"text": f"Document {i}. " + "The quick brown fox jumps over the lazy dog. " * (length + 1)}) + "\n")
        preprocess = [str(args.binary), "preprocess", "--input", str(input_path),
                      "--tokenizer", str(source / "tokenizer.json"), "--output", str(directory / "data"),
                      "--sequence-length", str(length), "--vocab-size", str(text_config["vocab_size"]),
                      "--validation-fraction", "0.25", "--seed", "42", "--bos-token-id", "2", "--eos-token-id", "1"]
        with (directory / "preprocess.log").open("w") as log:
            subprocess.run(preprocess, stdout=log, stderr=subprocess.STDOUT, env=env, timeout=120, check=True)
        metadata = json.loads((directory / "data/metadata.json").read_text())
        if metadata["train"]["windows"] < 4 or metadata["valid"]["windows"] < 2:
            raise RuntimeError("insufficient packed windows")
        command = [str(args.binary), "train", "--config", str(source / "config.json"),
                   "--checkpoint", str(source), "--adapter-checkpoint", str(args.adapter),
                   "--dataset", str(directory / "data"), "--output", str(directory / "run"),
                   "--mode", "quantized-lora", "--quantization", "q4-0", "--precision", "bf16",
                   "--lora-rank", "8", "--lora-alpha", "8", "--max-parameters", "32000000000",
                   "--steps", str(steps), "--batch-size", "1", "--gradient-accumulation-steps", "4",
                   "--learning-rate", "0.0001", "--weight-decay", "0", "--max-grad-norm", "1",
                   "--eval-every", "10", "--eval-batches", "2", "--save-every", "10", "--seed", "42", "--cuda"]
        item = {"name": directory.name, "sequence_length": length, "steps": steps,
                "command": command, "preprocess_command": preprocess,
                "input_sha256": sha256(input_path), "dataset_sha256": {
                    name: sha256(directory / "data" / name) for name in ["metadata.json", "train.bin", "valid.bin"]}}
        start, samples, timed_out = time.monotonic(), [], False
        print(f"Starting {directory.name}: {steps} optimizer updates", flush=True)
        with (directory / "train.log").open("w") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
            try:
                while process.poll() is None:
                    samples.append([round(time.monotonic() - start, 3), int(gpu_query("--query-gpu=memory.used"))])
                    if time.monotonic() - start > args.timeout:
                        timed_out = True
                        break
                    time.sleep(0.5)
            finally:
                if process.poll() is None:
                    process.kill()
                process.wait()
        log_text = (directory / "train.log").read_text()
        status = "timeout" if timed_out else "success" if process.returncode == 0 else "cuda_oom" if "CUDA_ERROR_OUT_OF_MEMORY" in log_text else "error"
        if status == "success":
            try:
                item.update(validate_run(directory / "run", length, steps, initial_hash))
            except (ValueError, KeyError, OSError) as error:
                status = "invalid_result"
                item["validation_error"] = str(error)
        item.update(status=status, exit_code=process.returncode, wall_seconds=time.monotonic() - start,
                    sampled_max_gpu_mib=max((s[1] for s in samples), default=None))
        if status != "success":
            item["error_excerpt"] = log_text[-4000:]
        (directory / "gpu.json").write_text(json.dumps(samples) + "\n")
        summary["trials"].append(item)
        save()
        print(json.dumps(item), flush=True)
        return status

    lower, upper, complete = search(trial, args.start, args.max_context, args.resolution)
    summary.update(largest_screened_success_tokens=lower or None, smallest_screened_oom_tokens=upper,
                   search_completed=complete, confirmed_training_tokens=None, repeated_oom_tokens=None)
    save()
    if complete and lower:
        if trial(lower, steps=20, label="confirm") == "success":
            summary["confirmed_training_tokens"] = lower
        if upper is not None and trial(upper, label="repeat-oom") == "cuda_oom":
            summary["repeated_oom_tokens"] = upper
    summary["gpu_processes_after"] = gpu_query("--query-compute-apps=pid")
    summary["gpu_memory_mib_after"] = int(gpu_query("--query-gpu=memory.used"))
    summary["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    summary["conclusion"] = "Report the confirmed lower bound and repeated CUDA-OOM upper bound, not an exact token maximum unless adjacent. No claim about sustained long runs or long-context quality. Errors/timeouts are inconclusive."
    save()


if __name__ == "__main__":
    main()
