#!/usr/bin/env python3
"""Measure trained Candle inference capacity, not long-context retrieval quality."""
import argparse
import hashlib
import json
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="new evidence directory")
    parser.add_argument("--binary", type=Path, default=Path("/home/azureuser/candle/target/release/examples/gemma4"))
    parser.add_argument("--adapter", type=Path, default=Path("/home/azureuser/gemma4/lora-q4-bf16-smoke-20261013/run/checkpoint-20"))
    parser.add_argument("--timeout", type=int, default=600, help="seconds per fresh process")
    parser.add_argument("--max-prompt", type=int, default=32768)
    parser.add_argument("--start-prompt", type=int, default=1024)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--confirm-boundary", action="store_true", help="repeat successful and OOM boundary trials in fresh processes")
    args = parser.parse_args()
    if args.timeout <= 0 or args.resolution <= 0 or args.max_prompt < 1024:
        parser.error("timeout/resolution must be positive; max-prompt must be >= 1024")
    if not 1 <= args.start_prompt <= args.max_prompt:
        parser.error("start-prompt must be between 1 and max-prompt")
    # Do not interfere with another GPU workload or reuse stale success reports.
    active = subprocess.check_output(["nvidia-smi", "-i", "0", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
    if active:
        parser.error("GPU already has compute processes; refusing to start")
    args.output.mkdir(parents=False, exist_ok=False)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0")
    env["LD_LIBRARY_PATH"] = "/usr/local/cuda-12.8/lib64:" + env.get("LD_LIBRARY_PATH", "")
    state = json.loads((args.adapter / "training-state.json").read_text())
    source = Path(state["source_checkpoint"])
    summary = {
        "format_version": 1,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "method": "Fresh process per trial; eager prefill then cached single-token decode. Synthetic repeated-token probes ignore EOS, not quality tests. Only CUDA OOM brackets capacity; timeout/other failures are inconclusive.",
        "gpu": subprocess.check_output(["nvidia-smi", "-i", "0", "--query-gpu=name,memory.total,driver_version", "--format=csv"], text=True),
        "memory_convention": "Whole-device memory.used sampled every 0.5 seconds; not instantaneous allocation peak.",
        "provenance_sha256": {},
        "trials": [],
    }
    for path in [args.binary, Path(__file__), args.adapter / "adapter.safetensors", args.adapter / "training-state.json", source / "config.json", source / "tokenizer.json"]:
        summary["provenance_sha256"][str(path)] = sha256(path)
    manifest = source.parent / "checkpoint-manifest.json"
    provenance = json.loads(manifest.read_text())
    if provenance["model_id"] != "google/gemma-4-31B-it" or provenance["revision"] != "842da3794eaa0b77d5f08bae87a17459d91ff475":
        raise SystemExit("Base manifest model/revision mismatch")
    index = source / "model.safetensors.index.json"
    summary["provenance_sha256"][str(index)] = sha256(index)
    for shard in set(json.loads(index.read_text())["weight_map"].values()):
        if Path(shard).name != shard:
            raise SystemExit("Invalid shard filename")
        actual = sha256(source / shard)
        if actual != provenance["sha256"][shard]:
            raise SystemExit(f"Base shard hash mismatch: {shard}")
        summary["provenance_sha256"][str(source / shard)] = actual
    summary["base_checkpoint_manifest"] = provenance
    summary["base_shards_verified_against_local_manifest"] = True

    def save():
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    def trial(name, prompt, extra):
        report = args.output / (name + ".json")
        command = [str(args.binary), "--model-id", "google/gemma-4-31B-it", "--revision", "842da3794eaa0b77d5f08bae87a17459d91ff475", "--adapter-checkpoint", str(args.adapter), "--prompt", prompt, "--repeat-penalty", "1", "--report-file", str(report), *extra]
        start = time.monotonic()
        samples = []
        timed_out = False
        with (args.output / (name + ".log")).open("w") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
            try:
                while process.poll() is None:
                    used = subprocess.check_output(["nvidia-smi", "-i", "0", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True)
                    samples.append([round(time.monotonic() - start, 3), int(used.strip())])
                    if time.monotonic() - start > args.timeout:
                        timed_out = True
                        process.kill()
                        break
                    time.sleep(0.5)
            finally:
                if process.poll() is None:
                    process.kill()
                process.wait()
        log_text = (args.output / (name + ".log")).read_text()
        oom = "CUDA_ERROR_OUT_OF_MEMORY" in log_text
        status = "timeout" if timed_out else "success" if process.returncode == 0 and report.exists() else "cuda_oom" if oom else "error"
        item = {"name": name, "command": command, "status": status, "exit_code": process.returncode, "wall_seconds": time.monotonic() - start, "sampled_max_gpu_mib": max((s[1] for s in samples), default=None)}
        if status == "success":
            result = json.loads(report.read_text())
            if "--probe-prompt-tokens" in extra:
                expected = int(extra[extra.index("--probe-prompt-tokens") + 1])
                valid = (
                    result.get("status") == "success"
                    and result.get("prompt_tokens") == expected
                    and result.get("generated_tokens") == 2
                    and result.get("forwarded_tokens") == expected + 1
                    and result.get("max_forwarded_position_zero_based") == expected
                    and result.get("runtime_context") == expected + 2
                    and result.get("cached_decode_tested") is True
                )
                if not valid:
                    status = item["status"] = "error"
                    item["error_excerpt"] = "Probe report failed exact token/context/decode validation"
            item.update({key: result.get(key) for key in ["prompt_tokens", "generated_tokens", "forwarded_tokens", "max_forwarded_position_zero_based", "timings_seconds", "output_text", "runtime_context", "declared_context"]})
        else:
            item["error_excerpt"] = log_text[-4000:]
        (args.output / (name + "-gpu.json")).write_text(json.dumps(samples) + "\n")
        summary["trials"].append(item)
        save()
        print(json.dumps(item), flush=True)
        return status

    # Use a raw training-style prefix as well as a normal held-out instruction.
    for name, prompt, extra in [
        ("generation-training-prefix", "<bos>the small red cat", ["--raw-prompt"]),
        ("generation-instruction", "Explain in two sentences why the sky appears blue.", []),
    ]:
        if trial(name, prompt, ["--max-context", "1024", "--sample-len", "64", *extra]) != "success":
            raise SystemExit("Generation failed; see saved evidence")

    def probe(length, prefix="probe"):
        # P+2 output budget, P+1 positions actually forwarded: final sample is not fed back.
        return trial(f"{prefix}-{length}", "<bos>The quick brown fox jumps over the lazy dog. ", ["--raw-prompt", "--max-context", str(length + 2), "--probe-prompt-tokens", str(length), "--sample-len", "2", "--ignore-eos"])

    lower, upper = 0, None
    length = args.start_prompt
    while True:
        status = probe(length)
        if status == "success":
            lower = length
            if length == args.max_prompt:
                break
            length = min(length * 2, args.max_prompt)
        else:
            if status == "cuda_oom":
                upper = length
            break
    while upper is not None and upper - lower > args.resolution:
        mid = ((lower + upper) // (2 * args.resolution)) * args.resolution
        if mid <= lower:
            break
        status = probe(mid)
        if status == "success":
            lower = mid
        elif status == "cuda_oom":
            upper = mid
        else:
            break
    summary["largest_successful_prompt_tokens"] = lower or None
    summary["smallest_oom_prompt_tokens"] = upper
    summary["resolution_tokens"] = args.resolution
    summary["observed_bracket_width_tokens"] = upper - lower if upper is not None and lower else None
    if args.confirm_boundary:
        confirmation = {}
        if lower:
            confirmation["success_boundary_status"] = probe(lower, "confirm-success")
        if upper is not None:
            confirmation["oom_boundary_status"] = probe(upper, "confirm-oom")
        confirmation["confirmed_interval"] = (
            confirmation.get("success_boundary_status") == "success"
            and confirmation.get("oom_boundary_status") == "cuda_oom"
        )
        summary["boundary_confirmation"] = confirmation
    summary["conclusion"] = "Observed capacity only, not an architectural maximum or retrieval-quality guarantee; runtime cap changes per trial without changing RoPE frequencies."
    summary["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save()


if __name__ == "__main__":
    main()
