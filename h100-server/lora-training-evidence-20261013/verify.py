"""Read-only checks for this bounded H100 run; no ML/Python packages required."""

import csv
import hashlib
import json
import math
from pathlib import Path
import struct
import sys


def adapter(path):
    hashes = {}
    parameters = 0
    with path.open("rb") as stream:
        length = struct.unpack("<Q", stream.read(8))[0]
        header = json.loads(stream.read(length))
        for name, tensor in header.items():
            if name == "__metadata__":
                continue
            assert tensor["dtype"] == "F32", name
            assert name.startswith("layers.") and name.endswith((".lora_a", ".lora_b")), name
            start, end = tensor["data_offsets"]
            count = math.prod(tensor["shape"])
            assert end - start == count * 4, name
            stream.seek(8 + length + start)
            data = stream.read(end - start)
            assert len(data) == end - start, name
            assert all(math.isfinite(v) for (v,) in struct.iter_unpack("<f", data)), name
            hashes[name] = hashlib.sha256(data).hexdigest()
            parameters += count
    return hashes, parameters


root = Path(sys.argv[1])
run = root / "run"
state = json.loads((run / "checkpoint-20/training-state.json").read_text())
metrics = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
assert state["mode"] == "quantized-lora"
assert state["parameters"] == 30697345340
assert state["quantization"] == "Q4_0" and state["dtype"] == "bf16"
assert state["step"] == 20 and state["tokens_seen"] == 2560
assert state["gradient_accumulation_steps"] == 4 and state["batch_size"] == 1
assert len(metrics) == 21
for index, metric in enumerate(metrics):
    assert metric["step"] == index
    assert all(math.isfinite(value) for value in metric.values())
before, count = adapter(run / "checkpoint-10/adapter.safetensors")
after, final_count = adapter(run / "checkpoint-20/adapter.safetensors")
assert count == final_count == state["trainable_parameters"]
assert before.keys() == after.keys()
assert len(after) == 2 * state["quantized_projections"] == 820
assert {int(name.split(".")[1]) for name in after} == set(range(60))
changed = sum(before[name] != digest for name, digest in after.items())
assert changed == len(after), "some adapters did not change between steps 10 and 20"
with (root / "gpu.csv").open() as stream:
    rows = csv.DictReader(stream)
    memory_key = next(key for key in rows.fieldnames if key.strip().startswith("memory.used"))
    memory = [int(row[memory_key].strip().split()[0]) for row in rows]
summary = {
    "text_parameters": state["parameters"],
    "trainable_parameters": count,
    "adapter_tensors": len(after),
    "adapter_tensors_changed_steps_10_to_20": changed,
    "layers": 60,
    "steps": 20,
    "tokens_seen": state["tokens_seen"],
    "initial_validation_loss": metrics[0]["validation_loss"],
    "final_validation_loss": metrics[-1]["validation_loss"],
    "sampled_max_gpu_mib": max(memory),
    "memory_sample_count": len(memory),
}
warm = root / "warm-start"
if warm.exists():
    warm_metrics = [json.loads(line) for line in (warm / "metrics.jsonl").read_text().splitlines()]
    assert len(warm_metrics) == 2 and warm_metrics[-1]["step"] == 1
    assert abs(warm_metrics[0]["validation_loss"] - metrics[-1]["validation_loss"]) < 1e-6
    restarted, warm_count = adapter(warm / "checkpoint-1/adapter.safetensors")
    assert restarted.keys() == after.keys() and warm_count == count
    assert all(restarted[name] != after[name] for name in after)
    summary["warm_start_initial_validation_loss"] = warm_metrics[0]["validation_loss"]
    summary["warm_start_final_validation_loss"] = warm_metrics[-1]["validation_loss"]
print(json.dumps(summary, indent=2))
