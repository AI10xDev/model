#!/usr/bin/env bash
set -euo pipefail

cloud-init status --wait
nvidia-smi
runuser -u azureuser -- python3 -m venv /home/azureuser/gpu-venv
runuser -u azureuser -- /home/azureuser/gpu-venv/bin/pip install --disable-pip-version-check torch --index-url https://download.pytorch.org/whl/cu128
runuser -u azureuser -- /home/azureuser/gpu-venv/bin/python - <<'PY'
import json
import torch

assert torch.cuda.is_available(), "CUDA is not available"
assert torch.cuda.device_count() == 1, "Expected one GPU"
gpu = torch.cuda.get_device_properties(0)
assert "H100" in gpu.name, gpu.name
assert torch.cuda.is_bf16_supported(), "BF16 is not supported"

# Exercise the weight-memory footprint of a 31B-parameter BF16 model.
weights = torch.empty(31_000_000_000, dtype=torch.bfloat16, device="cuda")
weights.zero_()
torch.cuda.synchronize()
allocated_bytes = torch.cuda.memory_allocated()
del weights
torch.cuda.empty_cache()

x = torch.ones((256, 256), dtype=torch.bfloat16, device="cuda")
y = x @ x
torch.cuda.synchronize()
assert torch.all(y == 256).item(), "GPU matrix multiplication failed"
print(json.dumps({
    "gpu": gpu.name,
    "vram_bytes": gpu.total_memory,
    "torch": torch.__version__,
    "cuda_runtime": torch.version.cuda,
    "weight_allocation_bytes": allocated_bytes,
    "bf16_matmul": "passed",
    "note": "Hardware smoke test only; no model downloaded or inference tested.",
}, indent=2))
PY
