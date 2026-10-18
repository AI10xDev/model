#!/usr/bin/env python3
"""Download and verify the pinned checkpoint, then create a bounded runtime config."""

import hashlib
import json
import socket
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

MODEL_ID = "google/gemma-4-31B-it"
REVISION = "842da3794eaa0b77d5f08bae87a17459d91ff475"


def main():
    if socket.gethostname() != "gemma-h100-aue":
        raise SystemExit("This installer targets gemma-h100-aue only")
    root = Path.home() / "gemma4"
    checkpoint = root / "checkpoint"
    info = HfApi().model_info(MODEL_ID, revision=REVISION, files_metadata=True)
    if info.sha != REVISION:
        raise RuntimeError("Hub returned a different checkpoint revision")
    snapshot_download(
        MODEL_ID,
        revision=REVISION,
        local_dir=checkpoint,
        allow_patterns=["*.json", "*.safetensors", "*.jinja", "README.md"],
        max_workers=2,
    )
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())
    shards = sorted(set(index["weight_map"].values()))
    metadata = {entry.rfilename: entry for entry in info.siblings}
    hashes = {}
    for name in shards:
        path = checkpoint / name
        expected = metadata[name].lfs
        if expected is None:
            raise RuntimeError(f"No SHA256 metadata for {name}")
        if path.stat().st_size != expected.size:
            raise RuntimeError(f"Wrong size: {name}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        hashes[name] = digest.hexdigest()
        if hashes[name] != expected.sha256:
            raise RuntimeError(f"SHA256 mismatch: {name}")
        print(f"Verified {name}: {hashes[name]}", flush=True)

    config = json.loads((checkpoint / "config.json").read_text())
    if config["model_type"] != "gemma4":
        raise RuntimeError("Expected a Gemma4 conditional generation checkpoint")
    original_context = config["text_config"]["max_position_embeddings"]
    if original_context < 4096:
        raise RuntimeError("Checkpoint context is smaller than the runtime limit")
    # Keep the downloaded config and RoPE frequencies unchanged.
    config["text_config"]["max_position_embeddings"] = 4096
    (root / "config-4096.json").write_text(json.dumps(config, indent=2) + "\n")
    (root / "checkpoint-manifest.json").write_text(
        json.dumps(
            {
                "model_id": MODEL_ID,
                "revision": REVISION,
                "original_context": original_context,
                "runtime_context": 4096,
                "sha256": hashes,
            },
            indent=2,
        ) + "\n"
    )
    print("Checkpoint verified; 4096-token runtime config ready.", flush=True)


if __name__ == "__main__":
    main()
