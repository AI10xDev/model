#!/usr/bin/env bash
set -euo pipefail
[[ $(hostname) == gemma-h100-aue ]]
[[ $PWD == /home/azureuser/candle ]]
export PATH="$HOME/.cargo/bin:/usr/local/cuda-12.8/bin:$PATH"
export CUDA_HOME=/usr/local/cuda-12.8
export CUDA_COMPUTE_CAP=90
export LD_LIBRARY_PATH="/usr/local/cuda-12.8/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CARGO_BUILD_JOBS=16
cargo +1.96.0 build --locked --release -p candle-examples --example gemma4-training --features cuda
cargo +1.96.0 test --locked --release -p candle-examples --example gemma4-training --features cuda
cargo +1.96.0 test --locked --release -p candle-transformers --test gemma4_lora_tests --test gemma4_quantized_decoder_tests --features cuda
target/release/examples/gemma4-training train --help
