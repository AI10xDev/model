#!/usr/bin/env bash
set -euo pipefail
root="$(dirname "$(readlink -f "$0")")"
export LD_LIBRARY_PATH="/usr/local/cuda-12.8/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
weights=$(jq -r --arg root "$root/checkpoint/" \
  '[.weight_map[]] | unique | map($root + .) | join(",")' \
  "$root/checkpoint/model.safetensors.index.json")
exec "$root/../candle/target/release/examples/gemma4" \
  --model-id google/gemma-4-31B-it \
  --revision 842da3794eaa0b77d5f08bae87a17459d91ff475 \
  --config-file "$root/config-4096.json" \
  --tokenizer-file "$root/checkpoint/tokenizer.json" \
  --weight-files "$weights" \
  --sample-len "${SAMPLE_LEN:-64}" \
  --prompt "${1:?Usage: run-gemma4.sh 'prompt' [additional sampling options]}" \
  "${@:2}"
