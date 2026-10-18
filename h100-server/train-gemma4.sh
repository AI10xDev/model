#!/usr/bin/env bash
# All-layer LoRA on the full-sized frozen Q4 text decoder, not full-parameter training.
set -euo pipefail
[[ $(hostname) == gemma-h100-aue ]]
if [[ $# != 2 ]]; then
    printf 'Usage: bash train-gemma4.sh INPUT_JSONL NEW_OUTPUT_DIRECTORY\n' >&2
    exit 2
fi
input=$(realpath "$1")
output=$(realpath -m "$2")
[[ -f "$input" && ! -e "$output" && -d $(dirname "$output") ]]
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH="/usr/local/cuda-12.8/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
binary=/home/azureuser/candle/target/release/examples/gemma4-training
checkpoint=/home/azureuser/gemma4/checkpoint
[[ $(nvidia-smi --query-gpu=name --format=csv,noheader -i 0) == *H100* ]]
mkdir "$output"
nvidia-smi --query-gpu=timestamp,index,name,memory.used,utilization.gpu --format=csv -i 0 -l 1 > "$output/gpu.csv" &
monitor=$!
trap 'kill "$monitor" 2>/dev/null || true; wait "$monitor" 2>/dev/null || true' EXIT
sha256sum "$binary" "$input" "$checkpoint/config.json" > "$output/provenance.sha256"
cp /home/azureuser/gemma4/checkpoint-manifest.json "$output/base-checkpoint-manifest.json"
"$binary" preprocess \
    --input "$input" --tokenizer "$checkpoint/tokenizer.json" \
    --output "$output/data" --sequence-length "${SEQUENCE_LENGTH:-32}" \
    --vocab-size 262144 --validation-fraction 0.25 --seed 42 \
    --bos-token-id 2 --eos-token-id 1 2>&1 | tee "$output/preprocess.log"
/usr/bin/time -v "$binary" train \
    --config "$checkpoint/config.json" --checkpoint "$checkpoint" \
    --dataset "$output/data" --output "$output/run" \
    --mode "${MODE:-quantized-lora}" --quantization q4-0 --precision bf16 \
    --lora-rank 8 --lora-alpha 8 --max-parameters 32000000000 \
    --steps "${STEPS:-20}" --batch-size 1 --learning-rate "${LEARNING_RATE:-0.0001}" \
    --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS:-1}" \
    --weight-decay 0 --eval-every 10 --eval-batches 2 --save-every 10 \
    --cuda 2>&1 | tee "$output/train.log"
