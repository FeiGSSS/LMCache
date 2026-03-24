#!/bin/bash
set -euo pipefail
source .venv/bin/activate

usage() {
    cat <<'EOF'
Usage: benchmarks/schedule/bench.sh <scenario> [--num-requests N]

Scenarios:
  cpu               uses http://localhost:8001
  cpu_disk          uses http://localhost:8002
  cpu_disk_tiering  uses http://localhost:8003
EOF
    exit 1
}

if [[ $# -lt 1 ]]; then
    usage
fi

SCENARIO="$1"
shift

NUM_REQUESTS=7000

while [[ $# -gt 0 ]]; do
    case "$1" in
        --num-requests)
            if [[ $# -lt 2 ]]; then
                usage
            fi
            NUM_REQUESTS="$2"
            shift 2
            ;;
        *)
            usage
            ;;
    esac
done

MODEL_PATH="/data/llm-models/Qwen3-8B"
MODEL_NAME="Qwen3-8B"
INPUT_FILE="benchmarks/schedule/dataset/sharegpt_conv_all_min5.json"
RESULTS_DIR="benchmarks/schedule/results"

case "$SCENARIO" in
    cpu)
        URL="http://localhost:8001"
        OUTPUT_FILE="$RESULTS_DIR/tiering_ttft_cpu_${NUM_REQUESTS}.json"
        ;;
    cpu_disk)
        URL="http://localhost:8002"
        OUTPUT_FILE="$RESULTS_DIR/tiering_ttft_cpu_disk_${NUM_REQUESTS}.json"
        ;;
    cpu_disk_tiering)
        URL="http://localhost:8003"
        OUTPUT_FILE="$RESULTS_DIR/tiering_ttft_cpu_disk_tiering_${NUM_REQUESTS}.json"
        ;;
    *)
        usage
        ;;
esac

if [[ ! -f "$INPUT_FILE" ]]; then
    echo "Missing input dataset: $INPUT_FILE" >&2
    exit 1
fi

mkdir -p "$RESULTS_DIR"

echo "Running benchmark scenario=$SCENARIO requests=$NUM_REQUESTS"
echo "Target URL: $URL"
echo "Output file: $OUTPUT_FILE"

exec python3 benchmarks/schedule/tiering_ttft_bench.py \
    --model "$MODEL_PATH" \
    --served-model-name "$MODEL_NAME" \
    --url "$URL" \
    --input-file "$INPUT_FILE" \
    --num-users 128 \
    --conversations-per-user 10 \
    --min-user-turns 5 \
    --max-parallel 32 \
    --request-rate-per-user 1 \
    --continue-prob 0.8 \
    --seed 42 \
    --max-num-requests "$NUM_REQUESTS" \
    --output-file "$OUTPUT_FILE"
